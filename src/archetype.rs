//! [`Archetype`] and related items.

use alloc::alloc::{alloc, dealloc, handle_alloc_error, realloc};
use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec, vec::Vec};
use core::alloc::Layout;
use core::hash::Hash;
use core::mem::transmute;
use core::ops::Index;
use core::panic::{RefUnwindSafe, UnwindSafe};
use core::ptr::NonNull;
use core::{fmt, mem, ptr, slice};

use ahash::RandomState;
use slab::Slab;

use crate::assume_unchecked;
use crate::component::{ComponentIdx, ComponentInfo, Components};
use crate::component_indices::{ComponentIndices, ComponentIndicesPtr};
use crate::drop::DropFn;
use crate::entity::{Entities, EntityId, EntityLocation};
use crate::event::{EventId, EventPtr, TargetedEventIdx};
use crate::handler::{
    HandlerConfig, HandlerInfo, HandlerInfoPtr, HandlerList, HandlerParam, Handlers, InitError,
};
use crate::map::{Entry, HashMap};
use crate::prelude::World;
use crate::sparse::SparseIndex;
use crate::sparse_map::SparseMap;
use crate::world::UnsafeWorldCell;

/// Contains all the [`Archetype`]s and their metadata for a world.
///
/// This can be obtained in a handler by using the `&Archetypes` handler
/// parameter.
///
/// ```
/// # use evenio::prelude::*;
/// # use evenio::archetype::Archetypes;
/// #
/// # #[derive(GlobalEvent)] struct E;
/// #
/// # let mut world = World::new();
/// world.add_handler(|_: Receiver<E>, archetypes: &Archetypes| {});
/// ```
#[derive(Debug)]
pub struct Archetypes {
    /// Always contains the empty archetype at index 0.
    archetypes: Slab<Archetype>,
    by_components: HashMap<ComponentIndicesPtr, ArchetypeIdx>,
}

impl Archetypes {
    /// Constructs a new `Archetypes` instance, which contains only the empty
    /// archetype.
    pub(crate) fn new() -> Self {
        let empty_arch = Archetype::empty();

        // SAFETY: We only remove archetypes in `remove_component`, where we
        // first remove the `by_components` entry (dropping the component
        // indices pointer) and *then* drop the archetype that owns the
        // component indices.
        let component_indices_ptr = unsafe { empty_arch.component_indices.as_ptr() };

        let mut by_components = HashMap::with_hasher(RandomState::new());
        by_components.insert(component_indices_ptr, ArchetypeIdx::EMPTY);

        Self {
            archetypes: Slab::from_iter([(0, empty_arch)]),
            by_components,
        }
    }

    /// Returns a reference to the empty archetype (The archetype with no
    /// components).
    ///
    /// The empty archetype is always present, so this function is infallible.
    pub fn empty(&self) -> &Archetype {
        // SAFETY: The empty archetype is always at index 0.
        unsafe { self.archetypes.get(0).unwrap_unchecked() }
    }

    /// Returns a mutable reference to the empty archetype.
    pub(crate) fn empty_mut(&mut self) -> &mut Archetype {
        // SAFETY: The empty archetype is always at index 0.
        unsafe { self.archetypes.get_mut(0).unwrap_unchecked() }
    }

    /// Gets a reference to the archetype identified by the given
    /// [`ArchetypeIdx`]. Returns `None` if the index is invalid.
    pub fn get(&self, idx: ArchetypeIdx) -> Option<&Archetype> {
        self.archetypes.get(idx.0 as usize)
    }

    /// Gets a reference to the archetype with the given set of components.
    ///
    /// Returns `None` if there is no archetype with the given set of
    /// components or the given [`ComponentIdx`] slice is not sorted and
    /// deduplicated.
    pub fn get_by_components(&self, components: &[ComponentIdx]) -> Option<&Archetype> {
        let idx = *self.by_components.get(components)?;
        Some(unsafe { self.get(idx).unwrap_unchecked() })
    }

    /// Spawns a new entity into the empty archetype with the given ID and
    /// returns its location.
    pub(crate) fn spawn(&mut self, id: EntityId) -> EntityLocation {
        let empty = self.empty_mut();

        // Reserve space for the spawned entity.
        let reallocated = unsafe { empty.reserve_one() };

        // Add the entity to the empty archetype. This does not involve adding
        // components, as the empty archetype does not have columns. Only the
        // spawned entity's id needs to be stored.
        let row = ArchetypeRow(empty.entity_count());
        empty.entity_ids.push(id);

        // If the archetype was empty or has been reallocated, notify the
        // listening handlers of this change.
        if empty.entity_count() == 1 || reallocated {
            for mut ptr in empty.refresh_listeners.iter().copied() {
                unsafe { ptr.as_info_mut().handler_mut().refresh_archetype(empty) };
            }
        }

        // Construct the entity's location.
        EntityLocation {
            archetype: ArchetypeIdx::EMPTY,
            row,
        }
    }

    /// Returns an iterator over all archetypes in an arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = &Archetype> {
        self.archetypes.iter().map(|(_, v)| v)
    }

    /// Returns a count of the archetypes.
    pub fn len(&self) -> usize {
        self.archetypes.len()
    }

    /// Registers an event handler for all archetypes.
    pub(crate) fn register_handler(&mut self, info: &mut HandlerInfo) {
        // TODO: use a `Component -> Vec<Archetype>` index to make this faster?
        for (_, arch) in &mut self.archetypes {
            arch.register_handler(info);
        }
    }

    /// Removes an event handler from all archetypes.
    pub(crate) fn remove_handler(&mut self, info: &HandlerInfo) {
        // TODO: use a `Component -> Vec<Archetype>` index to make this faster?
        for (_, arch) in &mut self.archetypes {
            // TODO: remove_handler method for Archetype for this?
            arch.refresh_listeners.remove(&info.ptr());

            if let EventId::Targeted(id) = info.received_event() {
                if let Some(list) = arch.event_listeners.get_mut(id.index()) {
                    list.remove(info.ptr());
                }
            }
        }
    }

    /// Removes a component. This removes all archetypes that have this
    /// component and calls `removed_entity_callback` on all their entities.
    pub(crate) fn remove_component<F>(
        &mut self,
        info: &mut ComponentInfo,
        components: &mut Components,
        mut removed_entity_callback: F,
    ) where
        F: FnMut(EntityId),
    {
        let removed_component_id = info.id();

        // Iterate over all archetypes that have the removed component as one of
        // their columns.
        for arch_idx in info.member_of.drain(..) {
            // Remove the archetype.
            let arch = self.archetypes.remove(arch_idx.0 as usize);

            // Notify all handlers listening for updates affecting the archetype
            // that it was removed.
            for mut ptr in arch.refresh_listeners.iter().copied() {
                unsafe { ptr.as_info_mut().handler_mut().remove_archetype(&arch) };
            }

            // Remove the archetype from the `member_of` lists of its
            // components.
            for &comp_idx in &**arch.component_indices() {
                if comp_idx != removed_component_id.index() {
                    let info = unsafe { components.get_by_index_mut(comp_idx).unwrap_unchecked() };
                    info.member_of.swap_remove(&arch_idx);
                }
            }

            // NOTE: Using plain `.remove()` here makes Miri sad.
            // SAFETY: The component indices pointer created is a temporary
            // value that is dropped before the archetype is dropped at the end
            // of the loop body.
            self.by_components
                .remove_entry(&unsafe { arch.component_indices.as_ptr() });

            // Call the removed entity callback for each entity id in the
            // archetype.
            for &entity_id in arch.entity_ids() {
                removed_entity_callback(entity_id);
            }
        }
    }

    /// Creates a new archetype with the given component indices and returns its
    /// index. If an archetype with the given component indices already exists,
    /// its index is returned and no archetype is created.
    ///
    /// # Safety
    ///
    /// Component indices must be valid (i.e. their respective components must
    /// exist).
    pub(crate) unsafe fn create_archetype(
        &mut self,
        component_indices: ComponentIndices,
        components: &mut Components,
        handlers: &mut Handlers,
    ) -> ArchetypeIdx {
        // SAFETY: We only remove archetypes in `remove_component`, where we
        // first remove the `by_components` entry (dropping the component
        // indices pointer) and *then* drop the archetype that owns the
        // component indices.
        let component_indices_ptr = component_indices.as_ptr();

        let archetypes_entry = self.archetypes.vacant_entry();
        let by_components_entry = match self.by_components.entry(component_indices_ptr) {
            Entry::Occupied(entry) => return *entry.get(),
            Entry::Vacant(entry) => entry,
        };

        let arch_idx = ArchetypeIdx(archetypes_entry.key() as u32);

        // Construct the new archetype.
        // SAFETY: Caller guarantees component indices are valid.
        let mut arch = Archetype::new(arch_idx, component_indices, components);

        // Register all event handlers for the new archetype.
        for info in handlers.iter_mut() {
            arch.register_handler(info);
        }

        // Insert the archetype into the vacant entries and
        // our archetype list.
        by_components_entry.insert(arch_idx);
        archetypes_entry.insert(arch);

        arch_idx
    }

    /// Moves an entity from one archetype to another. Returns the entity's row
    /// in the new archetype.
    ///
    /// - For each column present only in the source archetype, the entity's
    ///   component will be removed.
    /// - For each column present in both archetypes, the entity's component
    ///   will be moved, unless a new component for the column's component index
    ///   was provided. Then, the old component will be removed and the new
    ///   component will be added to the destination column
    /// - For each column present only in the destination archetype, a new
    ///   component of the column's component index must be provided and will be
    ///   inserted into the column.
    ///
    /// New components can be provided via the `new_component_indices` and
    /// `new_component_pointers` parameters. The parameters must have the same
    /// length, and at each index `i`, `new_component_pointers[i]` must point
    /// to a valid component for the component index `new_component_indices[i]`.
    ///
    /// # Safety
    ///
    /// - `src` must be valid.
    /// - `dst` must be valid.
    /// - `new_component_indices` and `new_component_pointers` must fulfill the
    ///   requirements stated above.
    pub(crate) unsafe fn move_entity(
        &mut self,
        src: EntityLocation,
        dst: ArchetypeIdx,
        new_component_indices: &ComponentIndices,
        new_component_pointers: &[*const u8],
        entities: &mut Entities,
    ) -> ArchetypeRow {
        // If the source and destination archetypes are equal, the entity does
        // not need to be moved. Instead, we reassign the components.
        if src.archetype == dst {
            let arch = self
                .archetypes
                .get_mut(src.archetype.0 as usize)
                .unwrap_unchecked();

            for (&comp_idx, &comp_ptr) in new_component_indices.iter().zip(new_component_pointers) {
                let col = arch.column_of_mut(comp_idx).unwrap_unchecked();

                // Replace the old component with the new component of the same
                // type.
                col.assign(src.row.0 as usize, comp_ptr);
            }

            return src.row;
        }

        let (src_arch, dst_arch) = self
            .archetypes
            .get2_mut(src.archetype.0 as usize, dst.0 as usize)
            .unwrap_unchecked();

        let dst_row = ArchetypeRow(dst_arch.entity_ids.len() as u32);

        // Reserve space for the moved entity in the destination archetype.
        let dst_arch_reallocated = dst_arch.reserve_one();

        // TODO: Remove these old and drafty notes after a commit.
        // Update components of each component index:
        // - If the source has a column for the index, but the destination does not,
        //   remove the moved entity's component from that column.
        // - If both the source and the destination have a column for the index,
        //   transfer the moved entity's component from the source column to the
        //   destination column.
        // - If the source does not have a column for the index, but the destination
        //   does, insert a new component from the `new_components` iterator into the
        //   destination column.
        //
        // The algorithm used to achieve this for every relevant component index
        // uses a loop, in which two indices, `src_idx` and `dst_idx`, are
        // gradually incremented. These are indices into simultaneously the
        // `component_indices` and `columns` slices of the source and
        // destination archetypes, respectively. To differentiate them from the
        // actual component indices, we will call them column indices.
        //
        // It might be best to explain the algorithm with an exemplary
        // walkthrough:
        // The world contains three components: A, B and C, with component
        // indices 0, 1 and 2. We are moving an entity from the archetype 'AB'
        // (with columns for A and B components) to the archetype 'BC'. We need
        // to (1) remove A, (2) move B and (3) insert C.
        //
        // But first, this is what our archetypes look like:
        //
        // src_arch: 'AB'
        // | src_idx | src_comp_idx |
        // |    0    |     0 (A)    |
        // |    1    |     1 (B)    |
        //
        // dst_arch: 'BC'
        // | dst_idx | dst_comp_idx |
        // |    0    |     1 (B)    |
        // |    1    |     2 (C)    |
        //
        // We start with `src_idx` and `dst_idx` both at zero. The corresponding
        // component indices are those of A and B (cf. the first rows of the
        // tables above).
        //
        // The `component_indices` slices are guaranteed to be sorted and free
        // from duplicate elements, which has useful implications for us. For
        // some column index `i` and component index `x` at `i`, we know for
        // every component index `y` appearing after it in the slice that `y` is
        // greater than `x`. This means that if we have another component index
        // `z` which we know to be less than `x`, we know that it will not
        // appear at any point in the slice at or after `x`.
        //
        // We can use this property here. We can see that the component index of
        // A is less than that of B. This means that the component index of A
        // does not appear anywhere in the component indices of the destination
        // archetype, so we have to remove component A.
        //
        // Whew, goal number 1 achieved!
        //
        // With this, we have finished processing `src_idx` zero, but not
        // `dst_idx` zero - we have neither transferred nor inserted a component
        // into the B-column of the destination archetype. Therefore, we will
        // increment `src_idx`, but not `dst_idx`.
        //
        // In the next iteration of the loop, we have `src_idx` at one and
        // `dst_idx` at zero, corresponding to the component indices of B and,
        // well, B (cf. the tables above).
        //
        // We have found a pair of matching columns, so we will transfer the
        // component B from the source over to the destination.
        //
        // Goal number 2 done!
        //
        // With this transfer operation, we have completed concluded all work we
        // need to do with `src_idx` one and `dst_idx` zero, so we increment
        // both and move to the next iteration.
        //
        // We now have `src_idx` at two and `dst_idx` at one. This means that
        // `src_idx` is out of bounds, so we have no source column to remove or
        // transfer components from. The only other option is to insert a new
        // component, C, into the remaining column of the destination archetype.
        //
        // With that, we have finally achieved all of our goals.
        //
        // Since `src_idx` is already out of bounds, there is no use in
        // incrementing it; we are, however, finished with processing `dst_idx`
        // one, so we increment that. This leaves us with both column indices
        // out of bounds in the next iteration, so we break the loop.
        /*

        ABC -> BCD with CD

        src A B C
        dst B C D
        new C D

        src <<, removing A

        src B C
        dst B C D
        new C D

        src, dst <<, moving B

        src C
        dst C D
        new C D

        src, dst, new <<, replacing C

        src -
        dst D
        new D

        dst, new <<, adding D

        src -
        dst -
        new -

        done


        patterns:

        src X
        dst Y
        new *

        => remove X, shift src

        src X
        dst X
        new Y

        => move X, shift src, dst

        src X
        dst X
        new X

        => replace X, shift src, dst, new

        src Y
        dst X
        new X

        => add X, shift dst, new

        src Y
        dst X
        new Y

        => error

        src Y
        dst X
        new Z

        => error


        patterns (variant):

        src < dst

        => remove X, shift src

        src == dst < new

        => move X, shift src, dst

        src == dst == new

        => replace X, shift src, dst, new

        src == dst > new

        => error: excess new

        src > dst > new

        => error: excess new

        src > dst == new

        => add X, shift dst, new

        src > dst < new

        => error: missing new

         */

        mod handle_components {
            //! Removing, transferring, replacing and adding components while
            //! moving an entity is a complicated process. This unusual little
            //! module embedded in a function is intended to bring structure to
            //! the algorithm.
            // TODO: Add a detailed explanation and more documentation below.

            use core::cmp::Ordering::*;

            use NewComponentState::*;
            use OldComponentState::*;
            use ProcessState::*;

            use crate::component_indices::ComponentIndices;

            /// Information and recommendations for action based on a comparison
            /// of the component indices of the current source and
            /// destination columns.
            pub(super) enum OldComponentState {
                /// Because we will not find a matching destination column to
                /// transfer the old component to or to add a new component to.
                RemoveFromSource,
                /// Because the source and destination columns store components
                /// of the same type.
                TransferOrReplace,
                /// Because we will not find a matching source column to
                /// transfer an old component from.
                OldComponentUnavailable,
            }

            /// Information and recommendations for action based on a comparison
            /// of the component indices of the current destination
            /// column and the current new component pointer.
            pub(super) enum NewComponentState {
                /// Because we will not find a new component we can place into
                /// this destination column.
                NewComponentUnavailable,
                /// Because the new component can be placed into the destination
                /// column.
                AddNew,
                /// Because we will not find a destination column we can place
                /// this new component into.
                ExcessNew,
            }

            /// The state of a [`Process`].
            pub(super) enum ProcessState {
                /// The process is in progress.
                InProgress(OldComponentState, NewComponentState),
                /// The process is finished, but there are new components left
                /// over.
                DoneWithExcessNew,
                /// The process is finished.
                Done,
            }

            pub(super) struct Process<'a> {
                src_idx: usize,
                dst_idx: usize,
                new_idx: usize,
                src_component_indices: &'a ComponentIndices,
                dst_component_indices: &'a ComponentIndices,
                new_component_indices: &'a ComponentIndices,
            }

            impl<'a> Process<'a> {
                #[inline]
                pub(super) fn new(
                    src_component_indices: &'a ComponentIndices,
                    dst_component_indices: &'a ComponentIndices,
                    new_component_indices: &'a ComponentIndices,
                ) -> Self {
                    Self {
                        src_idx: 0,
                        dst_idx: 0,
                        new_idx: 0,
                        src_component_indices,
                        dst_component_indices,
                        new_component_indices,
                    }
                }

                #[inline]
                pub(super) fn state(&self) -> ProcessState {
                    let src_component_index = self.src_component_indices.get(self.src_idx).copied();
                    let dst_component_index = self.dst_component_indices.get(self.dst_idx).copied();
                    let new_component_index = self.new_component_indices.get(self.new_idx).copied();

                    match (
                        src_component_index,
                        dst_component_index,
                        new_component_index,
                    ) {
                        (
                            Some(src_component_index),
                            Some(dst_component_index),
                            Some(new_component_index),
                        ) => InProgress(
                            match src_component_index.cmp(&dst_component_index) {
                                Less => RemoveFromSource,
                                Equal => TransferOrReplace,
                                Greater => OldComponentUnavailable,
                            },
                            match dst_component_index.cmp(&new_component_index) {
                                Less => NewComponentUnavailable,
                                Equal => AddNew,
                                Greater => ExcessNew,
                            },
                        ),
                        (Some(src_component_index), Some(dst_component_index), None) => InProgress(
                            match src_component_index.cmp(&dst_component_index) {
                                Less => RemoveFromSource,
                                Equal => TransferOrReplace,
                                Greater => OldComponentUnavailable,
                            },
                            NewComponentUnavailable,
                        ),
                        (Some(_), None, Some(_)) => InProgress(RemoveFromSource, ExcessNew),
                        (Some(_), None, None) => {
                            InProgress(RemoveFromSource, NewComponentUnavailable)
                        }
                        (None, Some(dst_component_index), Some(new_component_index)) => InProgress(
                            OldComponentUnavailable,
                            match dst_component_index.cmp(&new_component_index) {
                                Less => NewComponentUnavailable,
                                Equal => AddNew,
                                Greater => ExcessNew,
                            },
                        ),
                        (None, Some(_), None) => {
                            InProgress(OldComponentUnavailable, NewComponentUnavailable)
                        }
                        (None, None, Some(_)) => DoneWithExcessNew,
                        (None, None, None) => Done,
                    }
                }

                /// Should be called after the entity's component in the source
                /// column has been removed entirely or transferred to
                /// the destination column.
                #[inline]
                fn src_column_handled(&mut self) {
                    self.src_idx += 1;
                }

                /// Should be called after a component has been added to the
                /// destination column. This is either a new component
                /// or the entity's component from the source column.
                #[inline]
                fn dst_column_handled(&mut self) {
                    self.dst_idx += 1;
                }

                /// Should be called after a new component has been added to the
                /// destination column.
                #[inline]
                fn new_component_handled(&mut self) {
                    self.new_idx += 1;
                }

                /// Should be called after the entity's component has been
                /// removed from the source column.
                #[inline]
                pub(super) fn removed_component(&mut self) {
                    self.src_column_handled();
                }

                /// Should be called after the entity's component has been
                /// transferred from the source to the destination column.
                #[inline]
                pub(super) fn transferred_component(&mut self) {
                    self.src_column_handled();
                    self.dst_column_handled();
                }

                /// Should be called after the entity's old component in the
                /// source column has been removed and replaced
                /// with a new component added
                /// to the destination column.
                #[inline]
                pub(super) fn replaced_component(&mut self) {
                    self.removed_component();
                    self.added_component();
                }

                /// Should be called after a new component has been added to the
                /// destination column.
                #[inline]
                pub(super) fn added_component(&mut self) {
                    self.dst_column_handled();
                    self.new_component_handled();
                }

                #[inline]
                pub(super) fn src_idx(&self) -> usize {
                    self.src_idx
                }

                #[inline]
                pub(super) fn dst_idx(&self) -> usize {
                    self.dst_idx
                }

                #[inline]
                pub(super) fn new_idx(&self) -> usize {
                    self.new_idx
                }
            }
        }

        use handle_components::NewComponentState::*;
        use handle_components::OldComponentState::*;
        use handle_components::Process;
        use handle_components::ProcessState::*;

        let mut process = Process::new(
            src_arch.component_indices(),
            dst_arch.component_indices(),
            new_component_indices,
        );

        loop {
            match process.state() {
                InProgress(_, ExcessNew) | DoneWithExcessNew => {
                    panic!("move_entity called with excess new components");
                }
                InProgress(OldComponentUnavailable, NewComponentUnavailable) => {
                    panic!("move_entity called with new components missing");
                }
                InProgress(RemoveFromSource, _) => {
                    let src_col = &mut *src_arch.columns.as_ptr().add(process.src_idx());
                    src_col.swap_remove(src_arch.entity_ids.len(), src.row.0 as usize);

                    process.removed_component();
                }
                InProgress(TransferOrReplace, NewComponentUnavailable) => {
                    let src_col = &mut *src_arch.columns.as_ptr().add(process.src_idx());
                    let dst_col = &mut *dst_arch.columns.as_ptr().add(process.dst_idx());

                    src_col.transfer_elem(
                        src_arch.entity_ids.len(),
                        dst_col,
                        dst_arch.entity_ids.len(),
                        src.row.0 as usize,
                    );

                    process.transferred_component();
                }
                InProgress(TransferOrReplace, AddNew) => {
                    let src_col = &mut *src_arch.columns.as_ptr().add(process.src_idx());
                    let dst_col = &mut *dst_arch.columns.as_ptr().add(process.dst_idx());

                    src_col.swap_remove(src_arch.entity_ids.len(), src.row.0 as usize);

                    let new_component = new_component_pointers[process.new_idx()];

                    let dst_ptr = dst_col
                        .data
                        .as_ptr()
                        .add(dst_col.component_layout.size() * dst_arch.entity_ids.len());

                    // Insert the new component into the destination column.
                    ptr::copy_nonoverlapping(
                        new_component,
                        dst_ptr,
                        dst_col.component_layout.size(),
                    );

                    process.replaced_component();
                }
                InProgress(OldComponentUnavailable, AddNew) => {
                    let dst_col = &mut *dst_arch.columns.as_ptr().add(process.dst_idx());

                    let new_component = new_component_pointers[process.new_idx()];

                    let dst_ptr = dst_col
                        .data
                        .as_ptr()
                        .add(dst_col.component_layout.size() * dst_arch.entity_ids.len());

                    ptr::copy_nonoverlapping(
                        new_component,
                        dst_ptr,
                        dst_col.component_layout.size(),
                    );

                    process.added_component();
                }
                Done => break,
            }
        }

        // Transfer the entity id.
        let entity_id = src_arch.entity_ids.swap_remove(src.row.0 as usize);
        dst_arch.entity_ids.push(entity_id);

        // Update the location of the moved entity.
        *unsafe { entities.get_mut(entity_id).unwrap_unchecked() } = EntityLocation {
            archetype: dst,
            row: dst_row,
        };

        // Update the location of the entity that was moved by the swap remove
        // operation in the source archetype.
        if let Some(&swapped_entity_id) = src_arch.entity_ids.get(src.row.0 as usize) {
            unsafe { entities.get_mut(swapped_entity_id).unwrap_unchecked() }.row = src.row;
        }

        // If the source archetype no longer has any entities, notify all
        // handlers listening for updates affecting the archetype of this
        // change.
        if src_arch.entity_ids.is_empty() {
            for mut ptr in src_arch.refresh_listeners.iter().copied() {
                unsafe { ptr.as_info_mut().handler_mut().remove_archetype(src_arch) };
            }
        }

        // If the destination archetype was empty before or has been
        // reallocated, notify the listening handlers of this change.
        if dst_arch_reallocated || dst_arch.entity_count() == 1 {
            for mut ptr in dst_arch.refresh_listeners.iter().copied() {
                unsafe { ptr.as_info_mut().handler_mut().refresh_archetype(dst_arch) };
            }
        }

        dst_row
    }

    /// Remove an entity from its archetype.
    ///
    /// # Safety
    ///
    /// Entity location must be valid.
    pub(crate) unsafe fn remove_entity(&mut self, loc: EntityLocation, entities: &mut Entities) {
        // Get a mutable reference to the entity's archetype.
        let arch = unsafe {
            self.archetypes
                .get_mut(loc.archetype.0 as usize)
                .unwrap_unchecked()
        };

        let initial_len = arch.entity_ids.len();

        // Remove the entity's components.
        for col in arch.columns_mut() {
            unsafe { col.swap_remove(initial_len, loc.row.0 as usize) };
        }

        // Eliminate the bounds check in `swap_remove`.
        // SAFETY: Caller guaranteed that the location is valid.
        unsafe {
            assume_unchecked((loc.row.0 as usize) < arch.entity_ids.len());
        };

        // Remove the entity id from the archetype.
        let id = arch.entity_ids.swap_remove(loc.row.0 as usize);

        // Remove the entity location.
        let removed_loc = unsafe { entities.remove(id).unwrap_unchecked() };

        debug_assert_eq!(loc, removed_loc);

        // Update the location of the previously last entity that was moved into
        // the place of the removed entity by the swap-remove operation, unless
        // the last entity in the archetype was removed, in which case this is
        // not necessary.
        if (loc.row.0 as usize) < arch.entity_ids.len() {
            let displaced = *unsafe { arch.entity_ids.get_unchecked(loc.row.0 as usize) };
            unsafe { entities.get_mut(displaced).unwrap_unchecked() }.row = loc.row;
        }

        // If the removed entity's archetype no longer has any entities, notify
        // all handlers listening for updates affecting the archetype of this
        // change.
        if arch.entity_count() == 0 {
            for mut ptr in arch.refresh_listeners.iter().copied() {
                unsafe { ptr.as_info_mut().handler_mut().remove_archetype(arch) };
            }
        }
    }
}

impl Index<ArchetypeIdx> for Archetypes {
    type Output = Archetype;

    /// Panics if the index is invalid.
    fn index(&self, index: ArchetypeIdx) -> &Self::Output {
        if let Some(arch) = self.get(index) {
            arch
        } else {
            panic!("no such archetype with index of {index:?} exists")
        }
    }
}

unsafe impl HandlerParam for &'_ Archetypes {
    type State = ();

    type This<'a> = &'a Archetypes;

    fn init(_world: &mut World, _config: &mut HandlerConfig) -> Result<Self::State, InitError> {
        Ok(())
    }

    unsafe fn get<'a>(
        _state: &'a mut Self::State,
        _info: &'a HandlerInfo,
        _event_ptr: EventPtr<'a>,
        _target_location: EntityLocation,
        world: UnsafeWorldCell<'a>,
    ) -> Self::This<'a> {
        world.archetypes()
    }

    fn refresh_archetype(_state: &mut Self::State, _arch: &Archetype) {}

    fn remove_archetype(_state: &mut Self::State, _arch: &Archetype) {}
}

/// Unique identifier for an archetype.
///
/// Old archetype indices may be reused by new archetypes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ArchetypeIdx(pub u32);

impl ArchetypeIdx {
    /// Index of the archetype with no components.
    pub const EMPTY: Self = Self(0);
    /// The archetype index that is always invalid.
    pub const NULL: Self = Self(u32::MAX);
}

unsafe impl SparseIndex for ArchetypeIdx {
    const MAX: Self = ArchetypeIdx::NULL;

    fn index(self) -> usize {
        self.0.index()
    }

    fn from_index(idx: usize) -> Self {
        Self(u32::from_index(idx))
    }
}

/// Offset from the beginning of a component column. Combined with an
/// [`ArchetypeIdx`], this can identify the location of an entity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ArchetypeRow(pub u32);

impl ArchetypeRow {
    /// The archetype row that is always invalid.
    pub const NULL: Self = Self(u32::MAX);
}

/// A container for all entities with a particular set of components.
///
/// Each component has a corresponding column of contiguous data in
/// this archetype. Each row in is then a separate entity.
///
/// For instance, an archetype with component set `{A, B, C}` might look like:
///
/// | Entity ID | Column A | Column B | Column C |
/// |-----------|----------|----------|----------|
/// | Entity 0  | "foo"    | 123      | true     |
/// | Entity 1  | "bar"    | 456      | false    |
/// | Entity 2  | "baz"    | 789      | true     |
pub struct Archetype {
    /// The index of this archetype. Provided here for convenience.
    index: ArchetypeIdx,
    /// Component indices of this archetype, one per column in sorted order.
    component_indices: ComponentIndices,
    /// Columns of component data in this archetype. Sorted by component index.
    ///
    /// This is a `Box<[Column]>` with the length stripped out. The length field
    /// would be redundant since it's always the same as `component_indices`.
    columns: NonNull<Column>,
    /// A special column containing the [`EntityId`] for all entities in the
    /// archetype.
    entity_ids: Vec<EntityId>,
    /// Handlers that need to be notified about column changes.
    refresh_listeners: BTreeSet<HandlerInfoPtr>,
    /// Targeted event listeners for this archetype.
    event_listeners: SparseMap<TargetedEventIdx, HandlerList>,
}

impl Archetype {
    /// Constructs an empty archetype.
    fn empty() -> Self {
        Self {
            index: ArchetypeIdx::EMPTY,
            component_indices: ComponentIndices::empty(),
            columns: NonNull::dangling(),
            entity_ids: vec![],
            refresh_listeners: BTreeSet::new(),
            event_listeners: SparseMap::new(),
        }
    }

    /// Constructs a new archetype.
    ///
    /// # Safety
    ///
    /// All component indices must be valid.
    unsafe fn new(
        arch_idx: ArchetypeIdx,
        component_indices: ComponentIndices,
        components: &mut Components,
    ) -> Self {
        // Create a column for each component and register this archetype in the
        // component infos.
        let columns: Box<[Column]> = component_indices
            .iter()
            .map(|&idx| {
                // SAFETY: Caller guaranteed the component indices are valid.
                let info = unsafe { components.get_by_index_mut(idx).unwrap_unchecked() };

                // Register this archetype.
                info.member_of.insert(arch_idx);

                // Construct the column.
                Column {
                    component_layout: info.layout(),
                    data: unsafe {
                        // Create a dangling pointer that is correctly aligned.
                        // Correct alignment is necessary as we create borrows
                        // from this pointer, which have to *always* be aligned.
                        // TODO: Use `Layout::dangling` once it's stabilized.
                        // SAFETY: The alignment is guaranteed to be non-zero,
                        // so the pointer cannot be null.
                        NonNull::new_unchecked(transmute(info.layout().align()))
                    },
                    drop: info.drop(),
                }
            })
            .collect();

        // SAFETY: `Box::into_raw` guarantees non-null.
        let columns_ptr = unsafe { NonNull::new_unchecked(Box::into_raw(columns) as *mut Column) };

        Self {
            index: arch_idx,
            component_indices,
            columns: columns_ptr,
            entity_ids: vec![],
            refresh_listeners: BTreeSet::new(),
            event_listeners: SparseMap::new(),
        }
    }

    /// Registers an event handler for this archetype.
    fn register_handler(&mut self, info: &mut HandlerInfo) {
        // TODO: Self::has_component method

        // Tell the handler about this archetype and future updates to it.
        if info
            .archetype_filter()
            .matches_archetype(|idx| self.column_of(idx).is_some())
        {
            // Don't call `refresh_archetype` if this archetype is empty.
            if self.entity_count() > 0 {
                info.handler_mut().refresh_archetype(self);
            }

            self.refresh_listeners.insert(info.ptr());
        }

        // If the handler is targeted, and its component access matches this
        // archetype, insert it into our handler list for the event index.
        if let (Some(expr), EventId::Targeted(event_id)) = (
            info.targeted_event_component_access(),
            info.received_event(),
        ) {
            if expr.matches_archetype(|idx| self.column_of(idx).is_some()) {
                // Insert the handler into the existing list if present, or
                // make a new list and insert the handler into that.
                if let Some(list) = self.event_listeners.get_mut(event_id.index()) {
                    list.insert(info.ptr(), info.priority());
                } else {
                    let mut list = HandlerList::new();
                    list.insert(info.ptr(), info.priority());

                    self.event_listeners.insert(event_id.index(), list);
                }
            }
        }
    }

    /// Returns the list of targeted event listeners for this archetype and
    /// event index.
    pub(crate) fn handler_list_for(&self, idx: TargetedEventIdx) -> Option<&HandlerList> {
        self.event_listeners.get(idx)
    }

    /// Returns the index of this archetype.
    pub fn index(&self) -> ArchetypeIdx {
        self.index
    }

    /// Returns the total number of entities in this archetype.
    pub fn entity_count(&self) -> u32 {
        debug_assert!(u32::try_from(self.entity_ids.len()).is_ok());
        // This doesn't truncate because entity indices are less than u32::MAX.
        self.entity_ids.len() as u32
    }

    /// Returns a slice of [`EntityId`]s for all the entities in this archetype.
    pub fn entity_ids(&self) -> &[EntityId] {
        &self.entity_ids
    }

    /// Returns a sorted list of component indices corresponding to the
    /// component types of this archetype's columns.
    ///
    /// The returned list has the same length as the slice returned by
    /// [`columns`](Archetype::columns).
    pub fn component_indices(&self) -> &ComponentIndices {
        &self.component_indices
    }

    /// Returns a slice of columns sorted by [`ComponentIdx`].
    pub fn columns(&self) -> &[Column] {
        unsafe { slice::from_raw_parts(self.columns.as_ptr(), self.component_indices.len()) }
    }

    /// Returns a slice of columns sorted by [`ComponentIdx`].
    fn columns_mut(&mut self) -> &mut [Column] {
        unsafe { slice::from_raw_parts_mut(self.columns.as_ptr(), self.component_indices.len()) }
    }

    /// Finds the column with the given component. Returns `None` if it doesn't
    /// exist.
    pub fn column_of(&self, idx: ComponentIdx) -> Option<&Column> {
        let idx = self.component_indices().binary_search(&idx).ok()?;

        // SAFETY: `binary_search` ensures `idx` is in bounds.
        Some(unsafe { &*self.columns.as_ptr().add(idx) })
    }

    fn column_of_mut(&mut self, idx: ComponentIdx) -> Option<&mut Column> {
        let idx = self.component_indices().binary_search(&idx).ok()?;

        // SAFETY: `binary_search` ensures `idx` is in bounds.
        Some(unsafe { &mut *self.columns.as_ptr().add(idx) })
    }

    /// Reserve space for at least one additional entity in this archetype. Has
    /// no effect if there is already sufficient capacity. Returns a boolean
    /// indicating if a reallocation occurred.
    // TODO: Does not actually need to be marked unsafe: does not depend on
    //  guarantees made by caller.
    unsafe fn reserve_one(&mut self) -> bool {
        let old_cap = self.entity_ids.capacity();
        // Piggyback off the entity ID Vec's len and cap.
        self.entity_ids.reserve(1);
        // Non-zero because we just reserved space for one element.
        let new_cap = self.entity_ids.capacity();

        #[cold]
        fn capacity_overflow() -> ! {
            panic!("capacity overflow in archetype column")
        }

        if old_cap == new_cap {
            // No reallocation occurred.
            return false;
        }

        /// Scope guard that aborts the process when a panic is triggered while
        /// it is in scope.
        struct AbortOnPanic;

        impl Drop for AbortOnPanic {
            #[cold]
            fn drop(&mut self) {
                // A panic while another panic is happening will abort.
                panic!("column allocation failure");
            }
        }

        // If column reallocation panics for any reason, the columns will be left in an
        // inconsistent state. We have no choice but to abort here.
        let guard = AbortOnPanic;

        // Reallocate all columns.
        for col in self.columns_mut() {
            if col.component_layout.size() == 0 {
                // Skip zero-sized types.
                continue;
            }

            // Non-zero: checked product of two non-zero values.
            let Some(new_cap_in_bytes) = new_cap.checked_mul(col.component_layout.size()) else {
                capacity_overflow()
            };

            if new_cap_in_bytes > isize::MAX as usize {
                capacity_overflow()
            }

            // SAFETY: Alignment requirements checked when
            // `col.component_layout` was constructed. Size requirements were
            // just checked (under the assumption that the size is a multiple of
            // the alignment).
            let new_cap_layout =
                Layout::from_size_align_unchecked(new_cap_in_bytes, col.component_layout.align());

            let ptr = if old_cap == 0 {
                // SAFETY: We checked size > 0 above.
                alloc(new_cap_layout)
            } else {
                // SAFETY: Previous layout must have been valid.
                let old_cap_layout = Layout::from_size_align_unchecked(
                    old_cap * col.component_layout.size(),
                    col.component_layout.align(),
                );

                // SAFETY: `old_cap_layout` is the layout used for the last
                // allocation, from which `col.data.as_ptr()` was returned.
                // `new_cap_in_bytes` was checked to be in bounds (see above).
                realloc(col.data.as_ptr(), old_cap_layout, new_cap_in_bytes)
            };

            match NonNull::new(ptr) {
                Some(ptr) => col.data = ptr,
                None => handle_alloc_error(new_cap_layout),
            }
        }

        // Forget the scope guard to avoid a spurious panic.
        mem::forget(guard);

        true
    }
}

impl Drop for Archetype {
    fn drop(&mut self) {
        let mut columns = unsafe {
            Box::from_raw(slice::from_raw_parts_mut(
                self.columns.as_ptr(),
                self.component_indices.len(),
            ))
        };

        let len = self.entity_ids.len();
        let cap = self.entity_ids.capacity();

        for col in columns.iter_mut() {
            let cap_layout = unsafe {
                Layout::from_size_align_unchecked(
                    cap * col.component_layout.size(),
                    col.component_layout.align(),
                )
            };
            if cap_layout.size() > 0 {
                // Drop components.
                if let Some(drop) = col.drop {
                    for i in 0..len {
                        unsafe {
                            let ptr = col.data.as_ptr().add(i * col.component_layout.size());
                            drop(NonNull::new_unchecked(ptr));
                        };
                    }
                }

                // Free backing buffer.
                unsafe { dealloc(col.data.as_ptr(), cap_layout) };
            }
        }
    }
}

// SAFETY: The safe API of `Archetype` is thread safe, since `unsafe` is
// required to actually read or write the column data.
unsafe impl Send for Archetype {}
unsafe impl Sync for Archetype {}

// Similar logic as above follows for these impls.
impl UnwindSafe for Archetype {}
impl RefUnwindSafe for Archetype {}

impl fmt::Debug for Archetype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Archetype")
            .field("index", &self.index)
            .field("component_indices", &self.component_indices())
            .field("columns", &self.columns())
            .field("entity_ids", &self.entity_ids)
            .field("refresh_listeners", &self.refresh_listeners)
            .field("event_listeners", &self.event_listeners)
            .finish()
    }
}

/// All of the component data for a single component type in an [`Archetype`].
#[derive(Debug)]
pub struct Column {
    /// Layout of a single element.
    component_layout: Layout,
    /// Pointer to beginning of allocated buffer.
    data: NonNull<u8>,
    /// The component drop function.
    drop: DropFn,
}

impl Column {
    /// Returns a pointer to the beginning of the buffer holding the component
    /// data, or a dangling pointer if the buffer is empty.
    pub fn data(&self) -> NonNull<u8> {
        self.data
    }

    /// Overwrites the element at `idx` with the data at `elem`, which this
    /// method takes ownership of.
    ///
    /// # Safety
    ///
    /// - `idx` must be in bounds
    /// - `elem` must point to a component of the correct type
    /// - `elem` must not alias `&mut self`
    unsafe fn assign(&mut self, idx: usize, elem: *const u8) {
        // Obtain raw pointer to the overwritten element using pointer
        // arithmetic. SAFETY: Caller guaranteed `idx` is in bounds.
        let ptr = self.data.as_ptr().add(idx * self.component_layout.size());

        // Call drop function on the overwritten element, if it exists.
        if let Some(drop) = self.drop {
            drop(NonNull::new_unchecked(ptr));
        }

        // Copy the passed element into the buffer to overwrite the old element.
        ptr::copy_nonoverlapping(elem, ptr, self.component_layout.size());
    }

    /// Removes the element at `idx` and immediately moves the last element (at
    /// index `len - 1`) into the empty space.
    ///
    /// # Safety
    ///
    /// - `idx` must be in bounds
    /// - `len` must be correct
    unsafe fn swap_remove(&mut self, len: usize, idx: usize) {
        debug_assert!(idx < len, "index out of bounds");

        // Obtain raw pointer to the last element using pointer arithmetic.
        // SAFETY: Caller guaranteed `len` is correct, meaning `len - 1` is in
        // bounds.
        let src = self
            .data
            .as_ptr()
            .add(self.component_layout.size() * (len - 1));

        // Obtain raw pointer to the removed element using pointer arithmetic.
        // SAFETY: Caller guaranteed `idx` is in bounds.
        let dst = self.data.as_ptr().add(self.component_layout.size() * idx);

        // Call drop function on the overwritten element, if it exists.
        if let Some(drop) = self.drop {
            drop(NonNull::new_unchecked(dst));
        }

        // Copy the last element into the buffer to overwrite the removed
        // element, unless the two elements are equal. This means that the last
        // element of the column was removed, and nothing more needs to be done.
        if src != dst {
            ptr::copy_nonoverlapping(src, dst, self.component_layout.size());
        }
    }

    /// Moves the last element (at index `len - 1`) to `idx`, without dropping
    /// the element at `idx`.
    ///
    /// # Safety
    ///
    /// - `idx` must be in bounds
    /// - `len` must be correct
    unsafe fn swap_remove_no_drop(&mut self, len: usize, idx: usize) {
        debug_assert!(idx < len, "index out of bounds");

        // Obtain raw pointer to the last element using pointer arithmetic.
        // SAFETY: Caller guaranteed `len` is correct, meaning `len - 1` is in
        // bounds.
        let src = self
            .data
            .as_ptr()
            .add(self.component_layout.size() * (len - 1));

        // Obtain raw pointer to the removed element using pointer arithmetic.
        // SAFETY: Caller guaranteed `idx` is in bounds.
        let dst = self.data.as_ptr().add(self.component_layout.size() * idx);

        // Copy the last element into the buffer to overwrite the removed
        // element, unless the two elements are equal. This means that the last
        // element of the column was removed, and nothing more needs to be done.
        if src != dst {
            ptr::copy_nonoverlapping(src, dst, self.component_layout.size());
        }
    }

    /// Moves the element at `src_idx` from `self` to `other`.
    ///
    /// # Safety
    ///
    /// - `self` and `other` must store components of the same type
    /// - `self_len` and `other_len` must be correct
    /// - `src_idx` must be in bounds of `self`
    /// - `other` must have space allocated for the element to be moved to index
    ///   `other_len`
    unsafe fn transfer_elem(
        &mut self,
        self_len: usize,
        other: &mut Self,
        other_len: usize,
        src_idx: usize,
    ) {
        debug_assert_eq!(
            self.component_layout, other.component_layout,
            "component layouts must be the same"
        );
        debug_assert!(src_idx < self_len, "index out of bounds");

        // Obtain raw pointer to the moved element using pointer arithmetic.
        // SAFETY: Caller guaranteed `src_idx` is in bounds.
        let src = self
            .data
            .as_ptr()
            .add(src_idx * self.component_layout.size());

        // Obtain raw pointer to the move destination using pointer arithmetic.
        // SAFETY: Caller guaranteed `other_len` is part of `other`'s
        // allocation.
        let dst = other
            .data
            .as_ptr()
            .add(other_len * other.component_layout.size());

        // Copy the element.
        ptr::copy_nonoverlapping(src, dst, self.component_layout.size());

        // Remove the moved element from our buffer without dropping it, because
        // it is now owned by `other`.
        self.swap_remove_no_drop(self_len, src_idx);
    }
}

// SAFETY: The safe API of `Column` is thread safe, since `unsafe` is
// required to actually read or write the column data.
unsafe impl Send for Column {}
unsafe impl Sync for Column {}

// Similar logic as above follows for these impls.
impl UnwindSafe for Column {}
impl RefUnwindSafe for Column {}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use crate::prelude::*;

    #[derive(Component)]
    struct C(String);

    #[derive(GlobalEvent)]
    struct E1;

    #[derive(GlobalEvent)]
    struct E2;

    #[test]
    fn insert_overwrites() {
        let mut world = World::new();

        let e = world.spawn();

        world.insert(e, C("hello".into()));

        assert_eq!(world.get::<C>(e).unwrap().0, "hello");

        world.insert(e, C("goodbye".into()));

        assert_eq!(world.get::<C>(e).unwrap().0, "goodbye");
    }

    #[test]
    fn zst_components() {
        #[derive(Component)]
        struct Zst;

        let mut world = World::new();

        let e1 = world.spawn();
        world.insert(e1, Zst);

        let e2 = world.spawn();
        world.insert(e2, Zst);
    }

    #[test]
    fn move_entity() {
        #[derive(Component)]
        struct A;

        #[derive(Component)]
        struct B;

        #[derive(Component)]
        struct C;

        let mut world = World::new();

        let a = world.spawn();
        world.insert(a, A);

        let b = world.spawn();
        world.insert(b, B);

        let ab = world.spawn();
        world.insert(ab, A);
        world.insert(ab, B);
        world.insert(ab, C);

        world.remove::<B>(ab);
    }

    #[test]
    fn by_components() {
        #[derive(Component)]
        struct A;

        let mut world = World::new();
        let a_id = world.add_component::<A>();
        let e = world.spawn();
        world.insert(e, A);

        assert!(world
            .archetypes()
            .get_by_components(&[a_id.index()])
            .is_some());
    }

    #[test]
    fn test() {
        #[derive(Component)]
        struct A;

        #[derive(Component)]
        struct B;

        #[derive(Component)]
        #[repr(align(32))]
        struct C;

        println!("{:?}", Layout::new::<C>());

        let mut world = World::new();
        let e = world.spawn();

        world.insert(e, (A, B, C));
        world.get::<A>(e).unwrap();
        world.get::<B>(e).unwrap();
        world.get::<C>(e).unwrap();
    }
}
