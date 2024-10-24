//! Types for working with [`Component`]s.

use alloc::borrow::Cow;
use alloc::collections::BTreeSet;
use core::alloc::Layout;
use core::any::TypeId;
use core::mem::MaybeUninit;
use core::ops::Index;

use ahash::RandomState;
use evenio_macros::all_tuples;
pub use evenio_macros::{Component, ComponentSet};

use crate::archetype::{Archetype, ArchetypeIdx};
use crate::drop::DropFn;
use crate::entity::EntityLocation;
use crate::event::{EventPtr, GlobalEvent, TargetedEventId};
use crate::handler::{HandlerConfig, HandlerInfo, HandlerParam};
use crate::map::{Entry, IndexSet, TypeIdMap};
use crate::mutability::{Mutability, MutabilityMarker};
use crate::permutation::Permutation;
use crate::prelude::World;
use crate::slot_map::{Key, SlotMap};
use crate::sparse::SparseIndex;
use crate::world::UnsafeWorldCell;

/// Contains metadata for all the components in a world.
///
/// This can be obtained in a handler by using the `&Components` handler
/// parameter.
///
/// ```
/// # use evenio::prelude::*;
/// # use evenio::component::Components;
/// #
/// # #[derive(GlobalEvent)] struct E;
/// #
/// # let mut world = World::new();
/// world.add_handler(|_: Receiver<E>, components: &Components| {});
/// ```
#[derive(Debug)]
pub struct Components {
    infos: SlotMap<ComponentInfo>,
    by_type_id: TypeIdMap<ComponentId>,
}

impl Components {
    /// Constructs an empty `Components` instance.
    pub(crate) fn new() -> Self {
        Self {
            infos: SlotMap::new(),
            by_type_id: TypeIdMap::default(),
        }
    }

    /// Tries to add a component with the given descriptor. If the descriptor
    /// has a type id and a component with that type id already exists, returns
    /// its id and `false`. Otherwise, add a component with the given descriptor
    /// and returns its id and `true`.
    // TODO: Should this be marked unsafe and have the same safety requirements
    //  as its caller, `World::add_component_with_descriptor`?
    pub(crate) fn add(&mut self, desc: ComponentDescriptor) -> (ComponentId, bool) {
        // If the descriptor has a type id, look it up in our `by_type_id` map.
        if let Some(type_id) = desc.type_id {
            return match self.by_type_id.entry(type_id) {
                Entry::Vacant(v) => {
                    // No component with this type id already exists. Create a
                    // `ComponentInfo` for the new component, insert it into the
                    // `by_type_id` and `infos` maps and return the component's
                    // id.

                    let Some(k) = self.infos.insert_with(|k| ComponentInfo {
                        name: desc.name,
                        id: ComponentId(k),
                        type_id: desc.type_id,
                        layout: desc.layout,
                        drop: desc.drop,
                        mutability: desc.mutability,
                        spawn_events: BTreeSet::new(),
                        insert_events: BTreeSet::new(),
                        remove_events: BTreeSet::new(),
                        member_of: IndexSet::with_hasher(RandomState::new()),
                    }) else {
                        panic!("too many components")
                    };

                    (*v.insert(ComponentId(k)), true)
                }
                Entry::Occupied(entry) => {
                    // A component with this type id already exists, return its
                    // id.
                    (*entry.get(), false)
                }
            };
        }

        // The descriptor has no type id to look up. Create a `ComponentInfo`
        // for the new component, insert it into the `infos` map and return the
        // new component's id.
        let Some(k) = self.infos.insert_with(|k| ComponentInfo {
            name: desc.name,
            id: ComponentId(k),
            type_id: desc.type_id,
            layout: desc.layout,
            drop: desc.drop,
            mutability: desc.mutability,
            spawn_events: BTreeSet::new(),
            insert_events: BTreeSet::new(),
            remove_events: BTreeSet::new(),
            member_of: IndexSet::with_hasher(RandomState::new()),
        }) else {
            panic!("too many components")
        };

        (ComponentId(k), true)
    }

    /// Tries to remove a component by its id. Returns the component info of the
    /// removed component, or `None` if the id was invalid and no component was
    /// removed.
    pub(crate) fn remove(&mut self, component_id: ComponentId) -> Option<ComponentInfo> {
        let info = self.infos.remove(component_id.0)?;

        if let Some(type_id) = info.type_id {
            self.by_type_id.remove(&type_id);
        }

        Some(info)
    }

    /// Gets the [`ComponentInfo`] of the given component. Returns `None` if the
    /// ID is invalid.
    pub fn get(&self, id: ComponentId) -> Option<&ComponentInfo> {
        self.infos.get(id.0)
    }

    /// Gets the [`ComponentInfo`] for a component using its [`ComponentIdx`].
    /// Returns `None` if the index is invalid.
    pub fn get_by_index(&self, idx: ComponentIdx) -> Option<&ComponentInfo> {
        self.infos.get_by_index(idx.0).map(|(_, v)| v)
    }

    /// Returns a mutable reference to the [`ComponentInfo`] for a component
    /// by its [`ComponentIdx`]. Returns `None` if the index is invalid.
    pub(crate) fn get_by_index_mut(&mut self, idx: ComponentIdx) -> Option<&mut ComponentInfo> {
        self.infos.get_by_index_mut(idx.0).map(|(_, v)| v)
    }

    /// Gets the [`ComponentInfo`] for a component using its [`TypeId`]. Returns
    /// `None` if the `TypeId` does not map to a component.
    pub fn get_by_type_id(&self, type_id: TypeId) -> Option<&ComponentInfo> {
        let id = *self.by_type_id.get(&type_id)?;
        Some(unsafe { self.get(id).unwrap_unchecked() })
    }

    /// Returns `true` if the given component exists in the world.
    pub fn contains(&self, id: ComponentId) -> bool {
        self.get(id).is_some()
    }

    /// Returns an iterator over all component infos.
    pub fn iter(&self) -> impl Iterator<Item = &ComponentInfo> {
        self.infos.iter().map(|(_, v)| v)
    }
}

impl Index<ComponentId> for Components {
    type Output = ComponentInfo;

    fn index(&self, index: ComponentId) -> &Self::Output {
        if let Some(info) = self.get(index) {
            info
        } else {
            panic!("no such component with ID of {index:?} exists")
        }
    }
}

impl Index<ComponentIdx> for Components {
    type Output = ComponentInfo;

    fn index(&self, index: ComponentIdx) -> &Self::Output {
        if let Some(info) = self.get_by_index(index) {
            info
        } else {
            panic!("no such component with index of {index:?} exists")
        }
    }
}

impl Index<TypeId> for Components {
    type Output = ComponentInfo;

    fn index(&self, index: TypeId) -> &Self::Output {
        if let Some(info) = self.get_by_type_id(index) {
            info
        } else {
            panic!("no such component with type ID of {index:?} exists")
        }
    }
}

unsafe impl HandlerParam for &'_ Components {
    type State = ();

    type This<'a> = &'a Components;

    fn init(_world: &mut World, _config: &mut HandlerConfig) -> Self::State {}

    unsafe fn get<'a>(
        _state: &'a mut Self::State,
        _info: &'a HandlerInfo,
        _event_ptr: EventPtr<'a>,
        _target_location: EntityLocation,
        world: UnsafeWorldCell<'a>,
    ) -> Self::This<'a> {
        world.components()
    }

    fn refresh_archetype(_state: &mut Self::State, _arch: &Archetype) {}

    fn remove_archetype(_state: &mut Self::State, _arch: &Archetype) {}
}

/// Metadata for a component.
#[derive(Debug)]
pub struct ComponentInfo {
    name: Cow<'static, str>,
    id: ComponentId,
    type_id: Option<TypeId>,
    layout: Layout,
    drop: DropFn,
    mutability: Mutability,
    pub(crate) spawn_events: BTreeSet<TargetedEventId>,
    pub(crate) insert_events: BTreeSet<TargetedEventId>,
    pub(crate) remove_events: BTreeSet<TargetedEventId>,
    /// The set of archetypes that have this component as one of its columns.
    pub(crate) member_of: IndexSet<ArchetypeIdx>,
}

impl ComponentInfo {
    /// Gets the name of the component.
    ///
    /// This name is intended for debugging purposes and should not be relied
    /// upon for correctness.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the ID of the component.
    pub fn id(&self) -> ComponentId {
        self.id
    }

    /// Gets the [`TypeId`] of the component, or `None` if it was not assigned a
    /// type ID.
    pub fn type_id(&self) -> Option<TypeId> {
        self.type_id
    }

    /// Gets the [`Layout`] of the component.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Gets the [`DropFn`] of the component.
    pub fn drop(&self) -> DropFn {
        self.drop
    }

    /// Gets the [`Mutability`] of the component.
    pub fn mutability(&self) -> Mutability {
        self.mutability
    }

    /// Gets the set of [`Spawn`] events for this component.
    ///
    /// [`Spawn`]: crate::event::Spawn
    pub fn spawn_events(&self) -> &BTreeSet<TargetedEventId> {
        &self.spawn_events
    }

    /// Gets the set of [`Insert`] events for this component.
    ///
    /// [`Insert`]: crate::event::Insert
    pub fn insert_events(&self) -> &BTreeSet<TargetedEventId> {
        &self.insert_events
    }

    /// Gets the set of [`Remove`] components for this component.
    ///
    /// [`Remove`]: crate::event::Remove
    pub fn remove_events(&self) -> &BTreeSet<TargetedEventId> {
        &self.remove_events
    }
}

/// Types which store data on [entities].
///
/// A `Component` is a piece of data which can be attached to an entity. An
/// entity can have any combination of components, but cannot have more than one
/// component of the same type.
///
/// To add a component to an entity, use the [`Insert`] event. To access
/// components from handlers, use the [`Fetcher`] handler parameter.
///
/// [entities]: crate::entity
/// [`Insert`]: crate::event::Insert
/// [`Fetcher`]: crate::fetch::Fetcher
///
/// # Deriving
///
/// The `Component` trait can be implemented automatically by using the
/// associated derive macro. However, the type must still satisfy the `'static`
/// bound to do so.
///
/// ```
/// use evenio::prelude::*;
///
/// // Component with some data.
/// #[derive(Component)]
/// struct Username(String);
///
/// // Component without data, known as a "marker" or "tag" component.
/// #[derive(Component)]
/// struct Invisible;
///
/// // Derive it on structs with named fields.
/// #[derive(Component)]
/// struct Position {
///     x: f32,
///     y: f32,
///     z: f32,
/// }
///
/// // ...and on enums.
/// #[derive(Component)]
/// enum FriendStatus {
///     Friendly,
///     Neutral,
///     Unfriendly,
/// }
///
/// // Components can be immutable, which disallows mutable references
/// // to the component once it's attached to an entity.
/// #[derive(Component)]
/// #[component(immutable)] // Override the default mutability.
/// struct FooCounter(i32);
/// ```
pub trait Component: 'static {
    /// Indicates if this event is [`Mutable`] or [`Immutable`].
    ///
    /// Immutable components disallow mutable references, which can be used to
    /// ensure components are only modified via events.
    ///
    /// [`Mutable`]: crate::mutability::Mutable
    /// [`Immutable`]: crate::mutability::Immutable
    type Mutability: MutabilityMarker;
}

/// Data needed to create a new component.
#[derive(Clone, Debug)]
pub struct ComponentDescriptor {
    /// The name of this component.
    ///
    /// This name is intended for debugging purposes and should not be relied
    /// upon for correctness.
    pub name: Cow<'static, str>,
    /// The [`TypeId`] of this component, if any.
    pub type_id: Option<TypeId>,
    /// The [`Layout`] of the component.
    pub layout: Layout,
    /// The [`DropFn`] of the component. This is passed a pointer to the
    /// component in order to drop it.
    pub drop: DropFn,
    /// The [mutability](Component::Mutability) of this component.
    pub mutability: Mutability,
}

/// Lightweight identifier for a component type.
///
/// component identifiers are implemented using an [index] and a generation
/// count. The generation count ensures that IDs from removed components are
/// not reused by new components.
///
/// A component identifier is only meaningful in the [`World`] it was created
/// from. Attempting to use a component ID in a different world will have
/// unexpected results.
///
/// [index]: ComponentIdx
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct ComponentId(Key);

impl ComponentId {
    /// The component ID which never identifies a live component. This is the
    /// default value for `ComponentId`.
    pub const NULL: Self = Self(Key::NULL);

    /// Creates a new component ID from an index and generation count. Returns
    /// `None` if a valid ID is not formed.
    pub const fn new(index: u32, generation: u32) -> Option<Self> {
        match Key::new(index, generation) {
            Some(k) => Some(Self(k)),
            None => None,
        }
    }

    /// Returns the index of this ID.
    pub const fn index(self) -> ComponentIdx {
        ComponentIdx(self.0.index())
    }

    /// Returns the generation count of this ID.
    pub const fn generation(self) -> u32 {
        self.0.generation().get()
    }
}

/// A [`ComponentId`] with the generation count stripped out.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct ComponentIdx(pub u32);

unsafe impl SparseIndex for ComponentIdx {
    const MAX: Self = Self(u32::MAX);

    fn index(self) -> usize {
        self.0.index()
    }

    fn from_index(idx: usize) -> Self {
        Self(u32::from_index(idx))
    }
}

/// Collects type-erased component pointers in sequence. Reorders the added
/// pointers during insertion according to a [`Permutation`]. This struct is
/// marked public, but is only actually accessible from within the crate, as it
/// is placed in a `pub(crate)` module.
#[derive(Debug)]
pub struct ComponentPointerConsumer<'p> {
    permutation: &'p Permutation,
    pointers: &'p mut [MaybeUninit<*const u8>],
    position: usize,
}

impl<'p> ComponentPointerConsumer<'p> {
    /// Constructs a new component pointer consumer with the given permutation.
    pub(crate) fn new(
        permutation: &'p Permutation,
        pointers: &'p mut [MaybeUninit<*const u8>],
    ) -> Self {
        debug_assert_eq!(permutation.len(), pointers.len());
        Self {
            permutation,
            pointers,
            position: 0,
        }
    }

    /// Adds a pointer and advances the consumer's internal position. The
    /// pointer is inserted at the index that the consumer's permutation maps
    /// the current internal position to.
    #[inline(always)]
    pub(crate) fn add_pointer(&mut self, pointer: *const u8) {
        let insertion_index = self.permutation.new_index_of(self.position);
        self.position += 1;
        self.pointers[insertion_index] = MaybeUninit::new(pointer);
    }

    /// Returns the collected pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `add_pointer` has been called a number of
    /// times equal to the length of the permutation borrowed by the consumer,
    /// as the contents of the returned slice will not have been fully
    /// initialized otherwise.
    pub(crate) unsafe fn get_pointers_unchecked(&self) -> &[*const u8] {
        debug_assert_eq!(self.position, self.permutation.len());
        &*(self.pointers as *const [MaybeUninit<*const u8>] as *const [*const u8])
    }
}

/// A set of components.
///
/// On the type level, this functions as a set of component types, for example
/// for removing all components of certain types from an entity, or for removing
/// or inserting component types altogether.
///
/// On the value level, this functions as a set of component values, for example
/// for inserting components into an entity or spawning an entity with initial
/// components.
///
/// This trait is implemented for all components and tuples of components, so
/// `C1`, `()`, `(C1,)`, `(C1, C2)` etc. are all component sets. Due to language
/// limitations, these implementations only support tuples with up to 16
/// elements. If you need a bigger component set, you can nest component sets:
/// `((C1, C2), (C3, C4, C5), C6)` is also a component set.
///
/// # Safety
///
/// It is not recommended and usually not necessary to implement this trait
/// yourself. However, if you do, you must uphold the safety contract of
/// [`add_components`] and [`get_components`].
///
/// [`add_components`]: Self::add_components
/// [`get_components`]: Self::get_components
// TODO: #[derive(ComponentSet)] akin to bevy's #[derive(Bundle)]? would require
//  exposing at least some of the implementation details
pub unsafe trait ComponentSet: 'static {
    /// The number of components in the set.
    fn len() -> usize;

    /// Adds the components of this set to the world and calls the passed
    /// callback with their IDs. May call the callback with the same ID multiple
    /// times.
    ///
    /// Implementations must ensure that (1) the callback is called exactly
    /// [`Self::len()`] times and (2) the order of the calls exactly matches the
    /// order in which [`get_components`] adds component pointers to the
    /// [`ComponentPointerConsumer`] it is passed.
    ///
    /// [`get_components`]: Self::get_components
    // NOTE: Returning an iterator would be more idiomatic, but causes the
    // borrow checker to complain in the tuple implementations because of
    // multiple mutable borrows of the world existing at the same time. I
    // (AsterixxxGallier) could not get it to work.
    fn add_components(world: &mut World, add_component_id: impl FnMut(ComponentId));

    /// Removes the components of this set from the world and calls the passed
    /// callback with the [`ComponentInfo`] of each component that existed and
    /// was successfully removed.
    fn remove_components(world: &mut World, add_component_info: impl FnMut(ComponentInfo));

    /// Adds type-erased pointers to the components in this set to the given
    /// [`ComponentPointerConsumer`].
    ///
    /// Implementations must ensure that (1) exactly [`Self::len()`] pointers
    /// are added to the consumer, (2) the order in which pointers are added
    /// exactly matches the order in which [`add_components`] adds component
    /// indices to the collection it is passed and (3) each pointer added to
    /// the consumer must point to a component that is valid for the
    /// corresponding component index produced by `add_components`.
    ///
    /// [`add_components`]: Self::add_components
    fn get_components(&self, out: &mut ComponentPointerConsumer);
}

unsafe impl<C: Component> ComponentSet for C {
    #[inline]
    fn len() -> usize {
        1
    }

    fn add_components(world: &mut World, mut add_component_id: impl FnMut(ComponentId)) {
        add_component_id(world.add_component::<C>())
    }

    fn remove_components(world: &mut World, mut add_component_info: impl FnMut(ComponentInfo)) {
        if let Some(info) = world.remove_component::<C>() {
            add_component_info(info);
        }
    }

    fn get_components(&self, out: &mut ComponentPointerConsumer) {
        out.add_pointer(self as *const C as *const u8)
    }
}

macro_rules! impl_component_set_tuple {
    ($(($C:ident, $c:ident)),*) => {
        #[allow(unused_variables, unused_mut, clippy::unused_unit)]
        unsafe impl<$($C: ComponentSet),*> ComponentSet for ($($C,)*) {
            #[inline]
            fn len() -> usize {
                0 $(+ $C::len())*
            }

            fn add_components(world: &mut World, mut add_component_id: impl FnMut(ComponentId)) {
                $(
                    $C::add_components(world, &mut add_component_id);
                )*
            }

            fn remove_components(world: &mut World, mut add_component_info: impl FnMut(ComponentInfo)) {
                $(
                    $C::remove_components(world, &mut add_component_info);
                )*
            }

            fn get_components(&self, out: &mut ComponentPointerConsumer) {
                let ($($c,)*) = self;
                $(
                    $c.get_components(out);
                )*
            }
        }
    };
}

all_tuples!(impl_component_set_tuple, 0, 16, C, c);

/// An event sent immediately after a new component is added to the world.
/// Contains the ID of the added component.
#[derive(GlobalEvent, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AddComponent(pub ComponentId);

/// An event sent immediately before a component is removed from the world.
/// Contains the ID of the component to be removed.
#[derive(GlobalEvent, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RemoveComponent(pub ComponentId);

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[derive(GlobalEvent)]
    struct E;

    #[test]
    fn derive_component_set() {
        #[derive(Component, Clone, Debug, Eq, PartialEq)]
        struct A(String);

        #[derive(Component, Clone, Debug, Eq, PartialEq)]
        struct B(String);

        #[derive(ComponentSet, Clone, Debug, Eq, PartialEq)]
        struct StructSet {
            a: A,
            b: B,
        }

        let mut world = World::new();
        let set = StructSet {
            a: A("foo".to_owned()),
            b: B("bar".to_owned()),
        };
        let e = world.spawn(set.clone());

        // TODO: `world.get::<&StructSet>(e)` should work!
        assert_eq!(world.get::<&A>(e), Some(&set.a));
        assert_eq!(world.get::<&B>(e), Some(&set.b));
    }

    #[test]
    fn remove_component() {
        #[derive(Component)]
        struct A(String);

        #[derive(Component, PartialEq, Debug)]
        struct B(Vec<String>);

        let mut world = World::new();

        world.add_component::<A>();
        let e1 = world.spawn(());
        world.insert(e1, A("hello".into()));
        let s1 = world.add_handler(|_: Receiver<E>, mut a: Single<&mut A>| {
            a.0.push_str("hello");
        });
        world.send(E);

        assert!(world.remove_component::<A>().is_some());
        assert!(!world.handlers().contains(s1));
        assert!(!world.entities().contains(e1));
        assert_eq!(
            world.archetypes().len(),
            1,
            "only the empty archetype should be present"
        );

        world.add_component::<B>();
        let e2 = world.spawn(());
        assert!(world.entities().contains(e2));
        world.insert(e2, B(vec![]));
        let s2 = world.add_handler(|_: Receiver<E>, mut b: Single<&mut B>| {
            b.0.push("hello".into());
        });
        world.send(E);
        assert_eq!(world.get::<&B>(e2), Some(&B(vec!["hello".into()])));

        assert!(world.remove_component::<B>().is_some());
        assert!(!world.handlers().contains(s2));
        assert!(!world.entities().contains(e2));
        assert_eq!(world.archetypes().len(), 1);
    }

    #[test]
    fn component_member_of() {
        let mut world = World::new();

        #[derive(Component)]
        struct A;

        #[derive(Component)]
        struct B;

        #[derive(Component)]
        struct C;

        let c1 = world.add_component::<A>();
        let c2 = world.add_component::<B>();
        let c3 = world.add_component::<C>();

        let e1 = world.spawn(());
        let e2 = world.spawn(());
        let e3 = world.spawn(());

        world.insert(e1, A);

        world.insert(e2, A);
        world.insert(e2, B);

        world.insert(e3, A);
        world.insert(e3, B);
        world.insert(e3, C);

        assert_eq!(world.components()[c1].member_of.len(), 3);
        assert_eq!(world.components()[c2].member_of.len(), 2);
        assert_eq!(world.components()[c3].member_of.len(), 1);

        world.remove_component_by_id(c3);

        assert_eq!(world.components()[c1].member_of.len(), 2);
        assert_eq!(world.components()[c2].member_of.len(), 1);

        world.remove_component_by_id(c2);

        assert_eq!(world.components()[c1].member_of.len(), 1);
    }
}
