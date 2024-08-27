//! Implementation details of the [`ComponentSet`] trait.

use core::iter::once;
use core::mem::MaybeUninit;

use evenio_macros::all_tuples;

use crate::boxed_slice;
use crate::component::{Component, ComponentIdx, ComponentSet};
use crate::permutation::Permutation;
use crate::world::World;

/// Collects type-erased component pointers in sequence. Reorders the added
/// pointers during insertion according to a [`Permutation`]. This struct is
/// marked public, but is only actually accessible from within the crate, as it
/// is placed in a `pub(crate)` module.
#[derive(Debug)]
pub struct ComponentPointerConsumer<'p> {
    permutation: &'p Permutation,
    pointers: Box<[MaybeUninit<*const u8>]>,
    position: usize,
}

impl<'p> ComponentPointerConsumer<'p> {
    /// Constructs a new component pointer consumer with the given permutation.
    pub(crate) fn new(permutation: &'p Permutation) -> Self {
        let pointers = boxed_slice::uninit(permutation.len());
        Self {
            permutation,
            pointers,
            position: 0,
        }
    }

    /// Adds a pointer and advances the consumer's internal position. The
    /// pointer is inserted at the index that the consumer's permutation maps
    /// the current internal position to.
    pub(crate) fn add_pointer(&mut self, pointer: *const u8) {
        let insertion_index = self.permutation.new_index_of(self.position);
        self.position += 1;
        self.pointers[insertion_index] = MaybeUninit::new(pointer);
    }

    /// Unpacks the consumer and returns the collected pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `add_pointer` has been called a number of
    /// times equal to the length of the permutation borrowed by the consumer,
    /// as the contents of the returned boxed slice will not have been fully
    /// initialized otherwise.
    pub(crate) unsafe fn into_pointers_unchecked(self) -> Box<[*const u8]> {
        debug_assert_eq!(self.position, self.permutation.len());
        boxed_slice::assume_init(self.pointers)
    }
}

/// This trait houses the inner workings of [`ComponentSet`]s. It is marked
/// public, but only actually accessible from within the crate, as it is placed
/// in a `pub(crate)` module.
// NOTE: This trait was separated from `ComponentSet` in order to hide
// implementation details such as `ComponentPointerConsumer` or the
// `add_components` function from consumers of this library.
pub unsafe trait ComponentSetInternal {
    /// The number of components in the set.
    const LEN: usize;

    /// Adds the components of this set to the world and extends the passed
    /// collection by their component indices. May extend the collection with
    /// the same index multiple times.
    ///
    /// Implementations must ensure that (1) the collection is extended by
    /// exactly [`Self::LEN`] elements and (2) the order in which the collection
    /// is extended exactly matches the order in which [`get_components`]
    /// adds component pointers to the [`ComponentPointerConsumer`] it is
    /// passed.
    // NOTE: Returning an iterator would be more idiomatic, but causes the
    // borrow checker to complain in the tuple implementations because of
    // multiple mutable borrows of the world existing at the same time. I
    // (AsterixxxGallier) could not get it to work.
    // TODO use ComponentId's instead?
    fn add_components(world: &mut World, component_indices: &mut impl Extend<ComponentIdx>);

    /// Adds type-erased pointers to the components in this set to the given
    /// [`ComponentPointerConsumer`].
    ///
    /// Implementations must ensure that (1) exactly [`Self::LEN`] pointers are
    /// added to the consumer, (2) the order in which pointers are added exactly
    /// matches the order in which [`add_components`] adds component indices to
    /// the collection it is passed and (3) each pointer added to the consumer
    /// must point to a component that is valid for the corresponding component
    /// index produced by `add_components`.
    fn get_components(&self, out: &mut ComponentPointerConsumer);
}

unsafe impl<C: Component> ComponentSetInternal for C {
    const LEN: usize = 1;

    fn add_components(world: &mut World, component_indices: &mut impl Extend<ComponentIdx>) {
        // TODO: Use `Extend::extend_one` once it's stabilized.
        component_indices.extend(once(world.add_component::<C>().index()))
    }

    fn get_components(&self, out: &mut ComponentPointerConsumer) {
        out.add_pointer(self as *const C as *const u8)
    }
}

macro_rules! impl_component_set_tuple {
    ($(($C:ident, $c:ident)),*) => {
        #[allow(unused_variables, unused_mut, clippy::unused_unit)]
        unsafe impl<$($C: ComponentSet),*> ComponentSetInternal for ($($C,)*) {
            const LEN: usize = 0 $(+ $C::LEN)*;

            fn add_components(world: &mut World, component_indices: &mut impl Extend<ComponentIdx>) {
                $(
                    $C::add_components(world, component_indices);
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
