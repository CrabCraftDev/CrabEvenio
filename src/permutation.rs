//! [`Permutation`] struct for storing reorderings.

use core::mem::MaybeUninit;
use crate::boxed_slice;

/// A permutation used for reordering the elements of a slice.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Permutation {
    inner: Box<[usize]>,
}

impl Permutation {
    /// Constructs a permutation from a boxed slice of indices without checking
    /// its validity.
    ///
    /// # Safety
    ///
    /// The slice must contain each valid index into the slice exactly once. In
    /// other words, it must be a reordering of`(0..slice.len()).collect()`.
    pub(crate) unsafe fn new_unchecked(slice: Box<[usize]>) -> Self {
        Self { inner: slice }
    }

    /// Returns the identity permutation for a given length. When used to
    /// reorder a slice, the slice's order is unchanged.
    pub(crate) fn identity(len: usize) -> Self {
        unsafe { Self::new_unchecked((0..len).collect()) }
    }

    /// Returns a permutation that can be used to sort the given slice. When
    /// applied to reorder this slice, the resulting slice will be in sorted
    /// order.
    pub(crate) fn sorting<T: Ord>(to_sort: &[T]) -> Self {
        let mut permutation = Self::identity(to_sort.len());
        // SAFETY: Sorting a slice does not change which elements it contains,
        // only their order. Hence, all our invariants are maintained.
        permutation
            .inner
            .sort_unstable_by_key(|&index| &to_sort[index]);
        permutation
    }

    /// Returns the length of the permutation. The permutation can only be used
    /// to reorder slices of the returned length.
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the index that the given index is mapped to by this permutation.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    pub(crate) fn new_index_of(&self, index: usize) -> usize {
        self.inner[index]
    }

    /// Applies this permutation to the given values and returns the result as a
    /// boxed slice.
    pub(crate) fn apply_collect<T>(&self, values: impl IntoIterator<Item = T>) -> Box<[T]> {
        let mut values = values.into_iter();
        let mut result = boxed_slice::uninit(self.len());
        for index in 0..self.len() {
            let insertion_index = self.new_index_of(index);
            let element = values
                .next()
                .expect("iterator passed to apply_collect terminated early");
            result[insertion_index] = MaybeUninit::new(element);
        }
        // SAFETY: We iterated over all indices in this permutation and inserted
        // an initialized value at each of them. Since the permutation stores
        // each index into itself (and thus, into `result`, which has the same
        // length) exactly once, all elements in `result` are initialized now.
        unsafe { boxed_slice::assume_init(result) }
    }
}