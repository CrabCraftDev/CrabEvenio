use alloc::alloc::{alloc, handle_alloc_error, Layout};
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec};
use core::mem::{transmute, MaybeUninit};
use core::ptr::slice_from_raw_parts_mut;

/// Allocates an uninitialized slice using the global allocator.
/// 
/// # Safety
/// 
/// - `T` must not be zero-sized.
/// - `len` must not be zero.
// TODO: Use Box::new_uninit_slice instead once it's stable.
pub(crate) unsafe fn uninit<T>(len: usize) -> Box<[MaybeUninit<T>]> {
    // Allocate the slice.
    let target_layout = Layout::array::<T>(len).unwrap();
    // SAFETY: The size of `target_layout` cannot be zero, as the caller
    // guaranteed that `len` is non-zero and `T` is not zero-sized.
    let pointer = unsafe { alloc(target_layout) };
    if pointer.is_null() {
        handle_alloc_error(target_layout);
    }

    // Make the allocation a boxed slice.
    let pointer = pointer.cast::<MaybeUninit<T>>();
    let pointer = slice_from_raw_parts_mut(pointer, len);
    // SAFETY: We used the global allocator to allocate the correct layout
    // (which is correct because `MaybeUninit<T>` and `T` have the same
    // layout), so calling `Box::from_raw` here is fine.
    unsafe { Box::from_raw(pointer) }
}

/// Constructs a boxed slice which equals `source` with `element` inserted
/// at `index`. This doesn't actually allocate if `T` is zero-sized.
///
/// # Examples
///
/// Basic usage:
/// ```ignore
/// let source = [1, 2, 4, 5].as_slice();
/// let element = 3;
/// let result: Box<[u32]> = boxed_slice::insert(source, 2, element);
/// assert_eq!(result.len(), source.len() + 1);
/// assert_eq!(&result[0..2], &source[0..2]);
/// assert_eq!(&result[2], &element);
/// assert_eq!(&result[3..5], &source[2..4]);
/// ```
///
/// # Panics
///
/// Panics if the index is out of range for the constructed slice (it may
/// equal `source.len()`) or if `source.len() == usize::MAX`.
pub(crate) fn insert<T: Copy>(source: &[T], index: usize, element: T) -> Box<[T]> {
    let new_len = source.len().checked_add(1).unwrap();
    assert!(index < new_len);

    if size_of::<T>() == 0 {
        // If `T` is zero-sized, we use a shortcut. Values of a zero-sized
        // type are indiscernible, so we just "copy" `element` to the
        // appropriate length and return a boxed slice of that. Since `T`
        // is zero-sized, this doesn't actually allocate.
        return vec![element; new_len].into_boxed_slice();
    }

    // Allocate the boxed slice which we will insert the element into.
    // SAFETY: `new_len` is guaranteed to be positive and we checked above
    // that `T` is not zero-sized.
    let mut boxed = unsafe { uninit(new_len) };

    // Prepare the source slice by transmuting it from &[T] to
    // &[MaybeUninit<T>], so we can copy them into the uninitialized slice.
    // SAFETY: Transmuting from `T` to `MaybeUninit<T>` is safe. We could
    // call `map` and call `MaybeUninit::new` on each element, but this
    // `transmute` saves us the iterator overhead and handles the whole
    // slice at once.
    let source: &[MaybeUninit<T>] = unsafe { transmute(source) };

    // Copy the initial and final parts from `source`, leaving a gap for the
    // element to insert.
    boxed[..index].copy_from_slice(&source[..index]);
    boxed[index + 1..].copy_from_slice(&source[index..]);

    // Insert the element.
    boxed[index] = MaybeUninit::new(element);

    // Transmute the boxed slice from Box<[MaybeUninit<T>]> to Box<[T]>,
    // effectively assuming its elements are initialized. This is fine
    // because we initialized the elements at `..index`, `index` and
    // `index + 1..`, which covers all indices of the boxed slice.
    unsafe { transmute(boxed) }
}

/// Constructs a boxed slice which equals `source` with the element at
/// `index` removed. This doesn't actually allocate if `T` is zero-sized.
///
/// # Examples
///
/// Basic usage:
/// ```ignore
/// let source = [1, 2, 3, 4, 5].as_slice();
/// let result: Box<[u32]> = boxed_slice::remove(source, 2);
/// assert_eq!(result.len(), source.len() - 1);
/// assert_eq!(&result[0..2], &source[0..2]);
/// assert_eq!(&result[2..4], &source[3..5]);
/// ```
///
/// # Panics
///
/// Panics if the index is out of range or if `source.len() == 0`.
pub(crate) fn remove<T: Copy>(source: &[T], index: usize) -> Box<[T]> {
    assert!(index < source.len());
    let new_len = source.len().checked_sub(1).unwrap();

    if size_of::<T>() == 0 {
        // If `T` is zero-sized, we use a shortcut. Values of a zero-sized
        // type are indiscernible, so we just "copy" the first element in
        // `source` to the appropriate length and return a boxed slice of
        // that. Since `T` is zero-sized, this doesn't actually allocate.
        // SAFETY: We checked `source` is not empty using `checked_sub`
        // above.
        let element = unsafe { *source.get_unchecked(0) };
        return vec![element; new_len].into_boxed_slice();
    }
    
    if new_len == 0 {
        // If `new_len` is zero, we simply return an empty boxed slice. This
        // does not actually allocate.
        return Box::from([].as_slice());
    }

    // Allocate the boxed slice which we will insert the element into.
    // SAFETY: We checked above that `new_len` is non-zero and that `T` is
    // not zero-sized.
    let mut boxed = unsafe { uninit(new_len) };

    // Prepare the source slice by transmuting it from &[T] to
    // &[MaybeUninit<T>], so we can copy them into the uninitialized slice.
    // SAFETY: Transmuting from `T` to `MaybeUninit<T>` is safe. We could
    // call `map` and call `MaybeUninit::new` on each element, but this
    // `transmute` saves us the iterator overhead and handles the whole
    // slice at once.
    let source: &[MaybeUninit<T>] = unsafe { transmute(source) };

    // Copy the initial and final parts from `source`, skipping the removed
    // element.
    boxed[..index].copy_from_slice(&source[..index]);
    boxed[index..].copy_from_slice(&source[index + 1..]);

    // Transmute the boxed slice from Box<[MaybeUninit<T>]> to Box<[T]>,
    // effectively assuming its elements are initialized. This is fine
    // because we initialized the elements at `..index` and `index..`,
    // which covers all indices of the boxed slice.
    unsafe { transmute(boxed) }
}

#[cfg(test)]
mod tests {
    use crate::boxed_slice::{insert, remove};

    #[test]
    fn insert_at_start() {
        let source = [1, 2, 3, 4].as_slice();
        let result = insert(source, 0, 0);
        assert_eq!(&*result, &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn insert_in_mid() {
        let source = [1, 2, 4, 5].as_slice();
        let result = insert(source, 2, 3);
        assert_eq!(&*result, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn insert_at_end() {
        let source = [1, 2, 3, 4].as_slice();
        let result = insert(source, 4, 5);
        assert_eq!(&*result, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn insert_zero_sized() {
        let source = [(), (), (), ()].as_slice();
        let result = insert(source, 1, ());
        assert_eq!(&*result, &[(), (), (), (), ()])
    }

    #[test]
    #[should_panic]
    fn insert_out_of_bounds() {
        let source = [1, 2, 3, 4].as_slice();
        insert(source, 5, 6);
    }

    #[test]
    fn remove_at_start() {
        let source = [1, 2, 3, 4, 5].as_slice();
        let result = remove(source, 0);
        assert_eq!(&*result, &[2, 3, 4, 5]);
    }

    #[test]
    fn remove_in_mid() {
        let source = [1, 2, 3, 4, 5].as_slice();
        let result = remove(source, 2);
        assert_eq!(&*result, &[1, 2, 4, 5]);
    }

    #[test]
    fn remove_at_end() {
        let source = [1, 2, 3, 4, 5].as_slice();
        let result = remove(source, 4);
        assert_eq!(&*result, &[1, 2, 3, 4]);
    }

    #[test]
    fn remove_zero_sized() {
        let source = [(), (), (), (), ()].as_slice();
        let result = remove(source, 1);
        assert_eq!(&*result, &[(), (), (), ()])
    }

    #[test]
    #[should_panic]
    fn remove_out_of_bounds() {
        let source = [1, 2, 3, 4, 5].as_slice();
        remove(source, 5);
    }
}
