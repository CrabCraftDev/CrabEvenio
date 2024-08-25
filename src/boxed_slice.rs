use alloc::alloc::{alloc, handle_alloc_error, Layout};
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec};
use core::mem::{transmute, MaybeUninit};
use core::ptr::{slice_from_raw_parts_mut, NonNull};

/// Allocates an uninitialized slice of the specified length using the global
/// allocator.
// TODO: Use Box::new_uninit_slice instead once it's stable.
pub(crate) fn uninit<T>(len: usize) -> Box<[MaybeUninit<T>]> {
    let layout = Layout::array::<T>(len).unwrap();
    
    // If the layout we would allocate with is zero-sized, we instead use a
    // dangling pointer, since it's invalid to allocate with a zero-sized
    // layout. Otherwise, we allocate using the global allocator.
    let pointer = if layout.size() == 0 {
        // Use a dangling, but correctly aligned pointer.
        NonNull::<T>::dangling().as_ptr().cast()
    } else {
        // Allocate the slice.
        // SAFETY: We just checked that `layout` has a size greater than zero.
        let pointer = unsafe { alloc(layout) };
        if pointer.is_null() {
            handle_alloc_error(layout);
        }
        pointer
    };

    // Make the allocation a boxed slice.
    let pointer = pointer.cast::<MaybeUninit<T>>();
    let pointer = slice_from_raw_parts_mut(pointer, len);
    // SAFETY: We used the global allocator to allocate the correct layout
    // (which is correct because `MaybeUninit<T>` and `T` have the same
    // layout), so calling `Box::from_raw` here is fine.
    unsafe { Box::from_raw(pointer) }
}

/// Assumes that the given boxed slice of `MaybeUninit<T>` values is fully
/// initialized and converts it into `Box<[T]>`.
///
/// # Safety
///
/// The caller must guarantee that the slice is actually fully initialized.
pub(crate) unsafe fn assume_init<T>(slice: Box<[MaybeUninit<T>]>) -> Box<[T]> {
    transmute(slice)
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

    // Allocate the boxed slice which we will insert the element into.
    let mut boxed = uninit(new_len);

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

    // Assume the elements of `boxed` are all initialized now. This is fine
    // because we initialized the elements at `..index`, `index` and
    // `index + 1..`, which covers all indices of the boxed slice.
    unsafe { assume_init(boxed) }
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

    // Allocate the boxed slice which we will insert the element into.
    let mut boxed = uninit(new_len);

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

    // Assume the elements of `boxed` are all initialized now. This is fine
    // because we initialized the elements at `..index` and `index..`,
    // which covers all indices of the boxed slice.
    unsafe { assume_init(boxed) }
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
