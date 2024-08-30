use alloc::alloc::{alloc, handle_alloc_error};
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec};
use core::alloc::Layout;
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
