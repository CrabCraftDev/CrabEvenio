use alloc::alloc::{dealloc, handle_alloc_error, realloc, Layout};
use core::ptr::{copy_nonoverlapping, NonNull};

use crate::data_type::DataType;
#[derive(Debug)]
pub(crate) struct DataStore {
    /// The type of this data store's elements.
    element_type: DataType,
    /// Pointer to beginning of allocated buffer. Parts of this buffer may be
    /// uninitialized.
    data: NonNull<u8>,
}

impl DataStore {
    /// Constructs an empty data store. This method does not allocate.
    ///
    /// # Safety
    ///
    /// The element type's layout must have a size that is a multiple of its
    /// alignment. This is true for all Rust types, according to the
    /// [Reference]: "The size of a value is always a multiple of its
    /// alignment." However, it is possible to construct layouts that violate
    /// this property:
    ///
    /// ```
    /// # use std::alloc::Layout;
    /// #
    /// // Size (11) is not a multiple of alignment (4) here.
    /// let _layout = Layout::from_size_align(11, 4).unwrap();
    /// ```
    ///
    /// [Reference]: https://doc.rust-lang.org/reference/type-layout.html#size-and-alignment
    pub(crate) unsafe fn new(element_type: DataType) -> Self {
        // TODO: Use usize::is_multiple_of once it's stabilized.
        debug_assert!(element_type.layout.size() % element_type.layout.align() == 0);

        Self {
            element_type,
            data: element_type.dangling(),
        }
    }

    /// Reallocates this data store. The passed old and new capacities are
    /// expected in units of the data type's size.
    ///
    /// # Safety
    ///
    /// - `old_capacity` must be the actual current size of this data store's
    ///   allocation (in units of the data type's size).
    /// - `new_capacity` must be non-zero.
    pub(crate) unsafe fn reallocate(&mut self, old_capacity: usize, new_capacity: usize) {
        if self.element_type.layout.size() == 0 {
            // The element type is zero-sized, so we don't need an allocation.
            return;
        }

        #[cold]
        fn capacity_overflow() -> ! {
            panic!("capacity overflow in archetype column")
        }

        // `new_capacity_in_bytes` is non-zero, because it is the checked
        // product of two non-zero values (ruling out overflow):
        // - We checked that `self.element_type.layout.size()` is non-zero above.
        // - The caller guarantees that `new_capacity` is non-zero.
        let Some(new_capacity_in_bytes) = new_capacity.checked_mul(self.element_type.layout.size())
        else {
            capacity_overflow()
        };

        if new_capacity_in_bytes > isize::MAX as usize {
            capacity_overflow()
        }

        // SAFETY:
        // - Alignment requirements checked when `self.element_type.layout` was
        //   constructed.
        // - The only requirement for the size is that it does not overflow `isize::MAX`
        //   when rounded up to the next multiple of the alignment. When constructing
        //   the data store in `new`, it is already checked that the data type's size is
        //   a multiple of its alignment, and this invariant is nowhere violated. Hence,
        //   rounding up the size to the next multiple of the alignment does not yield a
        //   greater size than `new_capacity_in_bytes`, which we just checked does not
        //   overflow `isize::MAX`.
        let new_capacity_layout = Layout::from_size_align_unchecked(
            new_capacity_in_bytes,
            self.element_type.layout.align(),
        );

        let ptr = if old_capacity == 0 {
            // SAFETY: We checked size > 0 above.
            std::alloc::alloc(new_capacity_layout)
        } else {
            // SAFETY: Previous layout must have been valid; caller guarantees
            // that `old_capacity` is the actual current size of the allocation.
            let old_capacity_layout = Layout::from_size_align_unchecked(
                old_capacity * self.element_type.layout.size(),
                self.element_type.layout.align(),
            );

            // SAFETY:
            // - `old_capacity_layout` is the layout used for the last allocation, from
            //   which the pointer was returned that is now stored in `self.data`.
            // - `new_capacity_in_bytes` was checked to be in bounds (see above).
            realloc(
                self.data.as_ptr(),
                old_capacity_layout,
                new_capacity_in_bytes,
            )
        };

        match NonNull::new(ptr) {
            Some(ptr) => self.data = ptr,
            None => handle_alloc_error(new_capacity_layout),
        }
    }

    /// Deallocates this data store. The passed capacity is expected in units of
    /// the data type's size. The data store must not be used after calling this
    /// function. This function does not drop the store's elements.
    ///
    /// If the capacity is zero or the element type is zero-sized, this function
    /// does nothing.
    ///
    /// # Safety
    ///
    /// `capacity` must be the actual current size of this data store's
    /// allocation (in units of the data type's size).
    pub(crate) unsafe fn deallocate(&self, capacity: usize) {
        // If the capacity is zero or the element type is zero-sized, there is
        // no allocation to deallocate.
        if capacity == 0 || self.element_type.layout.size() > 0 {
            // SAFETY: Caller guarantees that `capacity` is the actual current size
            // of the allocation, so this layout must be valid (it has already been
            // used to allocate the store).
            let layout = Layout::from_size_align_unchecked(
                capacity * self.element_type.layout.size(),
                self.element_type.layout.align(),
            );
            dealloc(self.data.as_ptr(), layout);
        }
    }

    /// Returns a pointer to the underlying allocation.
    pub(crate) fn data(&self) -> NonNull<u8> {
        self.data
    }

    /// Returns a pointer to a stored element or vacant slot.
    ///
    /// # Safety
    ///
    /// The index must be in bounds of the store's current allocated capacity.
    pub(crate) unsafe fn get_unchecked(&self, index: usize) -> *const u8 {
        // SAFETY: The caller guarantees that the index is in bounds, and
        // the allocation cannot be larger than `isize::MAX` (this is a safety
        // invariant of `Layout::from_size_align_unchecked` and `realloc` that
        // we ensure is fulfilled in the `reallocate` method). Transitively,
        // `index * self.element_type.layout.size()` cannot overflow `isize::MAX`
        // either.
        self.data
            .as_ptr()
            .add(index * self.element_type.layout.size())
    }

    /// Returns a mutable pointer to a stored element or vacant slot.
    ///
    /// # Safety
    ///
    /// The index must be in bounds of the store's current allocated capacity.
    pub(crate) unsafe fn get_unchecked_mut(&self, index: usize) -> *mut u8 {
        // SAFETY: See safety comment in `get_unchecked`.
        self.data
            .as_ptr()
            .add(index * self.element_type.layout.size())
    }

    /// Drops a stored element. Its slot is considered vacant afterward.
    ///
    /// # Safety
    ///
    /// The index must be in bounds of the store's current allocated capacity,
    /// and the slot at this index must be occupied. In particular, you must not
    /// call this function with an index of an element that has already been
    /// dropped.
    pub(crate) unsafe fn drop(&self, index: usize) {
        if let Some(drop) = self.element_type.drop_fn {
            drop(NonNull::new_unchecked(self.get_unchecked_mut(index)));
        }
    }

    /// Inserts an element into the vacant slot at `index`, taking ownership of
    /// it. The slot at this index is considered occupied afterward.
    ///
    /// Although it is not recommended, it is safe to call this function with an
    /// index of an occupied slot. In that case, the old element will be
    /// overwritten without being dropped.
    ///
    /// # Safety
    ///
    /// - `index` must be in bounds of the store's current allocated capacity.
    /// - `element` must be a well-aligned pointer to a properly initialized
    ///   value of the store's element type that may be read.
    /// - `element` must not alias the slot at `index`.
    pub(crate) unsafe fn insert(&self, index: usize, element: *const u8) {
        copy_nonoverlapping(
            element,
            self.get_unchecked_mut(index),
            self.element_type.layout.size(),
        );
    }

    /// Moves the element at `from_index` into the vacant slot at `to_index`.
    /// The slot at `from_index` is considered vacant afterward, and the slot at
    /// `to_index` occupied.
    ///
    /// Although it is recommended to call this function with an occupied slot
    /// at `from_index` and a vacant slot at `to_index`, it is safe to call with
    /// occupied or vacant slots at either index. If the slot at `from_index`
    /// was vacant, so is the slot at `to_index` after this call.
    ///
    /// If this function is called with two equal indices, it does nothing.
    ///
    /// # Safety
    ///
    /// Both indices must be in bounds of the store's current allocated
    /// capacity.
    pub(crate) unsafe fn move_within(&self, from_index: usize, to_index: usize) {
        if from_index != to_index {
            copy_nonoverlapping(
                self.get_unchecked(from_index),
                self.get_unchecked_mut(to_index),
                self.element_type.layout.size(),
            );
        }
    }

    /// Transfers an element from one data store to another. Has the same
    /// semantics regarding vacant/occupied slots as [`move_within`].
    ///
    /// # Safety
    ///
    /// - The `from` and `to` stores must have the same element type.
    /// - Each index must be in bounds of its respective store's current
    ///   allocated capacity.
    pub(crate) unsafe fn transfer(from: &Self, to: &Self, from_index: usize, to_index: usize) {
        debug_assert!(DataType::layout_and_drop_fn_equal(
            from.element_type,
            to.element_type,
        ));
        
        copy_nonoverlapping(
            from.get_unchecked(from_index),
            to.get_unchecked_mut(to_index),
            from.element_type.layout.size(),
        )
    }
}
