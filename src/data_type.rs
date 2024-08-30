#![expect(unused, reason = "will be used in more places down the line")]

use core::alloc::Layout;
use core::any::{type_name, TypeId};
use core::ptr::NonNull;
use crate::drop::{drop_fn_of, DropFn};

/// A type whose properties are stored and processed dynamically at runtime.
#[derive(Debug, Copy, Clone)]
pub(crate) struct DataType {
    /// The size and alignment of values of this type.
    pub(crate) layout: Layout,
    /// A function to call to drop a value of this type, or `None` if values of
    /// this type don't need to be dropped.
    pub(crate) drop_fn: DropFn,
    /// An optional type id for debugging purposes.
    #[cfg(debug_assertions)]
    pub(crate) type_id: Option<TypeId>,
    /// An optional type name for debugging purposes.
    #[cfg(debug_assertions)]
    pub(crate) type_name: Option<&'static str>,
}

impl DataType {
    /// Constructs a [`DataType`] from the given Rust type.
    pub(crate) fn of<T: 'static>() -> Self {
        Self {
            layout: Layout::new::<T>(),
            drop_fn: drop_fn_of::<T>(),
            #[cfg(debug_assertions)]
            type_id: Some(TypeId::of::<T>()),
            #[cfg(debug_assertions)]
            type_name: Some(type_name::<T>()),
        }
    }

    /// Returns a dangling pointer that is correctly aligned for the layout of
    /// this data type.
    pub(crate) fn dangling(&self) -> NonNull<u8> {
        // TODO: Use `Layout::dangling` once it's stabilized.
        // SAFETY: The alignment is guaranteed to be non-zero, so the pointer
        // cannot be null.
        unsafe { NonNull::new_unchecked(self.layout.align() as *mut u8) }
    }
    
    pub(crate) fn layout_and_drop_fn_equal(a: DataType, b: DataType) -> bool {
        a.layout == b.layout && a.drop_fn == b.drop_fn
    }
}