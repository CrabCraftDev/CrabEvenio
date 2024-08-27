//! Lists of component indices.

use std::ops::Deref;
use std::cmp::Ordering;
use core::borrow::Borrow;
use std::hash::{Hash, Hasher};
use core::slice;
use crate::aliased_box::AliasedBox;
use crate::boxed_slice;
use crate::component::ComponentIdx;

/// A sorted, deduplicated list of component indices.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ComponentIndices {
    slice: AliasedBox<[ComponentIdx]>,
}

impl ComponentIndices {
    /// Constructs a list of component indices from the given slice without
    /// checking that it is sorted and deduplicated.
    ///
    /// # Safety
    ///
    /// The slice must be sorted in ascending order and deduplicated.
    pub(crate) unsafe fn new_unchecked(slice: Box<[ComponentIdx]>) -> Self {
        Self { slice: slice.into() }
    }

    /// Returns an empty list of component indices.
    pub fn empty() -> Self {
        // SAFETY: An empty slice is always sorted and deduplicated.
        unsafe { Self::new_unchecked(Box::from([].as_slice())) }
    }

    /// Returns a list of component indices that contains a single component
    /// index.
    pub fn single(component_index: ComponentIdx) -> Self {
        // SAFETY: A slice which contains only one element is always sorted and
        // deduplicated.
        unsafe { Self::new_unchecked(Box::from([component_index].as_slice())) }
    }

    /// Returns a list of component indices which contains all component
    /// indices of `self`, as well as `component_index`. If `self` already
    /// contained the given component index, a clone of `self` is returned.
    pub fn with(&self, component_index: ComponentIdx) -> Self {
        // Binary search for the inserted component index in our slice. If the
        // search succeeds, we already have the component index, so we return
        // a clone of `self`. If it fails, the `binary_search` function tells us
        // where to insert the new component index such that the slice remains
        // sorted, so we insert the component index at that position.
        match self.binary_search(&component_index) {
            Ok(_) => self.clone(),
            Err(position) => unsafe {
                // SAFETY: Inserting the component index at this position is
                // guaranteed to maintain sorted order. If the component index
                // were already contained in this list, we would have returned
                // `self.clone()` in the match arm above, so this insertion
                // cannot introduce duplicates.
                Self::new_unchecked(boxed_slice::insert(&self, position, component_index))
            },
        }
    }

    /// Returns a list of component indices which contains all component
    /// indices of `self`, but not `component_index`. If `self` already did not
    /// contain the given component index, a clone of `self` is returned.
    pub fn without(&self, component_index: ComponentIdx) -> Self {
        // Binary search for the removed component index in our slice. If the
        // search fails, we already don't have the component index, so we return
        // a clone of `self`. If it succeeds, the `binary_search` function tells
        // us where it found the component index, so we remove the component
        // index at that position.
        match self.binary_search(&component_index) {
            Err(_) => self.clone(),
            Ok(position) => unsafe {
                // SAFETY: Removing an element from a slice always maintains
                // sorted order and cannot introduce duplicates.
                Self::new_unchecked(boxed_slice::remove(&self, position))
            },
        }
    }

    /// Returns a list of component indices which contains all component
    /// indices of `self`, as well as those of `other`.
    pub fn with_all(&self, other: &Self) -> Self {
        // TODO: Explain and test this algorithm (and the algorithm used in
        //  `without_all`).
        
        match self[..] {
            [] => return other.clone(),
            [single] => return other.with(single),
            _ => (),
        }

        match other[..] {
            [] => return self.clone(),
            [single] => return self.with(single),
            _ => (),
        }

        let mut vec = Vec::with_capacity(self.len() + other.len());

        let mut self_pos = 0;
        let mut other_pos = 0;

        loop {
            if self_pos == self.len() {
                vec.extend(&other[self_pos..]);
                break;
            }
            if other_pos == other.len() {
                vec.extend(&self[other_pos..]);
                break;
            }

            let self_elem = self[self_pos];
            let other_elem = other[other_pos];

            match self_elem.cmp(&other_elem) {
                Ordering::Less => {
                    vec.push(self_elem);
                    self_pos += 1;
                }
                Ordering::Equal => {
                    vec.push(self_elem);
                    self_pos += 1;
                    other_pos += 1;
                }
                Ordering::Greater => {
                    vec.push(other_elem);
                    other_pos += 1;
                }
            }
        }

        unsafe { Self::new_unchecked(vec.into_boxed_slice()) }
    }

    /// Returns a list of component indices which contains all component
    /// indices of `self`, but not those of `other`.
    pub fn without_all(&self, other: &Self) -> Self {
        match self[..] {
            [] => return self.clone(),
            _ => (),
        }

        match other[..] {
            [] => return self.clone(),
            [single] => return self.without(single),
            _ => (),
        }

        let mut vec = Vec::with_capacity(self.len());

        let mut self_pos = 0;
        let mut other_pos = 0;

        loop {
            if self_pos == self.len() {
                break;
            }
            if other_pos == other.len() {
                vec.extend(&self[other_pos..]);
                break;
            }

            let self_elem = self[self_pos];
            let other_elem = other[other_pos];

            match self_elem.cmp(&other_elem) {
                Ordering::Less => {
                    vec.push(self_elem);
                    self_pos += 1;
                }
                Ordering::Equal => {
                    self_pos += 1;
                    other_pos += 1;
                }
                Ordering::Greater => {
                    other_pos += 1;
                }
            }
        }

        unsafe { Self::new_unchecked(vec.into_boxed_slice()) }
    }

    /// Returns a pointer to the contained slice.
    ///
    /// # Safety
    ///
    /// The returned pointer may **not** outlive `self`. Any usage of the
    /// pointer, even through safe APIs, can result in undefined behaviour after
    /// `self` is dropped.
    pub(crate) unsafe fn as_ptr(&self) -> ComponentIndicesPtr {
        ComponentIndicesPtr {
            ptr: &*self.slice as *const _,
        }
    }
}

// No BorrowMut implementation: that would allow aliased mutability via
// `ComponentIndicesPtr`.
impl Borrow<[ComponentIdx]> for ComponentIndices {
    fn borrow(&self) -> &[ComponentIdx] {
        &self.slice
    }
}

// No AsMut implementation: that would allow aliased mutability via
// `ComponentIndicesPtr`.
impl AsRef<[ComponentIdx]> for ComponentIndices {
    fn as_ref(&self) -> &[ComponentIdx] {
        &self.slice
    }
}

// No DerefMut implementation: that would allow aliased mutability via
// `ComponentIndicesPtr`.
impl Deref for ComponentIndices {
    type Target = [ComponentIdx];

    fn deref(&self) -> &Self::Target {
        &self.slice
    }
}

impl<'a> IntoIterator for &'a ComponentIndices {
    type Item = &'a ComponentIdx;
    type IntoIter = slice::Iter<'a, ComponentIdx>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A pointer to a sorted and deduplicated slice of component indices.
#[derive(Debug, Copy, Clone)]
pub(crate) struct ComponentIndicesPtr {
    ptr: *const [ComponentIdx],
}

impl PartialEq for ComponentIndicesPtr {
    fn eq(&self, other: &Self) -> bool {
        &**self == &**other
    }
}

impl Eq for ComponentIndicesPtr {}

impl Hash for ComponentIndicesPtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&**self).hash(state);
    }
}

// No BorrowMut implementation: that would allow aliased mutability.
impl Borrow<[ComponentIdx]> for ComponentIndicesPtr {
    fn borrow(&self) -> &[ComponentIdx] {
        // SAFETY: The pointer is only ever constructed in
        // `ComponentIndices::as_ptr`, where the caller guarantees that this
        // pointer does not outlive the `ComponentIndices` it references. Hence,
        // the slice must still be initialized at this point. We don't give out
        // mutable references to this slice anywhere, so aliasing rules are
        // trivially followed. All other requirements (alignment, non-null,
        // dereferenceability) for the dereference are guaranteed by `as_ptr`'s
        // implementation.
        unsafe { &*self.ptr }
    }
}

// No AsMut implementation: that would allow aliased mutability.
impl AsRef<[ComponentIdx]> for ComponentIndicesPtr {
    fn as_ref(&self) -> &[ComponentIdx] {
        self.borrow()
    }
}

// No DerefMut implementation: that would allow aliased mutability.
impl Deref for ComponentIndicesPtr {
    type Target = [ComponentIdx];

    fn deref(&self) -> &Self::Target {
        self.borrow()
    }
}