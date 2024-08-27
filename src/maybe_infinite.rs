//! Wrapper type for possibly infinite values.

use std::cmp::Ordering;

use crate::maybe_infinite::MaybeInfinite::{Finite, Infinite};

// TODO: Remove this old code after a commit.
/// A value that is either [`Finite`] or [`Infinite`]. `Infinite` refers to
/// positive infinity, such that an `Infinite` value is greater than all
/// `Finite` values. Two `Infinite` values are unequal and incomparable.
pub(crate) enum MaybeInfinite<T> {
    Finite(T),
    Infinite,
}

impl<T: PartialEq> PartialEq<Self> for MaybeInfinite<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Finite(a), Finite(b)) => a == b,
            (Finite(_), Infinite) | (Infinite, Finite(_)) | (Infinite, Infinite) => false,
        }
    }
}

impl<T: PartialOrd> PartialOrd for MaybeInfinite<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Finite(a), Finite(b)) => a.partial_cmp(b),
            (Finite(_), Infinite) => Some(Ordering::Less),
            (Infinite, Finite(_)) => Some(Ordering::Greater),
            (Infinite, Infinite) => None,
        }
    }
}

pub(crate) fn infinite_if_none<T: PartialOrd>(option: Option<T>) -> MaybeInfinite<T> {
    match option {
        Some(v) => Finite(v),
        None => Infinite,
    }
}
