use alloc::borrow::Cow;
use core::alloc::Layout;
use core::any::TypeId;
use core::ops::Index;

pub use evenio_macros::TargetedEvent;

use super::global::GlobalEvent;
use super::{Event, EventDescriptor, EventKind, EventPtr, Mutability};
use crate::archetype::Archetype;
use crate::drop::DropFn;
use crate::entity::EntityLocation;
use crate::handler::{HandlerConfig, HandlerInfo, HandlerParam};
use crate::map::{Entry, TypeIdMap};
use crate::slot_map::{Key, SlotMap};
use crate::sparse::SparseIndex;
use crate::world::{UnsafeWorldCell, World};

/// An event which is directed at a particular entity.
///
/// This trait is automatically implemented for all types which implement
/// `Event<EventIdx = TargetedEventIdx>`. Use the derive macro to create the
/// appropriate implementation of [`Event`].
///
/// This trait is intended to be mutually exclusive with [`GlobalEvent`].
///
/// # Deriving
///
/// ```
/// use evenio::prelude::*;
///
/// #[derive(TargetedEvent)]
/// struct MyEvent {
///     example_data: i32,
/// }
/// ```
///
/// Due to language limitations, types with generic type params will
/// have a `T: 'static` bound in the generated impl.
///
/// ```
/// use evenio::prelude::*;
///
/// // `T: 'static` is required for this to impl `Event`.
/// #[derive(TargetedEvent)]
/// struct TypeWithGeneric<T>(T);
/// ```
pub trait TargetedEvent: Event<EventIdx = TargetedEventIdx> {}
impl<E: Event<EventIdx = TargetedEventIdx>> TargetedEvent for E {}

/// Stores metadata for all [`Event`]s in the world.
///
/// This can be obtained in a handler by using the `&Events` handler
/// parameter.
///
/// ```
/// # use evenio::prelude::*;
/// # use evenio::event::TargetedEvents;
/// #
/// # #[derive(GlobalEvent)] struct E;
/// #
/// # let mut world = World::new();
/// world.add_handler(|_: Receiver<E>, events: &TargetedEvents| {});
#[derive(Debug)]
pub struct TargetedEvents {
    infos: SlotMap<TargetedEventInfo>,
    by_type_id: TypeIdMap<TargetedEventId>,
}

impl TargetedEvents {
    /// Constructs an empty `TargetedEvents` instance.
    pub(crate) fn new() -> Self {
        Self {
            infos: SlotMap::new(),
            by_type_id: TypeIdMap::default(),
        }
    }

    /// Tries to add an event with the given descriptor. If the descriptor has a
    /// type id and an event with that type id already exists, returns its id
    /// and `false`. Otherwise, add an event with the given descriptor and
    /// returns its id and `true`.
    // TODO: Should this be marked unsafe and have the same safety requirements
    //  as its caller, `World::add_targeted_event_with_descriptor`?
    pub(crate) fn add(&mut self, desc: EventDescriptor) -> (TargetedEventId, bool) {
        // Construct a `TargetedEventInfo` for this event. The id field will be
        // filled in when the info is inserted into our `infos` map.
        let mut info = TargetedEventInfo {
            id: TargetedEventId::NULL,
            name: desc.name,
            kind: desc.kind,
            type_id: desc.type_id,
            layout: desc.layout,
            drop: desc.drop,
            mutability: desc.mutability,
        };

        let insert = || {
            TargetedEventId(
                self.infos
                    .insert_with(|id| {
                        // Fill in the id field with the info's key in the map.
                        info.id = TargetedEventId(id);
                        info
                    })
                    .expect("too many targeted events"),
            )
        };

        if let Some(type_id) = desc.type_id {
            match self.by_type_id.entry(type_id) {
                Entry::Vacant(v) => {
                    // No event with this type id already exists. Call `insert`
                    // to insert the event info constructed above into our
                    // `infos` map and insert the resulting event id into the
                    // vacant `by_type_id` map entry. Finally, return the id.
                    (*v.insert(insert()), true)
                }
                Entry::Occupied(entry) => {
                    // An event with this type id already exists, return its id.
                    (*entry.get(), false)
                }
            }
        } else {
            // The descriptor has no type id to look up. Call `insert` to insert
            // the event info constructed above into our `infos` map and return
            // the resulting event id.
            (insert(), true)
        }
    }

    /// Gets the [`TargetedEventInfo`] of the given event. Returns `None` if the
    /// ID is invalid.
    pub fn get(&self, id: TargetedEventId) -> Option<&TargetedEventInfo> {
        self.infos.get(id.0)
    }

    /// Gets the [`TargetedEventInfo`] for an event using its
    /// [`TargetedEventIdx`]. Returns `None` if the index is invalid.
    #[inline]
    pub fn get_by_index(&self, idx: TargetedEventIdx) -> Option<&TargetedEventInfo> {
        Some(self.infos.get_by_index(idx.0)?.1)
    }

    /// Gets the [`TargetedEventInfo`] for an event using its [`TypeId`].
    /// Returns `None` if the `TypeId` does not map to an event.
    pub fn get_by_type_id(&self, type_id: TypeId) -> Option<&TargetedEventInfo> {
        let idx = *self.by_type_id.get(&type_id)?;
        Some(unsafe { self.get(idx).unwrap_unchecked() })
    }

    /// Does the given event exist in the world?
    pub fn contains(&self, id: TargetedEventId) -> bool {
        self.get(id).is_some()
    }

    /// Tries to remove an event by its id. Returns the event info of the
    /// removed event, or `None` if the id was invalid and no event was
    /// removed.
    pub(crate) fn remove(&mut self, id: TargetedEventId) -> Option<TargetedEventInfo> {
        let info = self.infos.remove(id.0)?;

        if let Some(type_id) = info.type_id {
            self.by_type_id.remove(&type_id);
        }

        Some(info)
    }

    /// Returns an iterator over all event infos.
    pub fn iter(&self) -> impl Iterator<Item = &TargetedEventInfo> {
        self.infos.iter().map(|(_, v)| v)
    }
}

impl Index<TargetedEventId> for TargetedEvents {
    type Output = TargetedEventInfo;

    fn index(&self, index: TargetedEventId) -> &Self::Output {
        if let Some(info) = self.get(index) {
            info
        } else {
            panic!("no such targeted event with ID of {index:?} exists")
        }
    }
}

impl Index<TargetedEventIdx> for TargetedEvents {
    type Output = TargetedEventInfo;

    fn index(&self, index: TargetedEventIdx) -> &Self::Output {
        if let Some(info) = self.get_by_index(index) {
            info
        } else {
            panic!("no such targeted event with index of {index:?} exists")
        }
    }
}

impl Index<TypeId> for TargetedEvents {
    type Output = TargetedEventInfo;

    fn index(&self, index: TypeId) -> &Self::Output {
        if let Some(info) = self.get_by_type_id(index) {
            info
        } else {
            panic!("no such targeted event with type ID of {index:?} exists")
        }
    }
}

unsafe impl HandlerParam for &'_ TargetedEvents {
    type State = ();

    type This<'a> = &'a TargetedEvents;

    fn init(_world: &mut World, _config: &mut HandlerConfig) -> Self::State {}

    unsafe fn get<'a>(
        _state: &'a mut Self::State,
        _info: &'a HandlerInfo,
        _event_ptr: EventPtr<'a>,
        _target_location: EntityLocation,
        world: UnsafeWorldCell<'a>,
    ) -> Self::This<'a> {
        world.targeted_events()
    }

    fn refresh_archetype(_state: &mut Self::State, _arch: &Archetype) {}

    fn remove_archetype(_state: &mut Self::State, _arch: &Archetype) {}
}

/// Contains all the metadata for an added [`TargetedEvent`].
#[derive(Debug)]
pub struct TargetedEventInfo {
    name: Cow<'static, str>,
    id: TargetedEventId,
    kind: EventKind,
    type_id: Option<TypeId>,
    layout: Layout,
    drop: DropFn,
    mutability: Mutability,
}

impl TargetedEventInfo {
    /// Gets the name of the event.
    ///
    /// This name is intended for debugging purposes and should not be relied
    /// upon for correctness.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the ID of the event.
    pub fn id(&self) -> TargetedEventId {
        self.id
    }

    /// Gets the [`EventKind`] of the event.
    pub fn kind(&self) -> &EventKind {
        &self.kind
    }

    /// Gets the [`TypeId`] of the event, if any.
    pub fn type_id(&self) -> Option<TypeId> {
        self.type_id
    }

    /// Gets the [`Layout`] of the event.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Gets the [`DropFn`] of the event.
    pub fn drop(&self) -> DropFn {
        self.drop
    }

    /// Gets the [mutability] of the event
    ///
    /// [mutability]: Event::Mutability
    pub fn mutability(&self) -> Mutability {
        self.mutability
    }
}

/// Lightweight identifier for a targeted event type.
///
/// Event identifiers are implemented using an [index] and a generation count.
/// The generation count ensures that IDs from despawned events are not reused
/// by new events.
///
/// An event identifier is only meaningful in the [`World`] it was created
/// from. Attempting to use an event ID in a different world will have
/// unexpected results.
///
/// [index]: TargetedEventIdx
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash, Debug)]
pub struct TargetedEventId(Key);

impl TargetedEventId {
    /// The global event ID which never identifies a live entity. This is the
    /// default value for `EntityId`.
    pub const NULL: Self = Self(Key::NULL);

    /// Creates a new entity ID from an index and generation count. Returns
    /// `None` if a valid ID is not formed.
    pub const fn new(index: u32, generation: u32) -> Option<Self> {
        match Key::new(index, generation) {
            Some(k) => Some(Self(k)),
            None => None,
        }
    }

    /// Returns the index of this ID.
    pub const fn index(self) -> TargetedEventIdx {
        TargetedEventIdx(self.0.index())
    }

    /// Returns the generation count of this ID.
    pub const fn generation(self) -> u32 {
        self.0.generation().get()
    }
}

/// A [`TargetedEventId`] with the generation count stripped out.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TargetedEventIdx(pub u32);

unsafe impl SparseIndex for TargetedEventIdx {
    const MAX: Self = Self(u32::MAX);

    fn index(self) -> usize {
        self.0.index()
    }

    fn from_index(idx: usize) -> Self {
        Self(u32::from_index(idx))
    }
}

/// An [`Event`] sent immediately after a new targeted event is added to the
/// world.
///
/// Contains the [`TargetedEventId`] of the added event.
#[derive(GlobalEvent, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AddTargetedEvent(pub TargetedEventId);

/// An [`Event`] sent immediately before a targeted event is removed from the
/// world.
///
/// Contains the [`TargetedEventId`] of the event to be removed.
#[derive(GlobalEvent, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RemoveTargetedEvent(pub TargetedEventId);
