//! Types for sending and receiving [`Event`]s.

mod global;
mod targeted;

use alloc::borrow::Cow;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::alloc::Layout;
use core::any::TypeId;
use core::marker::PhantomData;
use core::mem::{offset_of, transmute};
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};
use core::{any, fmt, slice, str};

use evenio_macros::all_tuples;
pub use global::*;
pub use targeted::*;

use crate::access::Access;
use crate::archetype::Archetype;
use crate::component::{ComponentIdx, ComponentPointerConsumer, ComponentSet};
use crate::drop::{drop_fn_of, DropFn};
use crate::entity::{EntityId, EntityLocation};
use crate::fetch::FetcherState;
use crate::handler::{HandlerConfig, HandlerInfo, HandlerParam};
use crate::mutability::{Immutable, Mutability, MutabilityMarker, Mutable};
use crate::permutation::Permutation;
use crate::prelude::Component;
use crate::query::Query;
use crate::world::{UnsafeWorldCell, World};

/// Messages which event handlers listen for. This is the base trait of
/// [`GlobalEvent`] and [`TargetedEvent`].
///
/// To send and receive events within handlers, see [`Sender`] and
/// [`Receiver`].
///
/// # Safety
///
/// This trait is `unsafe` to implement because unsafe code relies on correct
/// implementations of [`This`] and [`init`] to avoid undefined behavior. Note
/// that implementations produced by the derive macros are always safe.
///
/// [`This`]: Self::This
/// [`init`]: Self::init
pub unsafe trait Event {
    /// The type of `Self`, but with lifetimes modified to outlive `'a`.
    ///
    /// # Safety
    ///
    /// This type _must_ correspond to the type of `Self`. In particular, it
    /// must be safe to transmute between `Self` and `This<'a>` (assuming `'a`
    /// is correct). Additionally, the [`TypeId`] of `Self` must match that
    /// of `This<'static>`.
    type This<'a>: 'a;

    /// Either [`GlobalEventIdx`] or [`TargetedEventIdx`]. This indicates if the
    /// event is global or targeted.
    type EventIdx: EventIdxMarker;

    /// Indicates if this event is [`Mutable`] or [`Immutable`].
    ///
    /// Immutable events disallow mutable references to the event and ownership
    /// transfer via [`EventMut::take`]. This is useful for ensuring events
    /// are not altered during their lifespan.    
    type Mutability: MutabilityMarker;

    /// Gets the [`EventKind`] of this event and performs any necessary
    /// initialization work.
    ///
    /// # Safety
    ///
    /// Although this method is safe to call, it is unsafe to implement
    /// because unsafe code relies on the returned [`EventKind`] being correct
    /// for this type. Additionally, the `world` cannot be used in ways that
    /// would result in dangling indices during handler initialization.
    ///
    /// The exact safety requirements are currently unspecified, but the default
    /// implementation returns [`EventKind::Normal`] and is always safe.
    fn init(world: &mut World) -> EventKind {
        let _ = world;
        EventKind::Normal
    }
}

/// Additional behaviors for an event. This is used to distinguish normal
/// user events from special built-in events.
#[derive(Clone, PartialEq, Eq, Hash, Default, Debug)]
#[non_exhaustive]
pub enum EventKind {
    /// An event not covered by one of the other variants. Events of this kind
    /// have no special effects.
    #[default]
    Normal,
    /// The [`Insert`] event.
    Insert(InsertedComponentsInfo),
    /// The [`Remove`] event.
    Remove(RemovedComponentsInfo),
    /// The [`Spawn`] event.
    Spawn(SpawnInfo),
    /// The [`Despawn`] event.
    Despawn,
}

/// Additional data for [`EventKind::Spawn`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SpawnInfo {
    pub(crate) components_field_offset: usize,
    pub(crate) inserted_components: InsertedComponentsInfo,
}

/// Additional data for [`EventKind::Insert`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct InsertedComponentsInfo {
    /// The indices of the components to insert.
    pub(crate) component_indices: BitSet<ComponentIdx>,
    /// Permutation used to sort the component set.
    pub(crate) permutation: Permutation,
    /// A type-erased function pointer to the [`get_components`] function of
    /// the component set type.
    ///
    /// [`get_components`]: [`ComponentSetInternal::get_components`]
    pub(crate) get_components: GetComponentsFn,
}

/// Additional data for [`EventKind::Remove`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RemovedComponentsInfo {
    /// The indices of the components to remove.
    pub(crate) component_indices: BitSet<ComponentIdx>,
}

/// Data needed to create a new event.
#[derive(Clone, Debug)]
pub struct EventDescriptor {
    /// The name of this event.
    ///
    /// This name is intended for debugging purposes and should not be relied
    /// upon for correctness.
    pub name: Cow<'static, str>,
    /// The [`TypeId`] of this event, if any.
    pub type_id: Option<TypeId>,
    /// The [`EventKind`] of the event.
    pub kind: EventKind,
    /// The [`Layout`] of the event.
    pub layout: Layout,
    /// The [`DropFn`] of the event. This is passed a pointer to the
    /// event in order to drop it.
    pub drop: DropFn,
    /// The [mutability](Event::Mutability) of this event.
    pub mutability: Mutability,
}

impl EventDescriptor {
    /// Constructs and initializes an `EventDescriptor` for the given type in a
    /// world.
    pub(crate) fn new<E: Event>(world: &mut World) -> Self {
        Self {
            name: any::type_name::<E>().into(),
            type_id: Some(TypeId::of::<E::This<'static>>()),
            kind: E::init(world),
            layout: Layout::new::<E>(),
            drop: drop_fn_of::<E>(),
            mutability: Mutability::of::<E::Mutability>(),
        }
    }
}

/// An enum of either [`GlobalEventId`] or [`TargetedEventId`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum EventId {
    /// A global event.
    Global(GlobalEventId),
    /// A targeted event.
    Targeted(TargetedEventId),
}

impl EventId {
    /// Is this a [`EventId::Global`] event?
    pub const fn is_global(self) -> bool {
        matches!(self, Self::Global(_))
    }

    /// Is this a [`EventId::Targeted`] event?
    pub const fn is_targeted(self) -> bool {
        matches!(self, Self::Targeted(_))
    }
}

impl From<GlobalEventId> for EventId {
    fn from(value: GlobalEventId) -> Self {
        Self::Global(value)
    }
}

impl From<TargetedEventId> for EventId {
    fn from(value: TargetedEventId) -> Self {
        Self::Targeted(value)
    }
}

/// Sealed marker trait implemented for [`GlobalEventIdx`] and
/// [`TargetedEventIdx`].
pub trait EventIdxMarker: Send + Sync + 'static + event_idx_marker::Sealed {}

impl EventIdxMarker for GlobalEventIdx {}

impl EventIdxMarker for TargetedEventIdx {}

mod event_idx_marker {
    use super::*;

    pub trait Sealed {}

    impl Sealed for GlobalEventIdx {}
    impl Sealed for TargetedEventIdx {}
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EventQueueItem {
    pub(crate) meta: EventMeta,
    /// Type-erased pointer to this event. When null, ownership of the event
    /// has been transferred and no destructor needs to run.
    pub(crate) event: NonNull<u8>,
}

/// Metadata for an event in the event queue.
#[derive(Clone, Copy, Debug)]
pub(crate) enum EventMeta {
    Global {
        idx: GlobalEventIdx,
    },
    Targeted {
        idx: TargetedEventIdx,
        target: EntityId,
    },
}

/// Type-erased pointer to an event. Passed to handlers in [`Handler::run`].
///
/// [`Handler::run`]: crate::handler::Handler::run
#[derive(Clone, Copy, Debug)]
pub struct EventPtr<'a> {
    event: NonNull<u8>,
    // `false` when borrowed, `true` when taken.
    ownership_flag: NonNull<bool>,
    _marker: PhantomData<&'a mut u8>,
}

impl<'a> EventPtr<'a> {
    /// Constructs a new event pointer.
    pub(crate) fn new(event: NonNull<u8>, ownership_flag: NonNull<bool>) -> Self {
        Self {
            event,
            ownership_flag,
            _marker: PhantomData,
        }
    }

    /// Returns the underlying pointer to the type-erased event.
    #[track_caller]
    pub fn as_ptr(self) -> NonNull<u8> {
        let is_owned = unsafe { *self.ownership_flag.as_ptr() };
        debug_assert!(
            !is_owned,
            "`as_ptr` must not be called after the event has been marked as owned"
        );

        self.event
    }

    /// Marks the event as owned. It is then the handler's responsibility to
    /// drop the event.
    ///
    /// # Safety
    ///
    /// - Must have permission to access the event mutably.
    /// - Once the event is set as owned, [`as_ptr`] must not be called and any
    ///   pointer acquired through `as_ptr` may not be used anymore.
    ///
    /// [`as_ptr`]: Self::as_ptr
    pub unsafe fn set_owned(self) {
        *self.ownership_flag.as_ptr() = true;
    }
}

/// Mutable reference to an instance of event `E`.
///
/// To get at `E`, use the [`Deref`] and [`DerefMut`] implementations or
/// [`take`](Self::take).
pub struct EventMut<'a, E: Event> {
    ptr: EventPtr<'a>,
    _marker: PhantomData<&'a mut E::This<'a>>,
}

impl<'a, E: Event> EventMut<'a, E> {
    /// Constructs a new `EventMut` which wraps the given event pointer.
    fn new(ptr: EventPtr<'a>) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Takes ownership of the event. Any handlers expected to run after the
    /// current handler will not run.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use evenio::prelude::*;
    /// # let mut world = World::new();
    /// # #[derive(GlobalEvent)]
    /// # struct E;
    /// #
    /// # let mut world = World::new();
    /// #
    /// world.add_handler(|r: ReceiverMut<E>| {
    ///     EventMut::take(r.event); // Took ownership of event.
    /// });
    ///
    /// world.add_handler(|_: Receiver<E>| panic!("boom"));
    ///
    /// world.send(E);
    /// // ^ No panic occurs because the first handler took
    /// // ownership of the event before the second could run.
    /// ```
    pub fn take(this: Self) -> E {
        let res = unsafe { this.ptr.as_ptr().as_ptr().cast::<E>().read() };
        unsafe { this.ptr.set_owned() };
        res
    }
}

unsafe impl<'a, E> Send for EventMut<'a, E>
where
    E: Event,
    E::This<'a>: Send,
{
}

unsafe impl<'a, E> Sync for EventMut<'a, E>
where
    E: Event,
    E::This<'a>: Sync,
{
}

impl<'a, E: Event> Deref for EventMut<'a, E> {
    type Target = E::This<'a>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ptr().cast::<E::This<'_>>().as_ref() }
    }
}

impl<E: Event> DerefMut for EventMut<'_, E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_ptr().cast::<E::This<'_>>().as_mut() }
    }
}

impl<'a, E> fmt::Debug for EventMut<'a, E>
where
    E: Event,
    E::This<'a>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EventMut").field(&**self).finish()
    }
}

/// A [`HandlerParam`] which listens for events of type `E`.
///
/// For more information, see the relevant [tutorial
/// chapter](crate::tutorial#handlers-and-events).
///
/// # Examples
///
/// ```
/// use evenio::prelude::*;
///
/// #[derive(GlobalEvent)]
/// struct E;
///
/// let mut world = World::new();
///
/// world.add_handler(|r: Receiver<E>| {
///     println!("got event of type E!");
/// });
/// ```
#[derive(Clone, Copy)]
pub struct Receiver<'a, E: Event, Q: ReceiverQuery + 'static = NullReceiverQuery> {
    /// A reference to the received event.
    pub event: &'a E::This<'a>,
    /// The result of the query. This field is meaningless if `E` is not a
    /// [`TargetedEvent`].
    pub query: Q::Item<'a>,
}

unsafe impl<E: GlobalEvent> HandlerParam for Receiver<'_, E> {
    type State = ();

    type This<'a> = Receiver<'a, E>;

    fn init(world: &mut World, config: &mut HandlerConfig) -> Self::State {
        let event_id = world.add_global_event::<E>();

        config.set_received_event(event_id);
        config.set_received_event_access(Access::Read);
    }

    unsafe fn get<'a>(
        _state: &'a mut Self::State,
        _info: &'a HandlerInfo,
        event_ptr: EventPtr<'a>,
        _target_location: EntityLocation,
        _world: UnsafeWorldCell<'a>,
    ) -> Self::This<'a> {
        Receiver {
            // SAFETY:
            // - We have permission to access the event immutably.
            // - Handler was configured to listen for `E`.
            event: event_ptr.as_ptr().cast().as_ref(),
            query: (),
        }
    }

    fn refresh_archetype(_state: &mut Self::State, _arch: &Archetype) {}

    fn remove_archetype(_state: &mut Self::State, _arch: &Archetype) {}
}

impl<'a, E> fmt::Debug for Receiver<'a, E>
where
    E: GlobalEvent,
    E::This<'a>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Receiver")
            .field("event", &self.event)
            .finish_non_exhaustive()
    }
}

unsafe impl<E: TargetedEvent, Q: Query + 'static> HandlerParam for Receiver<'_, E, Q> {
    type State = FetcherState<Q>;

    type This<'a> = Receiver<'a, E, Q>;

    fn init(world: &mut World, config: &mut HandlerConfig) -> Self::State {
        let event_id = world.add_targeted_event::<E>();

        let state = Q::new_state(world);
        let ca = Q::get_access(&state, |idx| {
            config.referenced_components.insert(idx);
        });

        config.set_received_event(event_id);
        config.set_received_event_access(Access::Read);
        config.set_targeted_event_component_access(ca.clone());
        config.push_component_access(ca);

        FetcherState::new(state)
    }

    unsafe fn get<'a>(
        state: &'a mut Self::State,
        _info: &'a HandlerInfo,
        event_ptr: EventPtr<'a>,
        target_location: EntityLocation,
        _world: UnsafeWorldCell<'a>,
    ) -> Self::This<'a> {
        let event = event_ptr.as_ptr().cast::<E::This<'_>>().as_ref();

        // SAFETY: Caller guarantees the target entity matches the query.
        let query = state.get_by_location_mut(target_location);

        Receiver { event, query }
    }

    fn refresh_archetype(state: &mut Self::State, arch: &Archetype) {
        state.refresh_archetype(arch)
    }

    fn remove_archetype(state: &mut Self::State, arch: &Archetype) {
        state.remove_archetype(arch)
    }
}

impl<'a, E, Q> fmt::Debug for Receiver<'a, E, Q>
where
    E: TargetedEvent,
    E::This<'a>: fmt::Debug,
    Q: Query,
    Q::This<'a>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Receiver")
            .field("event", &self.event)
            .field("query", &self.query)
            .finish()
    }
}

/// Like [`Receiver`], but provides mutable access to the received event. Prefer
/// `Receiver` if mutable access is not needed.
///
/// For more information, see the relevant [tutorial
/// chapter](crate::tutorial#event-mutation).
pub struct ReceiverMut<'a, E: Event, Q: ReceiverQuery + 'static = NullReceiverQuery> {
    /// A mutable reference to the received event.
    pub event: EventMut<'a, E>,
    /// The result of the query. This field is meaningless if `E` is not a
    /// targeted event.
    pub query: Q::Item<'a>,
}

unsafe impl<E> HandlerParam for ReceiverMut<'_, E>
where
    E: GlobalEvent + Event<Mutability = Mutable>,
{
    type State = ();

    type This<'a> = ReceiverMut<'a, E>;

    fn init(world: &mut World, config: &mut HandlerConfig) -> Self::State {
        let event_id = world.add_global_event::<E>();

        config.set_received_event(event_id);
        config.set_received_event_access(Access::ReadWrite);
    }

    unsafe fn get<'a>(
        _state: &'a mut Self::State,
        _info: &'a HandlerInfo,
        event_ptr: EventPtr<'a>,
        _target_location: EntityLocation,
        _world: UnsafeWorldCell<'a>,
    ) -> Self::This<'a> {
        ReceiverMut {
            event: EventMut::new(event_ptr),
            query: (),
        }
    }

    fn refresh_archetype(_state: &mut Self::State, _arch: &Archetype) {}

    fn remove_archetype(_state: &mut Self::State, _arch: &Archetype) {}
}

impl<'a, E> fmt::Debug for ReceiverMut<'a, E>
where
    E: GlobalEvent,
    E::This<'a>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Receiver")
            .field("event", &self.event)
            .finish_non_exhaustive()
    }
}

unsafe impl<E, Q> HandlerParam for ReceiverMut<'_, E, Q>
where
    E: TargetedEvent + Event<Mutability = Mutable>,
    Q: Query + 'static,
{
    type State = FetcherState<Q>;

    type This<'a> = ReceiverMut<'a, E, Q>;

    fn init(world: &mut World, config: &mut HandlerConfig) -> Self::State {
        let event_id = world.add_targeted_event::<E>();

        let state = Q::new_state(world);
        let ca = Q::get_access(&state, |idx| {
            config.referenced_components.insert(idx);
        });

        config.set_received_event(event_id);
        config.set_received_event_access(Access::ReadWrite);
        config.set_targeted_event_component_access(ca.clone());
        config.push_component_access(ca);

        FetcherState::new(state)
    }

    unsafe fn get<'a>(
        state: &'a mut Self::State,
        _info: &'a HandlerInfo,
        event_ptr: EventPtr<'a>,
        target_location: EntityLocation,
        _world: UnsafeWorldCell<'a>,
    ) -> Self::This<'a> {
        let event = EventMut::<E>::new(event_ptr);

        // SAFETY: Caller guarantees the target entity matches the query.
        let query = state.get_by_location_mut(target_location);

        ReceiverMut { event, query }
    }

    fn refresh_archetype(state: &mut Self::State, arch: &Archetype) {
        state.refresh_archetype(arch)
    }

    fn remove_archetype(state: &mut Self::State, arch: &Archetype) {
        state.remove_archetype(arch)
    }
}

impl<'a, E, Q> fmt::Debug for ReceiverMut<'a, E, Q>
where
    E: TargetedEvent,
    E::This<'a>: fmt::Debug,
    Q: Query,
    Q::This<'a>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Receiver")
            .field("event", &self.event)
            .field("query", &self.query)
            .finish()
    }
}

/// Indicates the absence of a [`ReceiverQuery`].
#[derive(Clone, Copy, Debug)]
pub enum NullReceiverQuery {}

/// Targeted event queries used in [`Receiver`] and [`ReceiverMut`]. This trait
/// is implemented for all types which implement [`Query`].
///
/// This trait is sealed and cannot be implemented for types outside this crate.
pub trait ReceiverQuery: null_receiver_query::Sealed {
    /// The item produced by the query.
    type Item<'a>;
}

impl ReceiverQuery for NullReceiverQuery {
    type Item<'a> = ();
}

impl<Q: Query> ReceiverQuery for Q {
    type Item<'a> = Q::This<'a>;
}

mod null_receiver_query {
    use super::*;

    pub trait Sealed {}

    impl Sealed for NullReceiverQuery {}

    impl<Q: Query> Sealed for Q {}
}

/// A [`HandlerParam`] for sending events from the set `T`.
///
/// For more information, see the relevant [tutorial
/// chapter](crate::tutorial#sending-events-from-handlers).
#[derive(Clone, Copy)]
pub struct Sender<'a, T: EventSet> {
    state: &'a T::Indices,
    world: UnsafeWorldCell<'a>,
}

impl<'a, ES: EventSet> Sender<'a, ES> {
    /// Add a [`GlobalEvent`] to the queue of events to send.
    ///
    /// The queue is flushed once all handlers for the current event have run.
    ///
    /// # Panics
    ///
    /// - Panics if `E` is not in the [`EventSet`] of this sender.
    #[track_caller]
    pub fn send<E: GlobalEvent + 'a>(&self, event: E) {
        // The event type and event set are all compile time known, so the compiler
        // should be able to optimize this away.
        let event_idx = ES::find_index::<E>(self.state).unwrap_or_else(|| {
            panic!(
                "global event `{}` is not in the `EventSet` of this `Sender`",
                any::type_name::<E>()
            )
        });

        let ptr = self.alloc_layout(Layout::new::<E>());

        unsafe { ptr::write::<E>(ptr.as_ptr().cast(), event) };

        unsafe { self.world.queue_global(ptr, GlobalEventIdx(event_idx)) };
    }

    /// Add a [`TargetedEvent`] to the queue of events to send.
    ///
    /// The queue is flushed once all handlers for the current event have run.
    #[track_caller]
    pub fn send_to<E: TargetedEvent + 'a>(&self, target: EntityId, event: E) {
        // The event type and event set are all compile time known, so the compiler
        // should be able to optimize this away.
        let event_idx = ES::find_index::<E>(self.state).unwrap_or_else(|| {
            panic!(
                "targeted event `{}` is not in the `EventSet` of this `Sender`",
                any::type_name::<E>()
            )
        });

        let ptr = self.alloc_layout(Layout::new::<E>());

        unsafe { ptr::write::<E>(ptr.as_ptr().cast(), event) };

        unsafe {
            self.world
                .queue_targeted(target, ptr, TargetedEventIdx(event_idx))
        };
    }

    /// Queue the creation of a new entity with an initial set of components.
    /// Note that this set can be empty.
    ///
    /// This returns the [`EntityId`] of the to-be-spawned entity and queues the
    /// [`Spawn`] event. Note that the returned `EntityId` may not be valid
    /// until after the `Spawn` event has finished broadcasting.
    ///
    /// The returned `EntityId` will not have been used by any previous
    /// entities.
    ///
    /// # Panics
    ///
    /// Panics if `Spawn` is not in the [`EventSet`] of this sender.
    #[track_caller]
    pub fn spawn<C: ComponentSet>(&self, components: C) -> EntityId {
        let id = unsafe { self.world.queue_spawn() };
        self.send(Spawn(id, components));
        id
    }

    /// Queue an [`Insert`] event.
    ///
    /// This is equivalent to:
    ///
    /// ```
    /// # use evenio::prelude::*;
    /// # #[derive(Component)] struct C;
    /// # fn _f(sender: &mut Sender<Insert<C>>, target: EntityId, component: C) {
    /// sender.send_to(target, Insert(component));
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `Insert<C>` is not in the [`EventSet`] of this sender.
    #[track_caller]
    pub fn insert<C: ComponentSet>(&self, target: EntityId, components: C) {
        self.send_to(target, Insert(components))
    }

    /// Queue a [`Remove`] event.
    ///
    /// This is equivalent to:
    ///
    /// ```
    /// # use evenio::prelude::*;
    /// # #[derive(Component)] struct C;
    /// # fn _f(sender: &mut Sender<Remove<C>>, target: EntityId) {
    /// sender.send_to(target, Remove::<C>);
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `Remove<C>` is not in the [`EventSet`] of this sender.
    #[track_caller]
    pub fn remove<C: Component>(&self, target: EntityId) {
        self.send_to(target, Remove::<C>)
    }

    /// Queue a [`Despawn`] event.
    ///
    /// This is equivalent to:
    ///
    /// ```
    /// # use evenio::prelude::*;
    /// # fn _f(sender: &mut Sender<Despawn>, target: EntityId) {
    /// sender.send_to(target, Despawn);
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `Despawn` is not in the [`EventSet`] of this sender.
    #[track_caller]
    pub fn despawn(&self, target: EntityId) {
        self.send_to(target, Despawn)
    }

    /// Allocate an object into the bump allocator and return an exclusive
    /// reference to it.
    #[inline]
    pub fn alloc<T>(&self, value: T) -> &'a mut T {
        let ptr = self.alloc_layout(Layout::new::<T>()).cast::<T>().as_ptr();

        unsafe { ptr::write(ptr, value) };

        unsafe { &mut *ptr }
    }

    /// Allocate a slice into the bump allocator and return an exclusive
    /// reference to it.
    ///
    /// The elements of the slice are initialized using the supplied closure.
    /// The closure argument is the position in the slice.
    #[inline]
    pub fn alloc_slice<T, F>(&self, len: usize, mut f: F) -> &'a mut [T]
    where
        F: FnMut(usize) -> T,
    {
        let layout = Layout::array::<T>(len).expect("invalid slice length");
        let dst = self.alloc_layout(layout).cast::<T>();

        unsafe {
            for i in 0..len {
                ptr::write(dst.as_ptr().add(i), f(i));
            }

            let result = slice::from_raw_parts_mut(dst.as_ptr(), len);
            debug_assert_eq!(Layout::for_value(result), layout);
            result
        }
    }

    /// Copies the given string into the bump allocator and returns an exclusive
    /// reference to it.
    #[inline]
    pub fn alloc_str(&self, str: &str) -> &'a mut str {
        unsafe {
            let ptr = self
                .alloc_layout(Layout::from_size_align_unchecked(str.len(), 1))
                .as_ptr();

            ptr::copy_nonoverlapping(str.as_ptr(), ptr, str.len());
            let slice = slice::from_raw_parts_mut(ptr, str.len());
            str::from_utf8_unchecked_mut(slice)
        }
    }

    /// Allocate space for an object in the bump allocator with the given
    /// [`Layout`].
    ///
    /// The returned pointer points to uninitialized memory.
    #[inline]
    pub fn alloc_layout(&self, layout: Layout) -> NonNull<u8> {
        unsafe { self.world.alloc_layout(layout) }
    }
}

unsafe impl<T: EventSet> HandlerParam for Sender<'_, T> {
    type State = T::Indices;

    type This<'a> = Sender<'a, T>;

    fn init(world: &mut World, config: &mut HandlerConfig) -> Self::State {
        config.set_event_queue_access(Access::ReadWrite);

        let state = T::new_indices(world);

        T::for_each_index(&state, |is_targeted, idx| {
            if is_targeted {
                config.insert_sent_targeted_event(TargetedEventIdx(idx));
            } else {
                config.insert_sent_global_event(GlobalEventIdx(idx));
            }
        });

        state
    }

    unsafe fn get<'a>(
        state: &'a mut Self::State,
        _info: &'a HandlerInfo,
        _event_ptr: EventPtr<'a>,
        _target_location: EntityLocation,
        world: UnsafeWorldCell<'a>,
    ) -> Self::This<'a> {
        Sender { state, world }
    }

    fn refresh_archetype(_state: &mut Self::State, _arch: &Archetype) {}

    fn remove_archetype(_state: &mut Self::State, _arch: &Archetype) {}
}

impl<T: EventSet> fmt::Debug for Sender<'_, T>
where
    T::Indices: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sender")
            .field("state", &self.state)
            .field("world", &self.world)
            .finish()
    }
}

/// A set of [`Event`] types.
///
/// This trait is implemented for all events and tuples of events, so `E1`,
/// `()`, `(E1,)`, `(E1, E2)` etc. are all event sets.
///
/// # Safety
///
/// This trait is marked `unsafe` because unsafe code relies on implementations
/// being correct. It is not recommended to implement this trait yourself.
pub unsafe trait EventSet {
    /// The set of event indices.
    type Indices: 'static;

    /// Create a new set of events.
    fn new_indices(world: &mut World) -> Self::Indices;

    /// Find the event `F` in the set of events. Returns the event index.
    fn find_index<F: Event>(indices: &Self::Indices) -> Option<u32>;

    /// Run a function on every element of the set, passing in the event index
    /// and a boolean indicating if the event is targeted or not.
    fn for_each_index<F: FnMut(bool, u32)>(indices: &Self::Indices, f: F);
}

unsafe impl<E: Event> EventSet for E {
    type Indices = u32;

    fn new_indices(world: &mut World) -> Self::Indices {
        let desc = EventDescriptor::new::<E>(world);

        if TypeId::of::<E::EventIdx>() == TypeId::of::<TargetedEventIdx>() {
            unsafe { world.add_targeted_event_with_descriptor(desc) }
                .index()
                .0
        } else {
            unsafe { world.add_global_event_with_descriptor(desc) }
                .index()
                .0
        }
    }

    #[inline]
    fn find_index<F: Event>(index: &Self::Indices) -> Option<u32> {
        (TypeId::of::<E::This<'static>>() == TypeId::of::<F::This<'static>>()).then_some(*index)
    }

    fn for_each_index<F: FnMut(bool, u32)>(index: &Self::Indices, mut f: F) {
        f(
            TypeId::of::<E::EventIdx>() == TypeId::of::<TargetedEventIdx>(),
            *index,
        )
    }
}

macro_rules! impl_event_set_tuple {
    ($(($E:ident, $e:ident)),*) => {
        #[allow(unused_variables, unused_mut, clippy::unused_unit)]
        unsafe impl<$($E: EventSet),*> EventSet for ($($E,)*) {
            type Indices = ($($E::Indices,)*);

            fn new_indices(world: &mut World) -> Self::Indices {
                (
                    $(
                        $E::new_indices(world),
                    )*
                )
            }

            #[inline]
            fn find_index<F: Event>(($($e,)*): &Self::Indices) -> Option<u32> {
                $(
                    if let Some(id) = $E::find_index::<F>($e) {
                        return Some(id);
                    }
                )*

                None
            }

            fn for_each_index<F: FnMut(bool, u32)>(($($e,)*): &Self::Indices, mut f: F) {
                $(
                    $E::for_each_index($e, &mut f);
                )*
            }
        }
    };
}

all_tuples!(impl_event_set_tuple, 0, 64, E, e);

/// Adds all components of a set to a world and sorts their component indices.
/// Returns a tuple of the permutation used to sort the indices and the indices
/// themselves, or `Err(())` if any component index appeared twice.
// NOTE: It's fine that this returns component indices and not IDs. The indices
// are stored in Insert / Remove events. The IDs of these events are stored in
// the component infos of all components in the set. Whenever any component in
// the set is removed (invalidating its component index), the whole event is
// also removed. This prevents the returned component indices from being used
// after their invalidation.
fn initialize_component_set<C: ComponentSet>(
    world: &mut World,
) -> Result<(Permutation, BitSet<ComponentIdx>), ()> {
    /// Returns `true` if the given sorted slice contains any duplicates. This
    /// function assumes that `a == b` if and only if `a` and `b` are adjacent
    /// in the slice. If this condition is violated, the function will
    /// return a meaningless result, but not cause undefined behaviour.
    fn sorted_slice_contains_duplicates<T: PartialEq>(slice: &[T]) -> bool {
        if slice.is_empty() {
            return false;
        }

        // TODO: Use `slice.array_windows().any(|[a, b]| a == b)` once
        //  `[T]::array_windows` is stabilized.
        for i in 0..slice.len() - 1 {
            let this = &slice[i];
            let next = &slice[i + 1];
            if this == next {
                return true;
            }
        }
        false
    }

    // Add the components to the world and collect their indices.
    let mut unsorted_component_indices = Vec::with_capacity(C::len());
    C::add_components(world, |id| unsorted_component_indices.push(id.index()));

    // Compute a permutation to sort the indices.
    let permutation = Permutation::sorting(&unsorted_component_indices);

    // Apply the permutation, sorting the indices.
    let component_indices = permutation.apply_collect(unsorted_component_indices);
    debug_assert!(
        component_indices.windows(2).all(|w| w[0] <= w[1]),
        "Components should be ordered"
    );

    // Check if there are any duplicates.
    if sorted_slice_contains_duplicates(&component_indices) {
        Err(())
    } else {
        let component_indices = BitSet::<ComponentIdx>::from_iter(component_indices);

        // Return the permutation and the component indices.
        Ok((permutation, component_indices))
    }
}

/// Mirrors the signature of [`C::get_components`].
///
/// [`C::get_components`]: [`ComponentSetInternal::get_components`]
type TypedGetComponentsFn<C> = fn(set: &C, out: &mut ComponentPointerConsumer);

/// A type-erased function pointer matching the signature of
/// [`ComponentSetInternal::get_components`]. Callers of this function pointer
/// must ensure that `set_ptr` points to a component set of the correct type.
pub(crate) type GetComponentsFn = unsafe fn(set_ptr: *const u8, out: &mut ComponentPointerConsumer);

/// Returns a type-erased function pointer to `C::get_components`.
fn get_components_fn_of<C: ComponentSet>() -> GetComponentsFn {
    // Ensure that `get_components` has the signature we expect.
    let f: TypedGetComponentsFn<C> = C::get_components;

    // Cast the function pointer.
    // SAFETY: The signatures of `TypedGetComponentsFn` and `GetComponentsFn`
    // are compatible, as they only differ in their first argument and
    // it is safe to pass `*const u8` where `&C` is expected, as long as the
    // pointee type matches and the lifetime is correct.
    unsafe { transmute(f) }
}

/// A [`TargetedEvent`] which adds all components of a set `C` on an entity when
/// sent. If the entity already has a component, then the component is
/// replaced.
///
/// Any handler which listens for `Insert<C>` will run before the components are
/// inserted. `Insert<C>` has no effect if the target entity does not exist or
/// the event is consumed before it finishes broadcasting.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
pub struct Insert<C>(pub C);

unsafe impl<C: ComponentSet> Event for Insert<C> {
    type This<'a> = Insert<C>;

    type EventIdx = TargetedEventIdx;

    type Mutability = Mutable;

    fn init(world: &mut World) -> EventKind {
        let Ok((permutation, component_indices)) = initialize_component_set::<C>(world) else {
            panic!("component set contains duplicates");
        };
        EventKind::Insert(InsertedComponentsInfo {
            component_indices,
            permutation,
            get_components: get_components_fn_of::<C>(),
        })
    }
}

impl<C> Deref for Insert<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<C> DerefMut for Insert<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A [`TargetedEvent`] which removes all components in the set `C` from an
/// entity when sent. The components are dropped and cannot be recovered.
///
/// Any handler which listens for `Remove<C>` will run before the components are
/// removed. `Remove<C>` has no effect if the target entity does not exist or
/// the event is consumed before it finishes broadcasting.
///
/// This type behaves like a unit struct. Use `Remove::<C>` to instantiate the
/// type.
///
/// # Examples
///
/// ```
/// # use evenio::prelude::*;
/// #
/// # #[derive(Component)]
/// # struct C;
/// #
/// # fn _f(world: &mut World, target: EntityId) {
/// world.send_to(target, Remove::<C>);
/// # }
/// ```
#[derive(Default)]
pub enum Remove<C: ?Sized> {
    // Don't use these variants directly. They are implementation details.
    #[doc(hidden)]
    __Ignore(crate::ignore::Ignore<C>),
    #[doc(hidden)]
    #[default]
    __Value,
}

mod remove_value {
    #[doc(hidden)]
    pub use super::Remove::__Value as Remove;
}

pub use remove_value::*;

use crate::bit_set::BitSet;

impl<C: ?Sized> Copy for Remove<C> {}

impl<C: ?Sized> Clone for Remove<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: ?Sized> fmt::Debug for Remove<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Remove").finish()
    }
}

unsafe impl<C: ComponentSet> Event for Remove<C> {
    type This<'a> = Remove<C>;

    type EventIdx = TargetedEventIdx;

    type Mutability = Mutable;

    fn init(world: &mut World) -> EventKind {
        // TODO: Permutation is not used here. Just sort the component indices
        //  in place without computing a permutation here?
        let Ok((_permutation, component_indices)) = initialize_component_set::<C>(world) else {
            panic!("component set contains duplicates");
        };
        EventKind::Remove(RemovedComponentsInfo { component_indices })
    }
}

/// A [`GlobalEvent`] which signals the creation of an entity. Contains the
/// [`EntityId`] of the new entity, which may or may not exist by the time this
/// event is observed.
///
/// Note that the event by itself cannot be used to spawn new entities. Use
/// [`World::spawn`] or [`Sender::spawn`] instead.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Spawn<C>(pub EntityId, pub C);

unsafe impl<C: ComponentSet> Event for Spawn<C> {
    type This<'a> = Self;

    type EventIdx = GlobalEventIdx;

    type Mutability = Immutable;

    fn init(world: &mut World) -> EventKind {
        let Ok((permutation, component_indices)) = initialize_component_set::<C>(world) else {
            panic!("component set contains duplicates");
        };
        EventKind::Spawn(SpawnInfo {
            components_field_offset: offset_of!(Self, 1),
            inserted_components: InsertedComponentsInfo {
                component_indices,
                permutation,
                get_components: get_components_fn_of::<C>(),
            },
        })
    }
}

/// A [`TargetedEvent`] which removes an entity from the [`World`] when sent.
/// All components of the target entity are dropped.
///
/// Any handler which listens for `Despawn` will run before the entity is
/// removed. `Despawn` has no effect if the target entity does not exist or the
/// event is consumed before it finishes broadcasting.
///
/// # Examples
///
/// ```
/// use evenio::prelude::*;
///
/// let mut world = World::new();
///
/// let id = world.spawn(());
///
/// assert!(world.entities().contains(id));
///
/// world.add_handler(|r: Receiver<Despawn, EntityId>| {
///     println!("{:?} is about to despawn!", r.query);
/// });
///
/// world.send_to(id, Despawn);
///
/// assert!(!world.entities().contains(id));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Despawn;

unsafe impl Event for Despawn {
    type This<'a> = Despawn;

    type EventIdx = TargetedEventIdx;

    type Mutability = Mutable;

    fn init(_world: &mut World) -> EventKind {
        EventKind::Despawn
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use crate::prelude::*;

    #[test]
    fn change_entity_during_broadcast() {
        let mut world = World::new();

        #[derive(TargetedEvent)]
        struct E;

        #[derive(Component)]
        struct C(String);

        world.add_handler(|r: Receiver<E, EntityId>, s: Sender<Remove<C>>| {
            s.remove::<C>(r.query);
        });

        world.add_handler(|r: Receiver<E, &mut C>| {
            r.query.0.push_str("123");
        });

        let e = world.spawn(());
        world.insert(e, C("abc".into()));

        world.send_to(e, E);
    }

    #[test]
    fn event_order() {
        #[derive(GlobalEvent)]
        struct A;
        #[derive(GlobalEvent, Debug)]
        struct B(i32);
        #[derive(GlobalEvent, Debug)]
        struct C(i32);

        #[derive(Component)]
        struct Result(Vec<i32>);

        fn get_a_send_b(_: Receiver<A>, sender: Sender<B>) {
            sender.send(B(0));
            sender.send(B(3));
        }

        fn get_b_send_c(r: Receiver<B>, sender: Sender<C>, mut res: Single<&mut Result>) {
            res.0.push(r.event.0);
            sender.send(C(r.event.0 + 1));
            sender.send(C(r.event.0 + 2));
        }

        fn get_c(r: Receiver<C>, mut res: Single<&mut Result>) {
            res.0.push(r.event.0);
        }

        let mut world = World::new();

        let res = world.spawn(());
        world.insert(res, Result(vec![]));

        world.add_handler(get_a_send_b);
        world.add_handler(get_b_send_c);
        world.add_handler(get_c);

        world.send(A);

        assert_eq!(
            world.get::<&Result>(res).unwrap().0.as_slice(),
            &[0, 1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn despawn_many() {
        let mut world = World::new();

        #[derive(GlobalEvent)]
        struct E;

        #[derive(Component)]
        struct C(#[allow(unused)] i32);

        let mut entities = vec![];

        let n = 50;

        for i in 0..n {
            let e = world.spawn(());
            world.insert(e, C(i));
            entities.push(e);
        }

        entities.shuffle(&mut rand::thread_rng());

        world.add_handler(move |_: Receiver<E>, s: Sender<Despawn>| {
            for &e in &entities {
                s.despawn(e);
            }
        });

        world.send(E);

        assert_eq!(world.entities().len(), 0);
    }

    #[test]
    fn send_borrowed() {
        let mut buf = [1, 2, 3];

        #[derive(GlobalEvent, Debug)]
        struct A<'a>(&'a mut [i32]);

        impl Drop for A<'_> {
            fn drop(&mut self) {
                for item in self.0.iter_mut() {
                    *item *= 2;
                    println!("{item}");
                }
            }
        }

        let mut world = World::new();

        world.add_handler(|r: Receiver<A>| println!("{r:?}"));

        world.send(A(&mut buf));
    }

    #[test]
    #[should_panic]
    fn global_event_not_in_event_set() {
        let mut world = World::new();

        #[derive(GlobalEvent)]
        struct A;

        #[derive(GlobalEvent)]
        struct B;

        world.add_handler(|_: Receiver<A>, s: Sender<B>| {
            s.send(A);
        });

        world.send(A);
    }

    #[test]
    #[should_panic]
    fn targeted_event_not_in_event_set() {
        let mut world = World::new();

        #[derive(GlobalEvent)]
        struct A;

        #[derive(TargetedEvent)]
        struct B;

        world.add_handler(|_: Receiver<A>, s: Sender<A>| {
            s.send_to(EntityId::NULL, B);
        });

        world.send(A);
    }

    #[test]
    fn send_event_from_sender_with_lifetime() {
        let mut world = World::new();

        #[derive(GlobalEvent)]
        struct A;

        #[derive(GlobalEvent)]
        struct B<'a> {
            slice: &'a [i32],
            string: &'a str,
            array: &'a [u64; 4],
        }

        fn get_a_send_b(_: Receiver<A>, s: Sender<B>) {
            s.send(B {
                slice: s.alloc_slice(5, |i| i as i32 + 1),
                string: s.alloc_str("pineapple"),
                array: s.alloc([10, 20, 30, 40]),
            });
        }

        fn get_b(r: Receiver<B>) {
            assert_eq!(r.event.slice, &[1, 2, 3, 4, 5]);
            assert_eq!(r.event.string, "pineapple");
            assert_eq!(r.event.array, &[10, 20, 30, 40]);
        }

        world.add_handler(get_a_send_b);
        world.add_handler(get_b);
        world.send(A);
    }

    #[test]
    fn more_than_one_sender() {
        let mut world = World::new();

        #[derive(GlobalEvent)]
        struct A(#[allow(dead_code)] u32);

        #[derive(GlobalEvent)]
        struct B(#[allow(dead_code)] u32);

        fn send_b_x2(_: Receiver<A>, s1: Sender<B>, s2: Sender<B>) {
            s1.send(B(123));
            s2.send(B(456));
        }

        world.add_handler(send_b_x2);

        world.send(A(123));
    }

    #[allow(unused, clippy::type_complexity)]
    mod derive_event {
        use core::marker::PhantomData;

        use super::*;

        #[derive(GlobalEvent)]
        struct StructWithLifetime_<'a> {
            _ignore: &'a (),
        }

        #[derive(TargetedEvent)]
        struct StructWithLifetime<'a> {
            _ignore: &'a (),
        }

        #[derive(GlobalEvent)]
        struct StructWithGenericType_<T>(T);

        #[derive(TargetedEvent)]
        struct StructWithGenericType<T>(T);

        #[derive(GlobalEvent)]
        struct StructWithBoth_<'a, T>(PhantomData<(fn() -> T, &'a ())>);

        #[derive(TargetedEvent)]
        struct StructWithBoth<'a, T>(PhantomData<(fn() -> T, &'a ())>);
    }
}
