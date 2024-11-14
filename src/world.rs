//! Defines the [`World`] and related APIs.

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};
use core::alloc::Layout;
use core::any::{self, TypeId};
use core::fmt::Write;
use core::marker::PhantomData;
use core::mem;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use bumpalo::Bump;

use crate::access::ComponentAccess;
use crate::archetype::Archetypes;
use crate::bit_set::BitSet;
use crate::component::{
    AddComponent, Component, ComponentDescriptor, ComponentId, ComponentIdx, ComponentInfo,
    ComponentPointerConsumer, ComponentSet, Components, RemoveComponent,
};
use crate::drop::{drop_fn_of, DropFn};
use crate::entity::{Entities, EntityId, EntityLocation, ReservedEntities};
use crate::event::{
    AddGlobalEvent, AddTargetedEvent, Despawn, EventDescriptor, EventKind, EventMeta, EventPtr,
    EventQueueItem, GlobalEvent, GlobalEventId, GlobalEventIdx, GlobalEventInfo, GlobalEvents,
    Insert, InsertedComponentsInfo, Remove, RemoveGlobalEvent, RemoveTargetedEvent,
    RemovedComponentsInfo, Spawn, SpawnInfo, TargetedEvent, TargetedEventId, TargetedEventIdx,
    TargetedEventInfo, TargetedEvents,
};
use crate::fetch;
use crate::fetch::FetcherState;
use crate::handler::{
    AddHandler, Handler, HandlerConfig, HandlerId, HandlerInfo, HandlerInfoInner, HandlerList,
    Handlers, IntoHandler, MaybeInvalidAccess, ReceivedEventId, RemoveHandler,
};
use crate::map::IndexSet;
use crate::mutability::Mutability;
use crate::query::{Query, ReadOnlyQuery};

/// A container for all data in the ECS. This includes entities, components,
/// handlers, and events.
#[derive(Debug)]
pub struct World {
    entities: Entities,
    reserved_entities: ReservedEntities,
    components: Components,
    handlers: Handlers,
    archetypes: Archetypes,
    global_events: GlobalEvents,
    targeted_events: TargetedEvents,
    event_queue: Vec<EventQueueItem>,
    // Used in flush_event_queue to avoid frequent allocations.
    component_pointer_buffer: Vec<MaybeUninit<*const u8>>,
    bump: Bump,
    /// So the world doesn't accidentally implement `Send` or `Sync`.
    _marker: PhantomData<*const ()>,
}

impl World {
    /// Creates a new, empty world.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// let mut world = World::new();
    /// ```
    pub fn new() -> Self {
        Self {
            entities: Entities::new(),
            reserved_entities: ReservedEntities::new(),
            components: Components::new(),
            handlers: Handlers::new(),
            archetypes: Archetypes::new(),
            global_events: GlobalEvents::new(),
            targeted_events: TargetedEvents::new(),
            event_queue: vec![],
            component_pointer_buffer: vec![],
            bump: Bump::new(),
            _marker: PhantomData,
        }
    }

    /// Broadcast a global event to all handlers in this world. All handlers
    /// which listen for this event
    ///
    /// Any events sent by handlers will also broadcast. This process continues
    /// recursively until all events have finished broadcasting.
    ///
    /// See also [`World::send_to`] to send a [`TargetedEvent`].
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(GlobalEvent)]
    /// struct MyEvent(i32);
    ///
    /// fn my_handler(r: Receiver<MyEvent>) {
    ///     println!("got event: {}", r.event.0);
    /// }
    ///
    /// let mut world = World::new();
    ///
    /// world.add_handler(my_handler);
    /// world.send(MyEvent(123));
    /// ```
    ///
    /// Output:
    ///
    /// ```txt
    /// got event: 123
    /// ```
    pub fn send<E: GlobalEvent>(&mut self, event: E) {
        let idx = self.add_global_event::<E>().index();

        self.event_queue.push(EventQueueItem {
            meta: EventMeta::Global { idx },
            event: NonNull::from(self.bump.alloc(event)).cast(),
        });

        self.flush_event_queue();
    }

    /// Broadcast a targeted event to all handlers in this world.
    ///
    /// Any events sent by handlers will also broadcast. This process continues
    /// recursively until all events have finished broadcasting.
    ///
    /// See also [`World::send`] to send a [`GlobalEvent`].
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(TargetedEvent)]
    /// struct MyEvent(i32);
    ///
    /// fn my_handler(r: Receiver<MyEvent, EntityId>) {
    ///     println!("target of received event is {:?}", r.query);
    /// }
    ///
    /// let mut world = World::new();
    ///
    /// world.add_handler(my_handler);
    ///
    /// let target = world.spawn(());
    ///
    /// // Send my event to `target` entity.
    /// world.send_to(target, MyEvent(123));
    /// ```
    pub fn send_to<E: TargetedEvent>(&mut self, target: EntityId, event: E) {
        let idx = self.add_targeted_event::<E>().index();

        self.event_queue.push(EventQueueItem {
            meta: EventMeta::Targeted { target, idx },
            event: NonNull::from(self.bump.alloc(event)).cast(),
        });

        self.flush_event_queue();
    }

    /// Creates a new entity, returns its [`EntityId`], and sends the [`Spawn`]
    /// event to signal its creation. The entity is spawned with the given set
    /// of components attached. Note that this set can be empty.
    ///
    /// The returned `EntityId` will not have been used by any previous entities
    /// in this world.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(Component)]
    /// struct ComponentA;
    ///
    /// #[derive(Component)]
    /// struct ComponentB;
    ///
    /// let mut world = World::new();
    ///
    /// // Spawn entity without components.
    /// let id_1 = world.spawn(());
    ///
    /// // Spawn entity with components.
    /// let id_2 = world.spawn((ComponentA, ComponentB));
    ///
    /// assert!(world.entities().contains(id_1));
    /// assert!(world.entities().contains(id_2));
    /// ```
    pub fn spawn<C: ComponentSet>(&mut self, components: C) -> EntityId {
        let id = self.reserved_entities.reserve(&self.entities);

        self.send(Spawn(id, components));

        id
    }

    // TODO: Document batch component insertion and removal.
    /// Sends the [`Insert`] event.
    ///
    /// This is equivalent to:
    ///
    /// ```
    /// # use evenio::prelude::*;
    /// #
    /// # let mut world = World::new();
    /// #
    /// # let entity = world.spawn(());
    /// #
    /// # #[derive(Component)]
    /// # struct C;
    /// #
    /// # let component = C;
    /// #
    /// world.send_to(entity, Insert(component));
    /// ```
    pub fn insert<C: ComponentSet>(&mut self, entity: EntityId, components: C) {
        self.send_to(entity, Insert(components))
    }

    /// Sends the [`Remove`] event.
    ///
    /// This is equivalent to:
    ///
    /// ```
    /// # use evenio::prelude::*;
    /// #
    /// # let mut world = World::new();
    /// #
    /// # let entity = world.spawn(());
    /// #
    /// # #[derive(Component)]
    /// # struct C;
    /// #
    /// world.send_to(entity, Remove::<C>);
    /// ```
    pub fn remove<C: Component>(&mut self, entity: EntityId) {
        self.send_to(entity, Remove::<C>)
    }

    /// Sends the [`Despawn`] event.
    ///
    /// This is equivalent to:
    ///
    /// ```
    /// # use evenio::prelude::*;
    /// #
    /// # let mut world = World::new();
    /// #
    /// # let entity = world.spawn(());
    /// #
    /// world.send_to(entity, Despawn);
    /// ```
    pub fn despawn(&mut self, entity: EntityId) {
        self.send_to(entity, Despawn)
    }

    fn conflicts_error_message(&self, mut write: impl Write, conflicts: IndexSet<ComponentIdx>) {
        writeln!(write, "conflicting components are...").unwrap();

        for &idx in &conflicts {
            write!(write, "- ").unwrap();
            match self.components.get_by_index(idx) {
                Some(info) => writeln!(write, "{}", info.name()).unwrap(),
                None => writeln!(write, "{idx:?}").unwrap(),
            };
        }
    }

    /// Queries an entity with a read-only query. Returns `None` if the entity
    /// doesn't exist or doesn't match the query.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(Component, PartialEq, Debug)]
    /// struct MyComponent(i32);
    ///
    /// let mut world = World::new();
    ///
    /// let e = world.spawn(());
    /// world.insert(e, MyComponent(123));
    ///
    /// assert_eq!(world.get::<&MyComponent>(e), Some(&MyComponent(123)));
    /// ```
    pub fn get<Q: ReadOnlyQuery>(&self, entity: EntityId) -> Option<Q::This<'_>> {
        let loc = self.entities.get(entity)?;

        let arch = unsafe { self.archetypes().get(loc.archetype).unwrap_unchecked() };

        let state = Q::get_new_state(self)?;
        let arch_state = Q::new_arch_state(arch, &state)?;
        // SAFETY: Don't need to check component access here, because the query
        // is read-only anyway (no aliased mutability possible).
        unsafe { Some(Q::get(&arch_state, loc.row)) }
    }

    /// Queries an entity. Returns `None` if the entity doesn't exist or doesn't
    /// match the query.
    ///
    /// # Panics
    ///
    /// Panics if the query would result in aliased mutability.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(Component, PartialEq, Debug)]
    /// struct MyComponent(i32);
    ///
    /// let mut world = World::new();
    ///
    /// let e = world.spawn(());
    /// world.insert(e, MyComponent(123));
    ///
    /// assert_eq!(
    ///     world.get_mut::<&mut MyComponent>(e),
    ///     Some(&mut MyComponent(123))
    /// );
    /// ```
    pub fn get_mut<Q: Query>(&self, entity: EntityId) -> Option<Q::This<'_>> {
        let loc = self.entities.get(entity)?;

        let arch = unsafe { self.archetypes().get(loc.archetype).unwrap_unchecked() };

        let state = Q::get_new_state(self)?;
        let arch_state = Q::new_arch_state(arch, &state)?;
        // TODO: Cache this component access?
        let access = Q::get_access(&state, |_idx| {});
        let conflicts = access.collect_conflicts();
        if !conflicts.is_empty() {
            let mut message = "called World::get_mut with a query that has conflicting component \
                               access (aliased mutability)\n"
                .to_owned();

            self.conflicts_error_message(&mut message, conflicts);

            panic!("{}", message);
        }
        // SAFETY: Just checked component access.
        unsafe { Some(Q::get(&arch_state, loc.row)) }
    }

    /// Iterates over all results of a read-only query. Returns `None` if any
    /// component referenced by the query does not exist.
    pub fn iter<Q: ReadOnlyQuery>(&self) -> Option<fetch::IntoIter<Q>> {
        let query_state = Q::get_new_state(self)?;
        let access = Q::get_access(&query_state, |_idx| {});
        let mut fetcher_state = FetcherState::new(query_state);
        self.archetypes.init_fetcher(&mut fetcher_state, &access);
        // SAFETY: Don't need to check component access here, because the query
        // is read-only anyway (no aliased mutability possible).
        unsafe { Some(fetcher_state.into_iter(&self.archetypes)) }
    }

    /// Iterates over all results of a query. Returns `None` if any component
    /// referenced by the query does not exist.
    ///
    /// # Panics
    ///
    /// Panics if the query would result in aliased mutability.
    pub fn iter_mut<Q: Query>(&mut self) -> Option<fetch::IntoIter<Q>> {
        let query_state = Q::get_new_state(self)?;
        let access = Q::get_access(&query_state, |_idx| {});
        let conflicts = access.collect_conflicts();
        if !conflicts.is_empty() {
            let mut message = "called World::iter_mut with a query that has conflicting component \
                               access (aliased mutability)\n"
                .to_owned();

            self.conflicts_error_message(&mut message, conflicts);

            panic!("{}", message);
        }
        let mut fetcher_state = FetcherState::new(query_state);
        self.archetypes.init_fetcher(&mut fetcher_state, &access);
        // SAFETY: Just checked component access.
        unsafe { Some(fetcher_state.into_iter(&self.archetypes)) }
    }

    /// Runs a read-only query for all entities and returns the only result.
    /// This is useful for storing global resources as components that only one
    /// entity has. Returns `None` if any component referenced by the query does
    /// not exist.
    ///
    /// # Panics
    ///
    /// Panics if there is no or more than one entity that matches the query.
    pub fn single<Q: ReadOnlyQuery>(&self) -> Option<Q::This<'_>> {
        let mut iter = self.iter::<Q>()?;

        let Some(item) = iter.next() else {
            panic!("called World::single with a query that does not match any entity")
        };

        assert!(
            iter.next().is_none(),
            "called World::single with a query that matches more than one entity"
        );

        Some(item)
    }

    /// Runs a query for all entities and returns the only result. This is
    /// useful for storing global resources as components that only one entity
    /// has. Returns `None` if any component referenced by the query does
    /// not exist.
    ///
    /// # Panics
    ///
    /// Panics if there is no or more than one entity that matches the query, or
    /// if the query would result in aliased mutability.
    pub fn single_mut<Q: Query>(&mut self) -> Option<Q::This<'_>> {
        let mut iter = self.iter_mut::<Q>()?;

        let Some(item) = iter.next() else {
            panic!("called World::single_mut with a query that does not match any entity")
        };

        assert!(
            iter.next().is_none(),
            "called World::single_mut with a query that matches more than one entity"
        );

        Some(item)
    }

    /// Adds a new handler to the world, returns its [`HandlerId`], and sends
    /// the [`AddHandler`] event to signal its creation.
    ///
    /// If the handler already exists (as determined by [`Handler::type_id`])
    /// then the `HandlerId` of the existing handler is returned and no
    /// event is sent.
    ///
    /// Returns an error if the configuration of the handler is invalid.
    pub(crate) fn try_add_handler<H: IntoHandler<M>, M>(
        &mut self,
        handler: H,
    ) -> Result<HandlerId, String> {
        let mut handler = handler.into_handler();
        let mut config = HandlerConfig::default();

        let type_id = handler.type_id();

        if let Some(type_id) = type_id {
            if let Some(info) = self.handlers.get_by_type_id(type_id) {
                return Ok(info.id());
            }
        }

        let handler_name = handler.name();

        handler.init(self, &mut config);

        let received_event = match config.received_event {
            ReceivedEventId::None => {
                return Err(format!(
                    "handler {handler_name} did not specify an event to receive"
                ));
            }
            ReceivedEventId::Ok(event) => event,
            ReceivedEventId::Invalid => {
                return Err(format!(
                    "handler {handler_name} attempted to listen for more than one event type"
                ))
            }
        };

        let received_event_access = match config.received_event_access {
            MaybeInvalidAccess::Ok(access) => access,
            MaybeInvalidAccess::Invalid => {
                return Err(format!(
                    "handler {handler_name} has conflicting access to the received event"
                ))
            }
        };

        let component_access_conjunction = config
            .component_accesses
            .iter()
            .fold(ComponentAccess::new_true(), |acc, a| acc.and(a));

        let conflicts = component_access_conjunction.collect_conflicts();

        if !conflicts.is_empty() {
            let mut message = "handler {handler_name} contains conflicting component access \
                               (aliased mutability)\n"
                .into();

            self.conflicts_error_message(&mut message, conflicts);

            return Err(message);
        }

        let component_access_disjunction = config
            .component_accesses
            .iter()
            .fold(ComponentAccess::new_false(), |acc, a| acc.or(a));

        let info = HandlerInfo::new(HandlerInfoInner {
            name: handler_name,
            id: HandlerId::NULL, // Filled in later.
            type_id,
            order: 0, // Filled in later.
            received_event,
            received_event_access,
            targeted_event_component_access: config.targeted_event_component_access,
            sent_untargeted_events: config.sent_global_events,
            sent_targeted_events: config.sent_targeted_events,
            component_access: component_access_conjunction,
            archetype_filter: component_access_disjunction,
            referenced_components: config.referenced_components,
            priority: config.priority,
            handler,
        });

        let id = self.handlers.add(info);
        let info = self.handlers.get_mut(id).unwrap();

        self.archetypes.register_handler(info);

        self.send(AddHandler(id));

        Ok(id)
    }

    /// Adds a new handler to the world, returns its [`HandlerId`], and sends
    /// the [`AddHandler`] event to signal its creation.
    ///
    /// If the handler already exists (as determined by [`Handler::type_id`])
    /// then the `HandlerId` of the existing handler is returned and no
    /// event is sent.
    ///
    /// # Panics
    ///
    /// Panics if the configuration of the handler is invalid. This can occur
    /// when, for instance, the handler does not specify an event to receive.
    ///
    /// ```should_panic
    /// # use evenio::prelude::*;
    /// #
    /// # let mut world = World::new();
    /// #
    /// world.add_handler(|| {}); // Panics
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    /// # #[derive(GlobalEvent)]
    /// # struct MyEvent;
    ///
    /// fn my_handler(_: Receiver<MyEvent>) {};
    ///
    /// let mut world = World::new();
    /// let id = world.add_handler(my_handler);
    ///
    /// assert!(world.handlers().contains(id));
    /// ```
    ///
    /// [`Handler::type_id`]: crate::handler::Handler::type_id
    #[track_caller]
    pub fn add_handler<H: IntoHandler<M>, M>(&mut self, handler: H) -> HandlerId {
        match self.try_add_handler(handler) {
            Ok(id) => id,
            Err(e) => panic!("{e}"),
        }
    }

    /// Removes a handler from the world, returns its [`HandlerInfo`], and sends
    /// the [`RemoveHandler`] event. If the `handler` ID is invalid, then `None`
    /// is returned and no event is sent.
    ///
    /// # Example
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// let mut world = World::new();
    ///
    /// # #[derive(GlobalEvent)]
    /// # struct MyEvent;
    /// let handler_id = world.add_handler(|_: Receiver<MyEvent>| {});
    ///
    /// let info = world.remove_handler(handler_id).unwrap();
    ///
    /// assert_eq!(info.id(), handler_id);
    /// assert!(!world.handlers().contains(handler_id));
    /// ```
    pub fn remove_handler(&mut self, handler: HandlerId) -> Option<HandlerInfo> {
        if !self.handlers.contains(handler) {
            return None;
        }

        self.send(RemoveHandler(handler));

        let info = self.handlers.remove(handler).unwrap();

        self.archetypes.remove_handler(&info);

        Some(info)
    }

    /// Adds the component `C` to the world, returns its [`ComponentId`], and
    /// sends the [`AddComponent`] event to signal its creation.
    ///
    /// If the component already exists, then the [`ComponentId`] of the
    /// existing component is returned and no event is sent.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(Component)]
    /// struct MyComponent;
    ///
    /// let mut world = World::new();
    /// let id = world.add_component::<MyComponent>();
    ///
    /// assert_eq!(id, world.add_component::<MyComponent>());
    /// ```
    pub fn add_component<C: Component>(&mut self) -> ComponentId {
        let desc = ComponentDescriptor {
            name: any::type_name::<C>().into(),
            type_id: Some(TypeId::of::<C>()),
            layout: Layout::new::<C>(),
            drop: drop_fn_of::<C>(),
            mutability: Mutability::of::<C::Mutability>(),
        };

        unsafe { self.add_component_with_descriptor(desc) }
    }

    /// Adds all components of the set `C` to the world, returns their
    /// [`ComponentId`]s, and sends the [`AddComponent`] event for each added
    /// component to signal its creation.
    ///
    /// If a component already exists, then the [`ComponentId`] of the
    /// existing component is returned and no event is sent for that component.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(Component)]
    /// struct MyComponent;
    ///
    /// #[derive(Component)]
    /// struct MyOtherComponent;
    ///
    /// let mut world = World::new();
    /// let ids = world.add_components::<(MyComponent, MyOtherComponent)>();
    ///
    /// assert_eq!(
    ///     ids,
    ///     world.add_components::<(MyComponent, MyOtherComponent)>()
    /// );
    /// ```
    pub fn add_components<C: ComponentSet>(&mut self) -> Vec<ComponentId> {
        // TODO: Use an ArrayVec here and in other places where we call
        //  ComponentSet::add_components or ::remove_components? We know the
        //  maximum capacity, and this would save us a heap allocation.
        let mut ids = Vec::with_capacity(C::len());
        C::add_components(self, |id| ids.push(id));
        ids
    }

    /// Adds a component described by a given [`ComponentDescriptor`].
    ///
    /// Like [`add_component`], an [`AddComponent`] event is sent if the
    /// component is newly added. If the [`TypeId`] of the component matches an
    /// existing component, then the existing component's [`ComponentId`] is
    /// returned and no event is sent.
    ///
    /// # Safety
    ///
    /// - If the component is given a [`TypeId`], then the `layout` and `drop`
    ///   function must be compatible with the Rust type identified by the type
    ///   ID.
    /// - Drop function must be safe to call with a pointer to the component as
    ///   described by [`DropFn`]'s documentation.
    ///
    /// [`add_component`]: World::add_component
    pub unsafe fn add_component_with_descriptor(
        &mut self,
        desc: ComponentDescriptor,
    ) -> ComponentId {
        let (id, is_new) = self.components.add(desc);

        if is_new {
            self.send(AddComponent(id));
        }

        id
    }

    /// Removes a component from the world and returns its [`ComponentInfo`]. If
    /// the `component` ID is invalid, then `None` is returned and the function
    /// has no effect.
    ///
    /// Removing a component has the following effects in the order listed:
    /// 1. The [`RemoveComponent`] event is sent.
    /// 2. All entities with the component are despawned.
    /// 3. All handlers that reference the component are removed.
    /// 4. The corresponding [`Insert`] events for the component are removed.
    /// 5. The corresponding [`Remove`] events for the component are removed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use evenio::prelude::*;
    /// # let mut world = World::new();
    /// # #[derive(Component)] struct C;
    /// # #[derive(GlobalEvent)] struct E;
    /// #
    /// let component = world.add_component::<C>();
    /// let handler = world.add_handler(|_: Receiver<E>, _: Fetcher<&C>| {});
    ///
    /// assert!(world.components().contains(component));
    /// assert!(world.handlers().contains(handler));
    ///
    /// world.remove_component_by_id(component);
    ///
    /// assert!(!world.components().contains(component));
    /// // Handler was also removed because it references `C` in its `Fetcher`.
    /// assert!(!world.handlers().contains(handler));
    /// ```
    pub fn remove_component_by_id(&mut self, component: ComponentId) -> Option<ComponentInfo> {
        if !self.components.contains(component) {
            return None;
        }

        // Send event first.
        self.send(RemoveComponent(component));

        let despawn_idx = self.add_targeted_event::<Despawn>().index();

        // Attempt to despawn all entities that still have this component.
        for arch in self.archetypes.iter() {
            if arch.column_of(component.index()).is_some() {
                for &entity_id in arch.entity_ids() {
                    self.event_queue.push(EventQueueItem {
                        meta: EventMeta::Targeted {
                            idx: despawn_idx,
                            target: entity_id,
                        },
                        event: NonNull::<Despawn>::dangling().cast(),
                    });
                }
            }
        }

        self.flush_event_queue();

        // Remove all handlers that reference this component.
        let mut handlers_to_remove = vec![];

        for handler in self.handlers.iter() {
            if handler.references_component(component.index()) {
                handlers_to_remove.push(handler.id());
            }
        }

        for handler_id in handlers_to_remove {
            self.remove_handler(handler_id);
        }

        let info = &self.components[component];

        // Remove all the `Spawn`, `Insert` and `Remove` events for this
        // component.
        // TODO: Merge these sets into one?
        let events_to_remove = info
            .spawn_events()
            .iter()
            .copied()
            .chain(info.insert_events().iter().copied())
            .chain(info.remove_events().iter().copied())
            .collect::<Vec<_>>();

        for event in events_to_remove {
            self.remove_targeted_event(event);
        }

        let mut info = self
            .components
            .remove(component)
            .expect("component should still exist");

        // Remove all archetypes with this component. If there are still entities with
        // the component by this point, then they will be silently removed.
        self.archetypes
            .remove_component(&mut info, &mut self.components, |id| {
                self.entities.remove(id);
            });

        Some(info)
    }

    /// Removes a component from the world and returns its [`ComponentInfo`]. If
    /// the component does not exist, then `None` is returned and the function
    /// has no effect.
    ///
    /// Removing a component has the following effects in the order listed:
    /// 1. The [`RemoveComponent`] event is sent.
    /// 2. All entities with the component are despawned.
    /// 3. All handlers that reference the component are removed.
    /// 4. The corresponding [`Insert`] events for the component are removed.
    /// 5. The corresponding [`Remove`] events for the component are removed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use evenio::prelude::*;
    /// # let mut world = World::new();
    /// # #[derive(Component)] struct C;
    /// # #[derive(GlobalEvent)] struct E;
    /// #
    /// let component = world.add_component::<C>();
    /// let handler = world.add_handler(|_: Receiver<E>, _: Fetcher<&C>| {});
    ///
    /// assert!(world.components().contains(component));
    /// assert!(world.handlers().contains(handler));
    ///
    /// world.remove_component::<C>();
    ///
    /// assert!(!world.components().contains(component));
    /// // Handler was also removed because it references `C` in its `Fetcher`.
    /// assert!(!world.handlers().contains(handler));
    /// ```
    pub fn remove_component<C: Component>(&mut self) -> Option<ComponentInfo> {
        let info = self.components.get_by_type_id(TypeId::of::<C>())?;
        self.remove_component_by_id(info.id())
    }

    /// Removes all components of the set `C` from the world, and returns their
    /// [`ComponentInfo`]s. If a component in the set does not exist, the
    /// returned collection will not contain a [`ComponentInfo`] for it.
    ///
    /// See [`remove_component`] for the effects of this function for each
    /// component in the set.
    ///
    /// If a component already exists, then the [`ComponentId`] of the
    /// existing component is returned and no event is sent for that component.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(GlobalEvent)]
    /// struct MyEvent;
    ///
    /// #[derive(Component)]
    /// struct MyComponent;
    ///
    /// #[derive(Component)]
    /// struct MyOtherComponent;
    ///
    /// let mut world = World::new();
    /// let [my_component, my_other_component] =
    ///     world.add_components::<(MyComponent, MyOtherComponent)>()[..]
    /// else {
    ///     unreachable!("add_components returns one ID for each added component");
    /// };
    ///
    /// let my_handler = world.add_handler(|_: Receiver<MyEvent>, _: Fetcher<&MyComponent>| {});
    /// let my_other_handler =
    ///     world.add_handler(|_: Receiver<MyEvent>, _: Fetcher<&MyOtherComponent>| {});
    ///
    /// assert!(world.components().contains(my_component));
    /// assert!(world.components().contains(my_other_component));
    /// assert!(world.handlers().contains(my_handler));
    /// assert!(world.handlers().contains(my_other_handler));
    ///
    /// world.remove_components::<(MyComponent, MyOtherComponent)>();
    ///
    /// assert!(!world.components().contains(my_component));
    /// assert!(!world.components().contains(my_other_component));
    /// // Handlers were also removed because they referenced removed components
    /// // in their `Fetcher`s.
    /// assert!(!world.handlers().contains(my_handler));
    /// assert!(!world.handlers().contains(my_other_handler));
    /// ```
    ///
    /// [`remove_component`]: World::remove_component
    pub fn remove_components<C: ComponentSet>(&mut self) -> Vec<ComponentInfo> {
        let mut infos = Vec::with_capacity(C::len());
        C::remove_components(self, |id| infos.push(id));
        infos
    }

    /// Adds the global event `E` to the world, returns its [`GlobalEventId`],
    /// and sends the [`AddGlobalEvent`] event to signal its creation.
    ///
    /// If the event already exists, then the [`GlobalEventId`] of the existing
    /// event is returned and no event is sent.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(GlobalEvent)]
    /// struct MyEvent;
    ///
    /// let mut world = World::new();
    /// let id = world.add_global_event::<MyEvent>();
    ///
    /// assert_eq!(id, world.add_global_event::<MyEvent>());
    /// ```
    pub fn add_global_event<E: GlobalEvent>(&mut self) -> GlobalEventId {
        // Returning early here if the event already exists saves us the cost of
        // unnecessarily initializing the event in EventDescriptor::new.
        if let Some(info) = self
            .global_events
            .get_by_type_id(TypeId::of::<E::This<'static>>())
        {
            return info.id();
        }
        let desc = EventDescriptor::new::<E>(self);
        unsafe { self.add_global_event_with_descriptor(desc) }
    }

    /// Adds the targeted event `E` to the world, returns its
    /// [`TargetedEventId`], and sends the [`AddTargetedEvent`] event to
    /// signal its creation.
    ///
    /// If the event already exists, then the [`TargetedEventId`] of the
    /// existing event is returned and no event is sent.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(TargetedEvent)]
    /// struct MyEvent;
    ///
    /// let mut world = World::new();
    /// let id = world.add_targeted_event::<MyEvent>();
    ///
    /// assert_eq!(id, world.add_targeted_event::<MyEvent>());
    /// ```
    pub fn add_targeted_event<E: TargetedEvent>(&mut self) -> TargetedEventId {
        // Returning early here if the event already exists saves us the cost of
        // unnecessarily initializing the event in EventDescriptor::new.
        if let Some(info) = self
            .targeted_events
            .get_by_type_id(TypeId::of::<E::This<'static>>())
        {
            return info.id();
        }
        let desc = EventDescriptor::new::<E>(self);
        unsafe { self.add_targeted_event_with_descriptor(desc) }
    }

    /// Adds a global event described by a given [`EventDescriptor`].
    ///
    /// Like [`add_global_event`], an [`AddGlobalEvent`] event is sent if the
    /// event is newly added. If the [`TypeId`] of the event matches an
    /// existing global event, then the existing event's [`GlobalEventId`] is
    /// returned and no event is sent.
    ///
    /// # Safety
    ///
    /// - If the event is given a [`TypeId`], then the `layout` and `drop`
    ///   function must be compatible with the Rust type identified by the type
    ///   ID.
    /// - Drop function must be safe to call with a pointer to the event as
    ///   described by [`DropFn`]'s documentation.
    /// - The event's kind must be correct for the descriptor. See [`EventKind`]
    ///   for more information.
    ///
    /// [`add_global_event`]: World::add_global_event
    pub unsafe fn add_global_event_with_descriptor(
        &mut self,
        desc: EventDescriptor,
    ) -> GlobalEventId {
        let (id, is_new) = self.global_events.add(desc);

        if is_new {
            self.handlers.register_event(id.index());
            self.send(AddGlobalEvent(id));
        }

        id
    }

    /// Adds a targeted event described by a given [`EventDescriptor`].
    ///
    /// Like [`add_targeted_event`], an [`AddTargetedEvent`] event is sent if
    /// the event is newly added. If the [`TypeId`] of the event matches an
    /// existing targeted event, then the existing event's [`GlobalEventId`] is
    /// returned and no event is sent.
    ///
    /// # Safety
    ///
    /// - If the event is given a [`TypeId`], then the `layout` and `drop`
    ///   function must be compatible with the Rust type identified by the type
    ///   ID.
    /// - Drop function must be safe to call with a pointer to the event as
    ///   described by [`DropFn`]'s documentation.
    /// - The event's kind must be correct for the descriptor. See [`EventKind`]
    ///   for more information.
    ///
    /// [`add_targeted_event`]: World::add_targeted_event
    pub unsafe fn add_targeted_event_with_descriptor(
        &mut self,
        desc: EventDescriptor,
    ) -> TargetedEventId {
        let (id, is_new) = self.targeted_events.add(desc);

        if is_new {
            // SAFETY: We just added the event.
            let kind = self.targeted_events.get(id).unwrap_unchecked().kind();

            match kind {
                EventKind::Normal => {}
                EventKind::Insert(InsertedComponentsInfo {
                    component_indices, ..
                }) => {
                    for component_idx in component_indices {
                        if let Some(info) = self.components.get_by_index_mut(component_idx) {
                            info.insert_events.insert(id);
                        }
                    }
                }
                EventKind::Remove(RemovedComponentsInfo { component_indices }) => {
                    for component_idx in component_indices {
                        if let Some(info) = self.components.get_by_index_mut(component_idx) {
                            info.remove_events.insert(id);
                        }
                    }
                }
                EventKind::Spawn(SpawnInfo {
                    inserted_components:
                        InsertedComponentsInfo {
                            component_indices, ..
                        },
                    ..
                }) => {
                    for component_idx in component_indices {
                        if let Some(info) = self.components.get_by_index_mut(component_idx) {
                            info.spawn_events.insert(id);
                        }
                    }
                }
                EventKind::Despawn => {}
            }

            self.send(AddTargetedEvent(id))
        }

        id
    }

    /// Removes a global event from the world and returns its
    /// [`GlobalEventInfo`]. If the event ID is invalid, then `None` is
    /// returned and the function has no effect.
    ///
    /// Removing an event has the following additional effects in the order
    /// listed:
    /// 1. The [`RemoveTargetedEvent`] event is sent.
    /// 2. All handlers that send or receive the event are removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(GlobalEvent)]
    /// struct MyEvent;
    ///
    /// let mut world = World::new();
    ///
    /// let id = world.add_global_event::<MyEvent>();
    /// world.remove_global_event(id);
    ///
    /// assert!(!world.global_events().contains(id));
    /// ```
    pub fn remove_global_event(&mut self, event: GlobalEventId) -> Option<GlobalEventInfo> {
        assert!(self.event_queue.is_empty());

        if !self.global_events.contains(event) {
            return None;
        }

        // Send event before removing anything.
        self.send(RemoveGlobalEvent(event));

        // Remove all handlers that send or receive this event.
        let mut to_remove = vec![];

        for handler in self.handlers.iter() {
            if handler.received_event() == event.into()
                || handler.sent_global_events_bitset().contains(event.index())
            {
                to_remove.push(handler.id());
            }
        }

        for id in to_remove {
            self.remove_handler(id);
        }

        Some(self.global_events.remove(event).unwrap())
    }

    /// Removes a targeted event from the world and returns its
    /// [`TargetedEventInfo`]. If the event ID is invalid, then `None` is
    /// returned and the function has no effect.
    ///
    /// Removing an event has the following additional effects in the order
    /// listed:
    /// 1. The [`RemoveTargetedEvent`] event is sent.
    /// 2. All handlers that send or receive the event are removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(TargetedEvent)]
    /// struct MyEvent;
    ///
    /// let mut world = World::new();
    ///
    /// let id = world.add_targeted_event::<MyEvent>();
    /// world.remove_targeted_event(id);
    ///
    /// assert!(!world.targeted_events().contains(id));
    /// ```
    pub fn remove_targeted_event(&mut self, event: TargetedEventId) -> Option<TargetedEventInfo> {
        assert!(self.event_queue.is_empty());

        if !self.targeted_events.contains(event) {
            return None;
        }

        // Send event before doing anything else.
        self.send(RemoveTargetedEvent(event));

        // Remove all handlers that send or receive this event.
        let mut to_remove = vec![];

        for handler in self.handlers.iter() {
            if handler.received_event() == event.into()
                || handler
                    .sent_targeted_events_bitset()
                    .contains(event.index())
            {
                to_remove.push(handler.id());
            }
        }

        for id in to_remove {
            self.remove_handler(id);
        }

        let info = self.targeted_events.remove(event).unwrap();

        match info.kind() {
            EventKind::Normal => {}
            EventKind::Insert(InsertedComponentsInfo {
                component_indices, ..
            }) => {
                for component_idx in component_indices {
                    if let Some(info) = self.components.get_by_index_mut(component_idx) {
                        info.insert_events.remove(&event);
                    }
                }
            }
            EventKind::Remove(RemovedComponentsInfo { component_indices }) => {
                for component_idx in component_indices {
                    if let Some(info) = self.components.get_by_index_mut(component_idx) {
                        info.remove_events.remove(&event);
                    }
                }
            }
            EventKind::Spawn(SpawnInfo {
                inserted_components:
                    InsertedComponentsInfo {
                        component_indices, ..
                    },
                ..
            }) => {
                for component_idx in component_indices {
                    if let Some(info) = self.components.get_by_index_mut(component_idx) {
                        info.spawn_events.remove(&event);
                    }
                }
            }
            EventKind::Despawn => {}
        }

        Some(info)
    }

    /// Returns the [`Entities`] for this world.
    pub fn entities(&self) -> &Entities {
        &self.entities
    }

    /// Returns the [`Components`] for this world.  
    pub fn components(&self) -> &Components {
        &self.components
    }

    /// Returns the [`Handlers`] for this world.
    pub fn handlers(&self) -> &Handlers {
        &self.handlers
    }

    /// Returns the [`Archetypes`] for this world.
    pub fn archetypes(&self) -> &Archetypes {
        &self.archetypes
    }

    /// Returns the [`GlobalEvents`] for this world.
    pub fn global_events(&self) -> &GlobalEvents {
        &self.global_events
    }

    /// Returns the [`TargetedEvents`] for this world.
    pub fn targeted_events(&self) -> &TargetedEvents {
        &self.targeted_events
    }

    /// Broadcasts a global event to all handlers in this world and returns the
    /// event if it was not handled.
    ///
    /// This function sends the given global event to all handlers that are
    /// registered to listen for this event. If the event is not handled by
    /// any handler, it is returned.
    ///
    /// Any events sent by handlers will also be broadcast. This process
    /// continues recursively until all events have finished broadcasting.
    ///
    /// # Examples
    ///
    /// ```
    /// use evenio::prelude::*;
    ///
    /// #[derive(GlobalEvent, Debug, PartialEq)]
    /// struct MyEvent(i32);
    ///
    /// fn my_handler(r: Receiver<MyEvent>) {
    ///     println!("got event: {}", r.event.0);
    /// }
    ///
    /// let mut world = World::new();
    ///
    /// world.add_handler(my_handler);
    ///
    /// let response = world.send_with_response(MyEvent(123));
    ///
    /// assert_eq!(response, Some(MyEvent(123)));
    /// ```
    ///
    /// # Safety
    ///
    /// You CANNOT use this function in handlers!
    ///
    /// # Returns
    ///
    /// - `Some(E)` - final, modified event by all handlers
    /// - `None` if the event was consumed by at least one handler.
    pub fn send_with_response<E: GlobalEvent>(&mut self, event: E) -> Option<E> {
        let idx = self.add_global_event::<E>().index();

        let event_ptr = self.send_inner(EventQueueItem {
            meta: EventMeta::Global { idx },
            event: NonNull::from(self.bump.alloc(event)).cast(),
        })?;

        self.flush_event_queue();

        Some(unsafe { core::ptr::read(event_ptr.as_ptr() as *const E) })
    }

    /// Broadcast a targeted event to all handlers in this world.
    ///
    /// Any events sent by handlers will also broadcast. This process continues
    /// recursively until all events have finished broadcasting.
    ///
    /// See also [`World::send`] to send a [`GlobalEvent`].
    ///
    /// # Safety
    ///
    /// You CANNOT use this function in handlers!
    ///
    /// # Returns
    ///
    /// - `Some(E)` - final, modified event by all handlers
    /// - `None` if the event was consumed by at least one handler.
    pub fn send_to_with_response<E: TargetedEvent>(&mut self, target: EntityId, event: E) -> Option<E> {
        let idx = self.add_targeted_event::<E>().index();

        let event_ptr = self.send_inner(EventQueueItem {
            meta: EventMeta::Targeted { target, idx },
            event: NonNull::from(self.bump.alloc(event)).cast(),
        })?;

        self.flush_event_queue();

        Some(unsafe { core::ptr::read(event_ptr.as_ptr() as *const E) })
    }


    pub(crate) fn send_inner(&mut self, item: EventQueueItem) -> Option<NonNull<u8>> {
        struct EventDropper<'a> {
            event: NonNull<u8>,
            drop: DropFn,
            ownership_flag: bool,
            world: &'a mut World,
        }

        impl<'a> EventDropper<'a> {
            fn new(event: NonNull<u8>, drop: DropFn, world: &'a mut World) -> Self {
                Self {
                    event,
                    drop,
                    ownership_flag: false,
                    world,
                }
            }

            /// Extracts the event pointer and drop fn without running
            /// the destructor.
            #[inline]
            fn unpack(self) -> (NonNull<u8>, DropFn) {
                let event = self.event;
                let drop = self.drop;
                mem::forget(self);

                (event, drop)
            }

            /// Drops the held event.
            #[inline]
            unsafe fn drop_event(self) {
                if let (event, Some(drop)) = self.unpack() {
                    drop(event);
                }
            }
        }

        impl Drop for EventDropper<'_> {
            #[cold]
            fn drop(&mut self) {
                // In case `Handler::run` unwinds, we need to drop the event we're holding on
                // the stack as well as the events still the in the event queue.

                #[cfg(feature = "std")]
                debug_assert!(std::thread::panicking());

                // Drop the held event.
                if !self.ownership_flag {
                    if let Some(drop) = self.drop {
                        unsafe { drop(self.event) };
                    }
                }

                // Drop all events remaining in the event queue.
                // This must be done here instead of the World's destructor because events
                // could contain borrowed data.
                for item in &self.world.event_queue {
                    let drop = match item.meta {
                        EventMeta::Global { idx } => unsafe {
                            self.world
                                .global_events
                                .get_by_index(idx)
                                .unwrap_unchecked()
                                .drop()
                        },
                        EventMeta::Targeted { idx, .. } => unsafe {
                            self.world
                                .targeted_events
                                .get_by_index(idx)
                                .unwrap_unchecked()
                                .drop()
                        },
                    };

                    if let Some(drop) = drop {
                        unsafe { drop(item.event) };
                    }
                }

                self.world.event_queue.clear();
            }
        }

        let (mut ctx, event_kind, handlers, target_location) = match item.meta {
            EventMeta::Global { idx } => {
                let info = unsafe { self.global_events.get_by_index(idx).unwrap_unchecked() };
                let kind: *const _ = info.kind();
                let handlers: *const [_] =
                    unsafe { self.handlers.get_global_list(idx).unwrap_unchecked() }.slice();
                let ctx = EventDropper::new(item.event, info.drop(), self);

                let location = EntityLocation::NULL;

                (ctx, kind, handlers, location)
            }
            EventMeta::Targeted { idx, target } => {
                let info = unsafe { self.targeted_events.get_by_index(idx).unwrap_unchecked() };
                let kind: *const _ = info.kind();
                let ctx = EventDropper::new(item.event, info.drop(), self);

                let Some(location) = ctx.world.entities.get(target) else {
                    // Entity doesn't exist. Skip the event.
                    return Some(ctx.unpack().0);
                };

                let arch = unsafe {
                    ctx.world
                        .archetypes
                        .get(location.archetype)
                        .unwrap_unchecked()
                };

                static EMPTY: HandlerList = HandlerList::new();

                let handlers: *const [_] = arch.handler_list_for(idx).unwrap_or(&EMPTY).slice();

                (ctx, kind, handlers, location)
            }
        };

        for mut info_ptr in unsafe { (*handlers).iter().copied() } {
            let info = unsafe { info_ptr.as_info_mut() };

            let handler: *mut dyn Handler = info.handler_mut();

            let event_ptr = EventPtr::new(ctx.event, NonNull::from(&mut ctx.ownership_flag));

            let world_cell = ctx.world.unsafe_cell_mut();

            unsafe { (*handler).run(info, event_ptr, target_location, world_cell) };

            // Did the handler take ownership of the event?
            if ctx.ownership_flag {
                // Don't drop event since we don't own it anymore.
                ctx.unpack();
                return None;
            }
        }

        match unsafe { &*event_kind } {
            EventKind::Normal => {
                // Ordinary event
                return Some(ctx.unpack().0);
            }
            EventKind::Insert(InsertedComponentsInfo {
                                  component_indices,
                                  permutation,
                                  get_components,
                              }) => {
                debug_assert_ne!(target_location, EntityLocation::NULL);

                let src_arch = ctx
                    .world
                    .archetypes()
                    .get(target_location.archetype)
                    .unwrap();
                let dst_component_indices =
                    src_arch.component_indices().clone() | component_indices;
                let dst = unsafe {
                    ctx.world.archetypes.create_archetype(
                        dst_component_indices,
                        &mut ctx.world.components,
                        &mut ctx.world.handlers,
                    )
                };

                // The Insert event is repr(transparent), and its only field
                // is the set of inserted components.
                let components_ptr: *const u8 = ctx.event.as_ptr();

                // Resize the component pointer buffer.
                let buffer = &mut ctx.world.component_pointer_buffer;
                buffer.reserve(permutation.len().saturating_sub(buffer.len()));
                // SAFETY: We just ensured that the buffer's capacity is
                // sufficient. Elements being uninitialized is not an issue,
                // because they are MaybeUninit values anyway.
                unsafe {
                    buffer.set_len(permutation.len());
                }

                // Create a `ComponentPointerConsumer` with the event's
                // permutation to sort the collected component pointers.
                let mut consumer = ComponentPointerConsumer::new(permutation, buffer);

                // Collect the component pointers.
                unsafe { get_components(components_ptr, &mut consumer) };

                // Unpack the consumer to get the pointers.
                let component_pointers = unsafe { consumer.get_pointers_unchecked() };

                // Finally, move the entity to the destination archetype,
                // passing the component pointers.
                unsafe {
                    ctx.world.archetypes.move_entity(
                        target_location,
                        dst,
                        component_indices,
                        component_pointers,
                        &mut ctx.world.entities,
                    )
                };

                // Inserted components are owned by the archetype now. We
                // wait to unpack in case one of the above functions panics.
                let _ = ctx.unpack();
            }
            EventKind::Remove(RemovedComponentsInfo { component_indices }) => {
                // `Remove` doesn't need drop.
                let _ = ctx.unpack();

                let src_arch = self.archetypes().get(target_location.archetype).unwrap();
                let mut dst_component_indices = src_arch.component_indices().clone();
                dst_component_indices.remove_all(component_indices);
                let dst = unsafe {
                    self.archetypes.create_archetype(
                        dst_component_indices,
                        &mut self.components,
                        &mut self.handlers,
                    )
                };

                unsafe {
                    self.archetypes.move_entity(
                        target_location,
                        dst,
                        &BitSet::<ComponentIdx>::new(),
                        &[],
                        &mut self.entities,
                    )
                };
            }
            EventKind::Spawn(SpawnInfo {
                                 components_field_offset,
                                 inserted_components:
                                 InsertedComponentsInfo {
                                     component_indices,
                                     permutation,
                                     get_components,
                                 },
                             }) => {
                let arch = ctx
                    .world
                    .archetypes
                    .get_by_components(component_indices)
                    .unwrap_or_else(|| unsafe {
                        ctx.world.archetypes.create_archetype(
                            component_indices.clone(),
                            &mut ctx.world.components,
                            &mut ctx.world.handlers,
                        )
                    });

                // The offset of the `components` field of the event is
                // stored in the event kind.
                let components_ptr: *const u8 =
                    unsafe { ctx.event.as_ptr().byte_add(*components_field_offset) };

                // Resize the component pointer buffer.
                let buffer = &mut ctx.world.component_pointer_buffer;
                buffer.reserve(permutation.len().saturating_sub(buffer.len()));
                // SAFETY: We just ensured that the buffer's capacity is
                // sufficient. Elements being uninitialized is not an issue,
                // because they are MaybeUninit values anyway.
                unsafe {
                    buffer.set_len(permutation.len());
                }

                // Create a `ComponentPointerConsumer` with the event's
                // permutation to sort the collected component pointers.
                let mut consumer = ComponentPointerConsumer::new(permutation, buffer);

                // Collect the component pointers.
                unsafe { get_components(components_ptr, &mut consumer) };

                // Unpack the consumer to get the pointers.
                let component_pointers = unsafe { consumer.get_pointers_unchecked() };

                // Spawn the next entity from the reserved entity queue.
                ctx.world
                    .reserved_entities
                    .spawn_one(&mut ctx.world.entities, |id| unsafe {
                        ctx.world.archetypes.spawn(id, arch, component_pointers)
                    });

                // Inserted components are owned by the archetype now. We
                // wait to unpack in case one of the above functions panics.
                let _ = ctx.unpack();
            }
            EventKind::Despawn => {
                // `Despawn` doesn't need drop.
                let _ = ctx.unpack();

                unsafe {
                    self.archetypes
                        .remove_entity(target_location, &mut self.entities)
                };

                // Reset next key iter.
                self.reserved_entities.refresh(&self.entities);
            }
        }

        self.bump.reset();
        None
    }

    /// Send all queued events to handlers. The event queue will be empty after
    /// this call.
    pub fn flush_event_queue(&mut self) {
        'next_event: while let Some(item) = self.event_queue.pop() {
            struct EventDropper<'a> {
                event: NonNull<u8>,
                drop: DropFn,
                ownership_flag: bool,
                world: &'a mut World,
            }

            impl<'a> EventDropper<'a> {
                fn new(event: NonNull<u8>, drop: DropFn, world: &'a mut World) -> Self {
                    Self {
                        event,
                        drop,
                        ownership_flag: false,
                        world,
                    }
                }

                /// Extracts the event pointer and drop fn without running
                /// the destructor.
                #[inline]
                fn unpack(self) -> (NonNull<u8>, DropFn) {
                    let event = self.event;
                    let drop = self.drop;
                    mem::forget(self);

                    (event, drop)
                }

                /// Drops the held event.
                #[inline]
                unsafe fn drop_event(self) {
                    if let (event, Some(drop)) = self.unpack() {
                        drop(event);
                    }
                }
            }

            impl Drop for EventDropper<'_> {
                #[cold]
                fn drop(&mut self) {
                    // In case `Handler::run` unwinds, we need to drop the event we're holding on
                    // the stack as well as the events still the in the event queue.

                    #[cfg(feature = "std")]
                    debug_assert!(std::thread::panicking());

                    // Drop the held event.
                    if !self.ownership_flag {
                        if let Some(drop) = self.drop {
                            unsafe { drop(self.event) };
                        }
                    }

                    // Drop all events remaining in the event queue.
                    // This must be done here instead of the World's destructor because events
                    // could contain borrowed data.
                    for item in &self.world.event_queue {
                        let drop = match item.meta {
                            EventMeta::Global { idx } => unsafe {
                                self.world
                                    .global_events
                                    .get_by_index(idx)
                                    .unwrap_unchecked()
                                    .drop()
                            },
                            EventMeta::Targeted { idx, .. } => unsafe {
                                self.world
                                    .targeted_events
                                    .get_by_index(idx)
                                    .unwrap_unchecked()
                                    .drop()
                            },
                        };

                        if let Some(drop) = drop {
                            unsafe { drop(item.event) };
                        }
                    }

                    self.world.event_queue.clear();
                }
            }

            let (mut ctx, event_kind, handlers, target_location) = match item.meta {
                EventMeta::Global { idx } => {
                    let info = unsafe { self.global_events.get_by_index(idx).unwrap_unchecked() };
                    let kind: *const _ = info.kind();
                    let handlers: *const [_] =
                        unsafe { self.handlers.get_global_list(idx).unwrap_unchecked() }.slice();
                    let ctx = EventDropper::new(item.event, info.drop(), self);

                    let location = EntityLocation::NULL;

                    (ctx, kind, handlers, location)
                }
                EventMeta::Targeted { idx, target } => {
                    let info = unsafe { self.targeted_events.get_by_index(idx).unwrap_unchecked() };
                    let kind: *const _ = info.kind();
                    let ctx = EventDropper::new(item.event, info.drop(), self);

                    let Some(location) = ctx.world.entities.get(target) else {
                        // Entity doesn't exist. Skip the event.
                        unsafe { ctx.drop_event() };
                        continue;
                    };

                    let arch = unsafe {
                        ctx.world
                            .archetypes
                            .get(location.archetype)
                            .unwrap_unchecked()
                    };

                    static EMPTY: HandlerList = HandlerList::new();

                    let handlers: *const [_] = arch.handler_list_for(idx).unwrap_or(&EMPTY).slice();

                    (ctx, kind, handlers, location)
                }
            };

            let events_before = ctx.world.event_queue.len();

            for mut info_ptr in unsafe { (*handlers).iter().copied() } {
                let info = unsafe { info_ptr.as_info_mut() };

                let handler: *mut dyn Handler = info.handler_mut();

                let event_ptr = EventPtr::new(ctx.event, NonNull::from(&mut ctx.ownership_flag));

                let world_cell = ctx.world.unsafe_cell_mut();

                unsafe { (*handler).run(info, event_ptr, target_location, world_cell) };

                // Did the handler take ownership of the event?
                if ctx.ownership_flag {
                    // Don't drop event since we don't own it anymore.
                    ctx.unpack();

                    // Reverse pushed events so they're handled in FIFO order.
                    unsafe {
                        self.event_queue
                            .get_unchecked_mut(events_before..)
                            .reverse()
                    };

                    continue 'next_event;
                }
            }

            // Reverse pushed events so they're handled in FIFO order.
            unsafe {
                ctx.world
                    .event_queue
                    .get_unchecked_mut(events_before..)
                    .reverse()
            };

            match unsafe { &*event_kind } {
                EventKind::Normal => {
                    // Ordinary event. Run drop fn.
                    unsafe { ctx.drop_event() };
                }
                EventKind::Insert(InsertedComponentsInfo {
                    component_indices,
                    permutation,
                    get_components,
                }) => {
                    debug_assert_ne!(target_location, EntityLocation::NULL);

                    let src_arch = ctx
                        .world
                        .archetypes()
                        .get(target_location.archetype)
                        .unwrap();
                    let dst_component_indices =
                        src_arch.component_indices().clone() | component_indices;
                    let dst = unsafe {
                        ctx.world.archetypes.create_archetype(
                            dst_component_indices,
                            &mut ctx.world.components,
                            &mut ctx.world.handlers,
                        )
                    };

                    // The Insert event is repr(transparent), and its only field
                    // is the set of inserted components.
                    let components_ptr: *const u8 = ctx.event.as_ptr();

                    // Resize the component pointer buffer.
                    let buffer = &mut ctx.world.component_pointer_buffer;
                    buffer.reserve(permutation.len().saturating_sub(buffer.len()));
                    // SAFETY: We just ensured that the buffer's capacity is
                    // sufficient. Elements being uninitialized is not an issue,
                    // because they are MaybeUninit values anyway.
                    unsafe {
                        buffer.set_len(permutation.len());
                    }

                    // Create a `ComponentPointerConsumer` with the event's
                    // permutation to sort the collected component pointers.
                    let mut consumer = ComponentPointerConsumer::new(permutation, buffer);

                    // Collect the component pointers.
                    unsafe { get_components(components_ptr, &mut consumer) };

                    // Unpack the consumer to get the pointers.
                    let component_pointers = unsafe { consumer.get_pointers_unchecked() };

                    // Finally, move the entity to the destination archetype,
                    // passing the component pointers.
                    unsafe {
                        ctx.world.archetypes.move_entity(
                            target_location,
                            dst,
                            component_indices,
                            component_pointers,
                            &mut ctx.world.entities,
                        )
                    };

                    // Inserted components are owned by the archetype now. We
                    // wait to unpack in case one of the above functions panics.
                    let _ = ctx.unpack();
                }
                EventKind::Remove(RemovedComponentsInfo { component_indices }) => {
                    // `Remove` doesn't need drop.
                    let _ = ctx.unpack();

                    let src_arch = self.archetypes().get(target_location.archetype).unwrap();
                    let mut dst_component_indices = src_arch.component_indices().clone();
                    dst_component_indices.remove_all(component_indices);
                    let dst = unsafe {
                        self.archetypes.create_archetype(
                            dst_component_indices,
                            &mut self.components,
                            &mut self.handlers,
                        )
                    };

                    unsafe {
                        self.archetypes.move_entity(
                            target_location,
                            dst,
                            &BitSet::<ComponentIdx>::new(),
                            &[],
                            &mut self.entities,
                        )
                    };
                }
                EventKind::Spawn(SpawnInfo {
                    components_field_offset,
                    inserted_components:
                        InsertedComponentsInfo {
                            component_indices,
                            permutation,
                            get_components,
                        },
                }) => {
                    let arch = ctx
                        .world
                        .archetypes
                        .get_by_components(component_indices)
                        .unwrap_or_else(|| unsafe {
                            ctx.world.archetypes.create_archetype(
                                component_indices.clone(),
                                &mut ctx.world.components,
                                &mut ctx.world.handlers,
                            )
                        });

                    // The offset of the `components` field of the event is
                    // stored in the event kind.
                    let components_ptr: *const u8 =
                        unsafe { ctx.event.as_ptr().byte_add(*components_field_offset) };

                    // Resize the component pointer buffer.
                    let buffer = &mut ctx.world.component_pointer_buffer;
                    buffer.reserve(permutation.len().saturating_sub(buffer.len()));
                    // SAFETY: We just ensured that the buffer's capacity is
                    // sufficient. Elements being uninitialized is not an issue,
                    // because they are MaybeUninit values anyway.
                    unsafe {
                        buffer.set_len(permutation.len());
                    }

                    // Create a `ComponentPointerConsumer` with the event's
                    // permutation to sort the collected component pointers.
                    let mut consumer = ComponentPointerConsumer::new(permutation, buffer);

                    // Collect the component pointers.
                    unsafe { get_components(components_ptr, &mut consumer) };

                    // Unpack the consumer to get the pointers.
                    let component_pointers = unsafe { consumer.get_pointers_unchecked() };

                    // Spawn the next entity from the reserved entity queue.
                    ctx.world
                        .reserved_entities
                        .spawn_one(&mut ctx.world.entities, |id| unsafe {
                            ctx.world.archetypes.spawn(id, arch, component_pointers)
                        });

                    // Inserted components are owned by the archetype now. We
                    // wait to unpack in case one of the above functions panics.
                    let _ = ctx.unpack();
                }
                EventKind::Despawn => {
                    // `Despawn` doesn't need drop.
                    let _ = ctx.unpack();

                    unsafe {
                        self.archetypes
                            .remove_entity(target_location, &mut self.entities)
                    };

                    // Reset next key iter.
                    self.reserved_entities.refresh(&self.entities);
                }
            }
        }

        self.bump.reset();
        debug_assert!(self.event_queue.is_empty());
    }

    /// Returns a new [`UnsafeWorldCell`] with permission to _read_ all data in
    /// this world.
    pub fn unsafe_cell(&self) -> UnsafeWorldCell {
        UnsafeWorldCell {
            world: NonNull::from(self),
            _marker: PhantomData,
        }
    }

    /// Returns a new [`UnsafeWorldCell`] with permission to _read and write_
    /// all data in this world.
    pub fn unsafe_cell_mut(&mut self) -> UnsafeWorldCell {
        UnsafeWorldCell {
            world: NonNull::from(self),
            _marker: PhantomData,
        }
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference to a [`World`] where all methods take `self` and aliasing rules
/// are not checked. It is the caller's responsibility to ensure that Rust's
/// aliasing rules are not violated.
#[derive(Clone, Copy, Debug)]
pub struct UnsafeWorldCell<'a> {
    world: NonNull<World>,
    _marker: PhantomData<&'a World>,
}

impl<'a> UnsafeWorldCell<'a> {
    /// Allocate data in the world's bump allocator.
    ///
    /// This operation is not thread safe.
    ///
    /// # Safety
    ///
    /// - Must be called from within a handler.
    #[inline]
    pub unsafe fn alloc_layout(self, layout: Layout) -> NonNull<u8> {
        let bump = unsafe { &(*self.world.as_ptr()).bump };
        bump.alloc_layout(layout)
    }

    /// Add a global event to the event queue. Ownership of the event is
    /// transferred.
    ///
    /// # Safety
    ///
    /// - Must be called from within a handler.
    /// - Event must outlive call to top level [`World::send`] or
    ///   [`World::send_to`].
    /// - Event index must be correct for the given event.
    #[inline]
    pub unsafe fn queue_global(self, event: NonNull<u8>, idx: GlobalEventIdx) {
        let event_queue = &mut (*self.world.as_ptr()).event_queue;

        event_queue.push(EventQueueItem {
            meta: EventMeta::Global { idx },
            event,
        });
    }

    /// Add a targeted event to the event queue. Ownership of the event is
    /// transferred.
    ///
    /// # Safety
    ///
    /// - Must be called from within a handler.
    /// - Must have permission to access the event queue
    #[inline]
    pub unsafe fn queue_targeted(
        self,
        target: EntityId,
        event: NonNull<u8>,
        idx: TargetedEventIdx,
    ) {
        let event_queue = &mut (*self.world.as_ptr()).event_queue;

        event_queue.push(EventQueueItem {
            meta: EventMeta::Targeted { idx, target },
            event,
        });
    }

    /// # Safety
    ///
    /// - Must be called from within a handler.
    pub unsafe fn queue_spawn(self) -> EntityId {
        let entity_id = (*self.world.as_ptr())
            .reserved_entities
            .reserve(self.entities());

        entity_id
    }

    /// Returns the [`Entities`] for this world.
    pub fn entities(self) -> &'a Entities {
        unsafe { &(*self.world.as_ptr()).entities }
    }

    /// Returns the [`Components`] for this world.
    pub fn components(self) -> &'a Components {
        unsafe { &(*self.world.as_ptr()).components }
    }

    /// Returns the [`Handlers`] for this world.
    pub fn handlers(self) -> &'a Handlers {
        unsafe { &(*self.world.as_ptr()).handlers }
    }

    /// Returns the [`Archetypes`] for this world.
    pub fn archetypes(self) -> &'a Archetypes {
        unsafe { &(*self.world.as_ptr()).archetypes }
    }

    /// Returns the [`GlobalEvents`] for this world.
    pub fn global_events(self) -> &'a GlobalEvents {
        unsafe { &(*self.world.as_ptr()).global_events }
    }

    /// Returns the [`TargetedEvents`] for this world.
    pub fn targeted_events(self) -> &'a TargetedEvents {
        unsafe { &(*self.world.as_ptr()).targeted_events }
    }

    /// Returns an immutable reference to the underlying world.
    ///
    /// # Safety
    ///
    /// Must have permission to access the entire world immutably.
    pub fn world(self) -> &'a World {
        unsafe { &*self.world.as_ptr() }
    }

    /// Returns a mutable reference to the underlying world.
    ///
    /// # Safety
    ///
    /// Must have permission the access the entire world mutably.
    pub unsafe fn world_mut(self) -> &'a mut World {
        &mut *self.world.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use core::panic::AssertUnwindSafe;
    use std::panic;

    use crate::prelude::*;

    #[test]
    #[should_panic]
    fn self_aliasing_query() {
        #[derive(Component)]
        struct C;

        let mut world = World::new();
        let e = world.spawn(());
        world.insert(e, C);
        world.get_mut::<(&mut C, &mut C)>(e);
    }

    #[test]
    fn world_drops_events() {
        #[derive(GlobalEvent)]
        struct A(Rc<()>);

        #[derive(GlobalEvent)]
        struct B(Rc<()>);

        #[derive(GlobalEvent)]
        struct C(Rc<()>);

        let mut world = World::new();

        world.add_handler(|r: Receiver<A>, s: Sender<B>| {
            s.send(B(r.event.0.clone()));
            s.send(B(r.event.0.clone()));
        });

        world.add_handler(|r: Receiver<B>, s: Sender<C>| {
            s.send(C(r.event.0.clone()));
            s.send(C(r.event.0.clone()));
        });

        world.add_handler(|r: Receiver<C>| println!("got C {:?}", Rc::as_ptr(&r.event.0)));

        let rc = Rc::new(());

        world.send(A(rc.clone()));

        drop(world);

        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn world_drops_events_on_panic() {
        #[derive(GlobalEvent)]
        struct A(Rc<()>);

        impl Drop for A {
            fn drop(&mut self) {
                eprintln!("calling A destructor");
            }
        }

        #[derive(GlobalEvent)]
        struct B(Rc<()>);

        #[allow(dead_code)]
        #[derive(GlobalEvent)]
        struct C(Rc<()>);

        let mut world = World::new();

        world.add_handler(|r: Receiver<A>, s: Sender<B>| {
            s.send(B(r.event.0.clone()));
            s.send(B(r.event.0.clone()));
        });

        world.add_handler(|r: Receiver<B>, s: Sender<C>| {
            s.send(C(r.event.0.clone()));
            s.send(C(r.event.0.clone()));
        });

        world.add_handler(|_: Receiver<C>| panic!("oops!"));

        let arc = Rc::new(());
        let arc_cloned = arc.clone();

        let mut world = AssertUnwindSafe(world);

        let res = panic::catch_unwind(move || world.send(A(arc_cloned)));

        assert_eq!(*res.unwrap_err().downcast::<&str>().unwrap(), "oops!");

        assert_eq!(Rc::strong_count(&arc), 1);
    }

    #[test]
    fn bump_allocator_is_reset() {
        let mut world = World::new();

        #[derive(GlobalEvent)]
        struct Data(#[allow(dead_code)] u64);

        world.send(Data(123));

        let ptr1 = world.bump.alloc(1_u8) as *const u8;
        world.bump.reset();
        let ptr2 = world.bump.alloc(1_u8) as *const u8;

        assert_eq!(ptr1, ptr2);
    }
}
