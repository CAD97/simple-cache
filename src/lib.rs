use std::{
    borrow::Borrow,
    collections::{hash_map::RandomState, HashMap},
    fmt,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    ops::Deref,
    ptr::NonNull,
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
};

#[derive(Debug)]
pub struct HashCache<K, V, S = RandomState, F = ()> {
    /// SAFETY: produced PinBox value reference lifetimes are bound by &self.
    arena: RwLock<HashMap<K, PinBox<V>, S>>,
    provider: F,
}

#[non_exhaustive]
#[derive(Debug, Default)]
pub struct HashCacheConfig<S, F> {
    pub capacity: usize,
    pub hasher: S,
    pub provider: F,
}

impl<K, V, S, F> Default for HashCache<K, V, S, F>
where
    S: Default,
    F: Default,
{
    fn default() -> Self {
        Self::with_config(HashCacheConfig::default())
    }
}

impl<K, V> HashCache<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_config(HashCacheConfig {
            capacity,
            hasher: RandomState::default(),
            provider: (),
        })
    }
}

impl<K, V, S> HashCache<K, V, S> {
    pub fn with_hasher(hasher: S) -> Self {
        Self::with_config(HashCacheConfig {
            capacity: 0,
            hasher,
            provider: (),
        })
    }
}

impl<K, V, S, F> HashCache<K, V, S, F> {
    pub fn with_provider(provider: F) -> Self
    where
        S: Default,
    {
        Self::with_config(HashCacheConfig {
            capacity: 0,
            hasher: S::default(),
            provider,
        })
    }

    pub fn with_config(config: HashCacheConfig<S, F>) -> Self {
        let HashCacheConfig {
            capacity,
            hasher,
            provider,
        } = config;
        Self {
            arena: RwLock::new(HashMap::with_capacity_and_hasher(capacity, hasher)),
            provider,
        }
    }

    fn arena(&self) -> RwLockReadGuard<HashMap<K, PinBox<V>, S>> {
        // just ignore poisoning
        self.arena.read().unwrap_or_else(|e| e.into_inner())
    }

    fn arena_mut(&self) -> RwLockWriteGuard<HashMap<K, PinBox<V>, S>> {
        // just ignore poisoning
        self.arena.write().unwrap_or_else(|e| e.into_inner())
    }

    pub fn clear(&mut self) {
        // SAFETY: &mut self access invalidates all extant fn get(&self) -> &V.
        // just ignore poisoning
        self.arena
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

impl<K, V, S, F> HashCache<K, V, S, F>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let arena = &self.arena();
        let value = arena.get(key)?;
        // SAFETY: The returned value lifetime is derived from &self.
        Some(unsafe { value.as_ref() })
    }

    pub fn get_or_insert<Q>(&self, key: &Q) -> &V
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
        F: Fn(&K) -> V,
    {
        if let Some(v) = self.get(key) {
            return v;
        }

        let key = key.to_owned();
        let arena = &mut self.arena_mut();
        let value = arena.entry(key).or_insert_with_key(|k| {
            let v = (self.provider)(k);
            PinBox::new(Box::new(v))
        });
        // SAFETY: The returned value lifetime is derived from &self.
        unsafe { value.as_ref() }
    }

    pub fn get_or_insert_with<Q, G>(&self, key: &Q, f: G) -> &V
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
        G: FnOnce(&K) -> V,
    {
        if let Some(v) = self.get(key) {
            return v;
        }

        let key = key.to_owned();
        let arena = &mut self.arena_mut();
        let value = arena.entry(key).or_insert_with_key(|k| {
            let v = f(k);
            PinBox::new(Box::new(v))
        });
        // SAFETY: The returned value lifetime is derived from &self.
        unsafe { value.as_ref() }
    }
}

/// A wrapper around box that does not provide &mut access to the pointee and
/// uses raw-pointer borrowing rules to avoid invalidating extant references.
///
/// The resolved reference is guaranteed valid until the PinBox is dropped.
struct PinBox<T: ?Sized> {
    ptr: NonNull<T>,
    _marker: PhantomData<Box<T>>,
}

impl<T: ?Sized> PinBox<T> {
    fn new(x: Box<T>) -> Self {
        Self {
            ptr: NonNull::new(Box::into_raw(x)).unwrap(),
            _marker: PhantomData,
        }
    }

    unsafe fn as_ref<'a>(&self) -> &'a T {
        self.ptr.as_ref()
    }
}

impl<T: ?Sized> Drop for PinBox<T> {
    fn drop(&mut self) {
        unsafe { Box::from_raw(self.ptr.as_ptr()) };
    }
}

impl<T: ?Sized> Deref for PinBox<T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { self.as_ref() }
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for PinBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

unsafe impl<T: ?Sized> Send for PinBox<T> where Box<T>: Send {}
unsafe impl<T: ?Sized> Sync for PinBox<T> where Box<T>: Sync {}
