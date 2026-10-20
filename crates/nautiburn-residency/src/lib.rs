use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub model_id: String,
    pub checkpoint_sha: String,
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub fn new(model_id: &str, checkpoint_sha: &str, layer: u32, expert: u32) -> Self {
        Self {
            model_id: model_id.to_string(),
            checkpoint_sha: checkpoint_sha.to_string(),
            layer,
            expert,
        }
    }

    fn lru_token(&self, last_used: u64) -> LruToken {
        (
            last_used,
            self.model_id.clone(),
            self.checkpoint_sha.clone(),
            self.layer,
            self.expert,
        )
    }
}

type LruToken = (u64, String, String, u32, u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tier {
    Hot { device: u32 },
    Warm,
    Cold,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    Resident,
    Load {
        from: Tier,
        to: Tier,
        evict: Vec<ExpertKey>,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    OverBudget,
    UnknownDevice,
    NotPrefetched,
}

#[derive(Debug)]
struct Entry {
    bytes: u64,
    tier: Tier,
    last_used: u64,
}

#[derive(Debug)]
struct DeviceState {
    budget: u64,
    used: u64,
    lru: BTreeSet<LruToken>,
}

#[derive(Debug)]
struct PendingLoad {
    device: u32,
    #[allow(dead_code)]
    deadline_ms: u64,
}

#[derive(Debug)]
pub struct Inventory {
    host_budget: u64,
    host_used: u64,
    clock: u64,
    devices: HashMap<u32, DeviceState>,
    entries: HashMap<ExpertKey, Entry>,
    pinned: HashSet<ExpertKey>,
    in_use: HashSet<ExpertKey>,
    warm_lru: BTreeSet<LruToken>,
    pending: HashMap<ExpertKey, PendingLoad>,
}

impl Inventory {
    pub fn new(host_budget_bytes: u64) -> Self {
        Self {
            host_budget: host_budget_bytes,
            host_used: 0,
            clock: 0,
            devices: HashMap::new(),
            entries: HashMap::new(),
            pinned: HashSet::new(),
            in_use: HashSet::new(),
            warm_lru: BTreeSet::new(),
            pending: HashMap::new(),
        }
    }

    pub fn add_device(&mut self, device: u32, budget_bytes: u64) {
        self.devices.insert(
            device,
            DeviceState {
                budget: budget_bytes,
                used: 0,
                lru: BTreeSet::new(),
            },
        );
    }

    pub fn register(&mut self, key: ExpertKey, bytes: u64, tier: Tier) -> Result<(), Error> {
        if self.pinned.contains(&key) || self.in_use.contains(&key) {
            return Ok(());
        }

        if let Some(entry) = self.entries.get(&key) {
            let old_tier = entry.tier.clone();

            if old_tier == tier {
                let delta = bytes.saturating_sub(entry.bytes);
                let shrink = entry.bytes.saturating_sub(bytes);

                match tier {
                    Tier::Hot { device } => {
                        if let Some(state) = self.devices.get_mut(&device) {
                            state.used = state.used.saturating_sub(shrink);
                            if state.used + delta > state.budget {
                                state.used = state.used.saturating_sub(delta);
                                return Ok(());
                            }
                            state.used += delta;
                        }
                    }
                    Tier::Warm => {
                        self.host_used = self.host_used.saturating_sub(shrink);
                        if self.host_used + delta > self.host_budget {
                            self.host_used = self.host_used.saturating_sub(delta);
                            return Ok(());
                        }
                        self.host_used += delta;
                    }
                    Tier::Cold => {}
                }

                let entry = self.entries.get_mut(&key).unwrap();
                entry.bytes = bytes;
                return Ok(());
            }

            self.remove_from_tier(&key);
        }

        let last_used = self.clock;
        self.entries.insert(
            key.clone(),
            Entry {
                bytes,
                tier: tier.clone(),
                last_used,
            },
        );

        match tier {
            Tier::Hot { device } => {
                if !self.devices.contains_key(&device) {
                    self.entries.remove(&key);
                    return Err(Error::UnknownDevice);
                }

                let free = self.devices[&device].budget.saturating_sub(self.devices[&device].used);
                if bytes > free {
                    self.entries.remove(&key);
                    return Err(Error::OverBudget);
                }

                self.insert_into_hot(&key, bytes, device);
            }
            Tier::Warm => {
                if bytes > self.host_budget.saturating_sub(self.host_used) {
                    self.entries.remove(&key);
                    return Err(Error::OverBudget);
                }
                self.insert_into_warm(&key, bytes);
            }
            Tier::Cold => {}
        }

        Ok(())
    }

    pub fn pin(&mut self, key: ExpertKey) {
        self.pinned.insert(key);
    }

    pub fn unpin(&mut self, key: ExpertKey) {
        self.pinned.remove(&key);
    }

    pub fn acquire(&mut self, key: ExpertKey) {
        self.in_use.insert(key.clone());
        self.touch_lru(&key);
    }

    pub fn release(&mut self, key: ExpertKey) {
        self.in_use.remove(&key);
    }

    pub fn request(&mut self, key: ExpertKey, device: u32) -> Result<Decision, Error> {
        if !self.devices.contains_key(&device) {
            return Err(Error::UnknownDevice);
        }
        if self.entries.get(&key).is_none() {
            return Err(Error::UnknownKey);
        }

        self.touch_lru(&key);

        if let Some(entry) = self.entries.get(&key) {
            if entry.tier == (Tier::Hot { device }) {
                return Ok(Decision::Resident);
            }
        }

        self.load_to(&key, device)
    }

    pub fn prefetch(&mut self, key: ExpertKey, device: u32, deadline_ms: u64) -> Result<(), Error> {
        if !self.devices.contains_key(&device) {
            return Err(Error::UnknownDevice);
        }
        if self.entries.get(&key).is_none() {
            return Err(Error::UnknownKey);
        }
        self.pending.insert(
            key,
            PendingLoad {
                device,
                deadline_ms,
            },
        );
        Ok(())
    }

    pub fn commit_load(&mut self, key: ExpertKey) -> Result<Decision, Error> {
        let pending = self
            .pending
            .remove(&key)
            .ok_or(Error::NotPrefetched)?;

        let device = pending.device;

        if !self.devices.contains_key(&device) {
            return Err(Error::UnknownDevice);
        }
        if self.entries.get(&key).is_none() {
            return Err(Error::UnknownKey);
        }

        self.touch_lru(&key);

        if let Some(entry) = self.entries.get(&key) {
            if entry.tier == (Tier::Hot { device }) {
                return Ok(Decision::Resident);
            }
        }

        self.load_to(&key, device)
    }

    fn load_to(&mut self, key: &ExpertKey, device: u32) -> Result<Decision, Error> {
        let entry = self.entries.get(key).unwrap();
        let from = entry.tier.clone();
        let bytes = entry.bytes;

        if bytes > self.devices[&device].budget {
            return Err(Error::OverBudget);
        }

        let free = self.devices[&device].budget.saturating_sub(self.devices[&device].used);
        let needed = bytes.saturating_sub(free);

        let evict = self.choose_evictions(device, needed)?;

        for evicted in &evict {
            self.evict_hot_to_warm(evicted, device)?;
        }

        self.remove_from_tier(key);
        self.entries.insert(
            key.clone(),
            Entry {
                bytes,
                tier: Tier::Hot { device },
                last_used: self.clock,
            },
        );
        self.insert_into_hot(key, bytes, device);

        Ok(Decision::Load {
            from,
            to: Tier::Hot { device },
            evict,
        })
    }

    pub fn invalidate_checkpoint(&mut self, model_id: &str, new_sha: &str) {
        let stale: Vec<ExpertKey> = self
            .entries
            .iter()
            .filter(|(key, _)| key.model_id == model_id && key.checkpoint_sha != new_sha)
            .map(|(key, _)| key.clone())
            .collect();

        for key in stale {
            self.remove_from_tier(&key);
            self.entries.remove(&key);
            self.pending.remove(&key);
            self.pinned.remove(&key);
            self.in_use.remove(&key);
        }
    }

    pub fn tier_of(&self, key: &ExpertKey) -> Option<Tier> {
        self.entries.get(key).map(|entry| entry.tier.clone())
    }

    pub fn used_bytes(&self, device: u32) -> u64 {
        self.devices.get(&device).map(|state| state.used).unwrap_or(0)
    }

    fn touch_lru(&mut self, key: &ExpertKey) {
        self.clock = self.clock.wrapping_add(1);

        if let Some(entry) = self.entries.get_mut(key) {
            let old_token = key.lru_token(entry.last_used);

            match entry.tier {
                Tier::Hot { device } => {
                    if let Some(state) = self.devices.get_mut(&device) {
                        state.lru.remove(&old_token);
                    }
                }
                Tier::Warm => {
                    self.warm_lru.remove(&old_token);
                }
                Tier::Cold => {}
            }

            entry.last_used = self.clock;

            let new_token = key.lru_token(entry.last_used);

            match entry.tier {
                Tier::Hot { device } => {
                    if let Some(state) = self.devices.get_mut(&device) {
                        state.lru.insert(new_token);
                    }
                }
                Tier::Warm => {
                    self.warm_lru.insert(new_token);
                }
                Tier::Cold => {}
            }
        }
    }

    fn insert_into_hot(&mut self, key: &ExpertKey, bytes: u64, device: u32) {
        let last_used = self
            .entries
            .get(key)
            .map(|entry| entry.last_used)
            .unwrap_or(self.clock);

        let token = key.lru_token(last_used);

        if let Some(state) = self.devices.get_mut(&device) {
            state.used += bytes;
            state.lru.insert(token);
        }
    }

    fn insert_into_warm(&mut self, key: &ExpertKey, bytes: u64) {
        let last_used = self
            .entries
            .get(key)
            .map(|entry| entry.last_used)
            .unwrap_or(self.clock);

        self.host_used += bytes;
        self.warm_lru.insert(key.lru_token(last_used));
    }

    fn remove_from_tier(&mut self, key: &ExpertKey) {
        if let Some(entry) = self.entries.get(key) {
            let token = key.lru_token(entry.last_used);

            match entry.tier {
                Tier::Hot { device } => {
                    if let Some(state) = self.devices.get_mut(&device) {
                        state.used = state.used.saturating_sub(entry.bytes);
                        state.lru.remove(&token);
                    }
                }
                Tier::Warm => {
                    self.host_used = self.host_used.saturating_sub(entry.bytes);
                    self.warm_lru.remove(&token);
                }
                Tier::Cold => {}
            }
        }
    }

    fn choose_evictions(&mut self, device: u32, needed: u64) -> Result<Vec<ExpertKey>, Error> {
        if needed == 0 {
            return Ok(Vec::new());
        }

        let mut evicted = Vec::new();
        let mut freed = 0u64;

        let tokens: Vec<LruToken> = self
            .devices
            .get(&device)
            .map(|state| state.lru.iter().cloned().collect())
            .unwrap_or_default();

        for token in tokens {
            if freed >= needed {
                break;
            }

            let key = ExpertKey {
                model_id: token.1,
                checkpoint_sha: token.2,
                layer: token.3,
                expert: token.4,
            };

            if self.pinned.contains(&key) || self.in_use.contains(&key) {
                continue;
            }

            let bytes = self
                .entries
                .get(&key)
                .map(|entry| entry.bytes)
                .unwrap_or(0);

            evicted.push(key);
            freed = freed.saturating_add(bytes);
        }

        if freed < needed {
            return Err(Error::OverBudget);
        }

        Ok(evicted)
    }

    fn evict_hot_to_warm(&mut self, key: &ExpertKey, _device: u32) -> Result<(), Error> {
        let bytes = self
            .entries
            .get(key)
            .map(|entry| entry.bytes)
            .unwrap_or(0);

        self.remove_from_tier(key);

        if bytes > self.host_budget.saturating_sub(self.host_used) {
            let mut freed = 0u64;
            let needed = bytes - self.host_budget.saturating_sub(self.host_used);

            let warm_tokens: Vec<LruToken> = self.warm_lru.iter().cloned().collect();

            for token in warm_tokens {
                if freed >= needed {
                    break;
                }

                let warm_key = ExpertKey {
                    model_id: token.1,
                    checkpoint_sha: token.2,
                    layer: token.3,
                    expert: token.4,
                };

                if self.pinned.contains(&warm_key) || self.in_use.contains(&warm_key) {
                    continue;
                }

                let warm_bytes = self
                    .entries
                    .get(&warm_key)
                    .map(|entry| entry.bytes)
                    .unwrap_or(0);

                self.remove_from_tier(&warm_key);
                if let Some(entry) = self.entries.get_mut(&warm_key) {
                    entry.tier = Tier::Cold;
                }

                freed = freed.saturating_add(warm_bytes);
            }

            if freed < needed {
                return Err(Error::OverBudget);
            }
        }

        self.insert_into_warm(key, bytes);

        if let Some(entry) = self.entries.get_mut(key) {
            entry.tier = Tier::Warm;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(model: &str, sha: &str, layer: u32, expert: u32) -> ExpertKey {
        ExpertKey::new(model, sha, layer, expert)
    }

    #[test]
    fn lru_order_picks_least_recently_used_key() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let a = key("m", "s", 0, 0);
        let b = key("m", "s", 0, 1);
        let c = key("m", "s", 0, 2);

        inv.register(a.clone(), 50, Tier::Hot { device: 0 }).unwrap();
        inv.register(b.clone(), 50, Tier::Hot { device: 0 }).unwrap();

        inv.request(a.clone(), 0).unwrap();

        let decision = inv.request(c.clone(), 0).unwrap();

        assert_eq!(
            decision,
            Decision::Load {
                from: Tier::Cold,
                to: Tier::Hot { device: 0 },
                evict: vec![b.clone()],
            }
        );

        assert_eq!(inv.tier_of(&b), Some(Tier::Warm));
        assert_eq!(inv.tier_of(&c), Some(Tier::Hot { device: 0 }));
    }

    #[test]
    fn pinned_key_is_never_evicted() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let pinned = key("m", "s", 0, 0);
        let evictable = key("m", "s", 0, 1);
        let incoming = key("m", "s", 0, 2);

        inv.register(pinned.clone(), 60, Tier::Hot { device: 0 }).unwrap();
        inv.register(evictable.clone(), 40, Tier::Hot { device: 0 }).unwrap();
        inv.pin(pinned.clone());

        inv.request(incoming.clone(), 0).unwrap();

        assert_eq!(inv.tier_of(&pinned), Some(Tier::Hot { device: 0 }));
        assert_eq!(inv.tier_of(&evictable), Some(Tier::Warm));
        assert_eq!(inv.tier_of(&incoming), Some(Tier::Hot { device: 0 }));
    }

    #[test]
    fn in_use_key_is_never_evicted() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let busy = key("m", "s", 0, 0);
        let evictable = key("m", "s", 0, 1);
        let incoming = key("m", "s", 0, 2);

        inv.register(busy.clone(), 60, Tier::Hot { device: 0 }).unwrap();
        inv.register(evictable.clone(), 40, Tier::Hot { device: 0 }).unwrap();
        inv.acquire(busy.clone());

        inv.request(incoming.clone(), 0).unwrap();

        assert_eq!(inv.tier_of(&busy), Some(Tier::Hot { device: 0 }));
        assert_eq!(inv.tier_of(&evictable), Some(Tier::Warm));
        assert_eq!(inv.tier_of(&incoming), Some(Tier::Hot { device: 0 }));
    }

    #[test]
    fn over_budget_when_pinned_keys_alone_exceed_budget() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let pinned = key("m", "s", 0, 0);
        let incoming = key("m", "s", 0, 1);

        inv.register(pinned.clone(), 100, Tier::Hot { device: 0 });
        inv.pin(pinned.clone());

        assert_eq!(inv.request(incoming.clone(), 0), Err(Error::OverBudget));
        assert_eq!(inv.tier_of(&pinned), Some(Tier::Hot { device: 0 }));
        assert_eq!(inv.tier_of(&incoming), None);
    }

    #[test]
    fn checkpoint_invalidation_removes_stale_keys() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let stale = key("m", "old", 0, 0);
        let fresh = key("m", "new", 0, 1);

        inv.register(stale.clone(), 50, Tier::Hot { device: 0 }).unwrap();
        inv.register(fresh.clone(), 50, Tier::Hot { device: 0 }).unwrap();

        inv.invalidate_checkpoint("m", "new");

        assert_eq!(inv.tier_of(&stale), None);
        assert_eq!(inv.tier_of(&fresh), Some(Tier::Hot { device: 0 }));
        assert_eq!(inv.used_bytes(0), 50);
    }

    #[test]
    fn prefetch_then_commit_load_moves_tier() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let k = key("m", "s", 0, 0);

        inv.register(k.clone(), 50, Tier::Warm).unwrap();
        inv.prefetch(k.clone(), 0, 1000).unwrap();

        assert_eq!(inv.tier_of(&k), Some(Tier::Warm));

        inv.commit_load(k.clone()).unwrap();

        assert_eq!(inv.tier_of(&k), Some(Tier::Hot { device: 0 }));
        assert_eq!(inv.used_bytes(0), 50);
    }

    #[test]
    fn randomized_sequence_preserves_budget_and_protection() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);
        inv.add_device(1, 120);

        let mut rng = XorShift::new(0x1234_5678_9abc_def0);

        for _ in 0..10_000 {
            let op = rng.next() % 8;
            let device = (rng.next() % 2) as u32;
            let layer = (rng.next() % 4) as u32;
            let expert = (rng.next() % 16) as u32;
            let bytes = 10 + (rng.next() % 40);
            let model = "m";
            let sha = if rng.next() % 2 == 0 { "old" } else { "new" };
            let k = key(model, sha, layer, expert);

            match op {
                0 => { let _ = inv.register(k.clone(), bytes, Tier::Cold); }
                1 => { let _ = inv.register(k.clone(), bytes, Tier::Warm); }
                2 => { let _ = inv.register(k.clone(), bytes, Tier::Hot { device }); }
                3 => {
                    inv.pin(k.clone());
                    inv.acquire(k.clone());
                }
                4 => {
                    inv.unpin(k.clone());
                    inv.release(k.clone());
                }
                5 => {
                    let _ = inv.request(k.clone(), device);
                }
                6 => {
                    let _ = inv.prefetch(k.clone(), device, rng.next() % 1000);
                    let _ = inv.commit_load(k.clone());
                }
                _ => inv.invalidate_checkpoint(model, "new"),
            }

            for dev in 0..2 {
                assert!(inv.used_bytes(dev) <= inv.devices[&dev].budget);
            }

            for k in &inv.pinned {
                if let Some(entry) = inv.entries.get(k) {
                    assert_eq!(entry.tier, inv.tier_of(k).unwrap());
                }
            }

            for k in &inv.in_use {
                if let Some(entry) = inv.entries.get(k) {
                    assert_eq!(entry.tier, inv.tier_of(k).unwrap());
                }
            }
        }
    }

    #[test]
    fn request_on_unknown_key_returns_unknown_key() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let unknown = key("m", "s", 0, 99);
        assert_eq!(inv.request(unknown, 0), Err(Error::UnknownKey));
    }

    #[test]
    fn commit_load_on_unknown_key_returns_unknown_key() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let k = key("m", "s", 0, 0);
        inv.register(k.clone(), 50, Tier::Warm).unwrap();
        inv.prefetch(k.clone(), 0, 1000).unwrap();

        let unknown = key("m", "s", 0, 99);
        assert_eq!(inv.commit_load(unknown), Err(Error::UnknownKey));
    }

    #[test]
    fn prefetch_on_unknown_device_returns_unknown_device() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let k = key("m", "s", 0, 0);
        inv.register(k.clone(), 50, Tier::Warm).unwrap();

        assert_eq!(inv.prefetch(k.clone(), 99, 1000), Err(Error::UnknownDevice));
    }

    #[test]
    fn prefetch_on_unknown_key_returns_unknown_key() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let unknown = key("m", "s", 0, 99);
        assert_eq!(inv.prefetch(unknown, 0, 1000), Err(Error::UnknownKey));
    }

    #[test]
    fn register_hot_on_unknown_device_returns_unknown_device() {
        let mut inv = Inventory::new(1000);

        let k = key("m", "s", 0, 0);
        assert_eq!(inv.register(k, 50, Tier::Hot { device: 0 }), Err(Error::UnknownDevice));
    }

    #[test]
    fn register_warm_over_budget_returns_over_budget() {
        let mut inv = Inventory::new(100);

        let k = key("m", "s", 0, 0);
        assert_eq!(inv.register(k, 101, Tier::Warm), Err(Error::OverBudget));
    }

    #[test]
    fn register_hot_over_budget_returns_over_budget() {
        let mut inv = Inventory::new(1000);
        inv.add_device(0, 100);

        let k = key("m", "s", 0, 0);
        assert_eq!(inv.register(k, 101, Tier::Hot { device: 0 }), Err(Error::OverBudget));
    }

    struct XorShift {
        state: u64,
    }

    impl XorShift {
        fn new(seed: u64) -> Self {
            Self {
                state: seed | 1,
            }
        }

        fn next(&mut self) -> u64 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            self.state
        }
    }
}
