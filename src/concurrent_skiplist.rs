use rand::prelude::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Geometric};
use std::convert::TryInto;
use std::fmt::Debug;
use std::iter::FusedIterator;
use std::mem;
use std::ptr::NonNull;
use std::sync::atomic::{self, AtomicPtr, AtomicUsize};

type Link<K, V> = Option<*mut SkipNode<K, V>>;

/// A node in the skip list.
#[derive(Debug)]
struct SkipNode<K: Ord + Debug, V: Clone> {
    /// The key should only be `None` for the `head` node.
    key: Option<K>,
    /// The Value should only be `None` for the `head` node.
    value: Option<V>,
    levels: Vec<Link<K, V>>,
}

/// Private methods
impl<K: Ord + Debug, V: Clone> SkipNode<K, V> {
    fn new(key: K, value: V, height: usize) -> Self {
        SkipNode {
            key: Some(key),
            value: Some(value),
            levels: vec![None; height],
        }
    }

    fn head() -> Self {
        SkipNode {
            key: None,
            value: None,
            levels: vec![],
        }
    }

    /// Get an immutable reference to the next node at level 0 if it exists. Otherwise, `None`.
    fn next(&self) -> Option<&SkipNode<K, V>> {
        self.next_at_level(0)
    }

    /// Get an immutable reference to the next node at the specified level if it exists.
    /// Otherwise, `None`.
    fn next_at_level(&self, level: usize) -> Option<&SkipNode<K, V>> {
        if self.levels.is_empty() {
            return None;
        }

        self.levels[level].as_ref().and_then(|node_ptr| {
            // Get atomic access to the underlying pointer
            let atomic_ptr: AtomicPtr<SkipNode<K, V>> = AtomicPtr::from(*node_ptr);
            let maybe_node = atomic_ptr.load(atomic::Ordering::Acquire);

            unsafe {
                /*
                SAFETY:
                This is safe because there are no removals from the skip list and because we
                use an acquire load to get the pointer. We also do a null check right before.
                The value must also be initialized or the `Option` containing this pointer
                would be `None` for no next pointer.
                */
                maybe_node.as_ref()
            }
        })
    }

    /// Get mutable reference to the next node at the specified level if it exists.
    /// Otherwise, `None`.
    fn next_at_level_mut(&mut self, level: usize) -> Option<&mut SkipNode<K, V>> {
        if self.levels.is_empty() {
            return None;
        }

        self.levels[level].as_mut().and_then(|node_ptr| {
            // Get atomic access to the underlying pointer
            let atomic_ptr: AtomicPtr<SkipNode<K, V>> = AtomicPtr::from(*node_ptr);
            let maybe_node = atomic_ptr.load(atomic::Ordering::Acquire);

            unsafe {
                /*
                SAFETY:
                This is safe because there are no removals from the skip list and because we
                use an acquire load to get the pointer. We also do a null check right before.
                Insertions also require external synchronization so we can be sure that the
                users of the skip list only have one thread performing mutations. The value must
                also be initialized or the `Option` containing this pointer would be `None` for no
                next pointer.
                */
                maybe_node.as_mut()
            }
        })
    }
}

/// A skip list that allows for concurrent reads without external synchronization (i.e. via locks).
///
/// # Concurrency
///
/// **NOTE: Insertions will require an external lock!**
///
/// This skip list implementation is used by an append only database project that only needs append
/// and not deletion. Because of this use-case, deletion is not implemented. It is a larger project
/// to support full functionality here.
///
/// Because reads are lock-free, iterators may see new values pop up from concurrent insertions.
/// However, we should not miss any values that were present when the iterator was created.
///
/// # Safety
///
/// Invariants:
///
/// - If an Link exists it must be valid to dereference to a `SkipNode`.
/// - Nodes are never deleted.
/// - Only insertions modify the pointers and this is done with [`Ordering::Release`] stores.
///
/// [`Ordering::Release`]: std::sync::atomic::Ordering
#[derive(Debug)]
pub struct ConcurrentSkipList<K: Ord + Debug, V: Clone> {
    /// A pointer to a dummy head node.
    head_ptr: NonNull<SkipNode<K, V>>,
    /// The number of elements in the skip list.
    length: AtomicUsize,
    /// The probability of success used in probability distribution for determining height of a new
    /// node. Defaults to 0.25.
    probability: f64,
    /// Approximate memory used by the skip list in number of bytes.
    approximate_mem_usage: AtomicUsize,
}

// Public methods of SkipList
impl<K: Ord + Debug, V: Clone> ConcurrentSkipList<K, V> {
    /// Create a new skip list.
    ///
    /// `probability` is the probability of success used in probability distribution for determining
    /// height of a new node. Defaults to 0.25.
    ///
    /// # Examples
    /// ```
    /// use nerdondon_hopscotch::skiplist::ConcurrentSkiplist;
    ///
    /// let skiplist = SkipList::<i32, String>::new(None);
    /// ```
    pub fn new(probability: Option<f64>) -> Self {
        let mut head = Box::new(SkipNode::head());
        let skiplist = ConcurrentSkipList {
            head_ptr: NonNull::new(head.as_mut()).unwrap(),
            length: AtomicUsize::new(0),
            probability: probability.unwrap_or(0.25),
            approximate_mem_usage: AtomicUsize::new(0),
        };
        Box::leak(head);

        // TODO: Make size tracking a feature?
        let size = mem::size_of_val(&skiplist) + mem::size_of::<SkipNode<K, V>>();
        skiplist
            .approximate_mem_usage
            .store(size, atomic::Ordering::Release);

        skiplist
    }

    /// Get an immutable reference to the value corresponding to the specified `key`.
    ///
    /// Returns `Some(V)` if found and `None` if not
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.is_empty() {
            return None;
        }

        let potential_record = self.find_greater_or_equal_node(key);
        if let Some(record) = potential_record {
            if record.key.as_ref().unwrap().eq(key) {
                return record.value.as_ref();
            }
        }

        None
    }

    /// Insert a key-value pair.
    ///
    /// # Safety
    ///
    /// The caller must have an external lock on this skiplist in order to safely call this method.
    ///
    /// # Examples
    /// ```
    /// use nerdondon_hopscotch::skiplist::ConcurrentSkiplist;
    ///
    /// let skiplist = ConcurrentSkiplist::<i32, String>::new(None);
    /// skiplist.insert(2, "banana".to_string());
    /// skiplist.insert(3, "orange".to_string());
    /// skiplist.insert(1, "apple".to_string());
    ///
    /// let some_value = skiplist.get(&2).unwrap();
    /// assert_eq!(some_value, "banana");
    /// ```
    pub unsafe fn insert(&self, key: K, value: V) {
        let new_node_height = self.random_height();
        if new_node_height > self.height() {
            self.adjust_head(new_node_height);
        }

        // Track where we end on each level
        let mut nodes_to_update: Vec<Option<*mut SkipNode<K, V>>> = vec![None; self.height()];
        let list_height = self.height();
        let mut current_node_ptr: *mut _ = self.head_mut();

        // Start iteration at the top of the skip list "towers" and find the insert position at each
        // level
        for level_idx in (0..list_height).rev() {
            // Get an optional of the next node
            // SAFETY: It is safe to dereference the raw pointer because we have a lock
            let mut maybe_next_node = (*current_node_ptr).next_at_level_mut(level_idx);

            while let Some(next_node) = maybe_next_node {
                match next_node.key.as_ref().unwrap().cmp(&key) {
                    std::cmp::Ordering::Less => {
                        current_node_ptr = next_node;
                        maybe_next_node = next_node.next_at_level_mut(level_idx);
                    }
                    _ => break,
                }
            }

            // Keep track of the node we stopped at. This is either the node right before our new
            // node or head node of the level if no lesser node was found.
            //
            // We are effectively giving out multiple mutable pointers right here. This seems ok
            // since we are the only writer (it is part of the contract that callers have
            // a lock). The bottom loop also ensures that only one cast of a potentially duplicate
            // pointer is active at a time. Explicitly, two mutable pointers that point to the same
            // memory location are never casted to mutable references at the same time.
            nodes_to_update[level_idx] = Some(current_node_ptr);
        }

        let mut new_node = Box::new(SkipNode::new(key, value, new_node_height));
        let new_node_ptr = new_node.as_mut() as *mut SkipNode<K, V>;
        for level_idx in (0..new_node_height).rev() {
            /*
            SAFETY:
            Dereferencing the *mut is ok here because we have exclusive access to modifying the
            node pointers and we casted into a *mut above. The node we casted is from a pointer
            that we got via an acquire load which also ensure validity.
            */
            let previous_node = &mut **(nodes_to_update[level_idx].as_mut().unwrap());

            // Set the new node's next pointer for this level. Specifically, `previous_node`'s next
            // node at this level becomes `new_node`'s next node at this level
            new_node.levels[level_idx] = previous_node.levels[level_idx].take();

            // Set the next pointer of the previous node to the new node
            previous_node.levels[level_idx] = Some(new_node_ptr);
        }

        // Book keeping for size
        // The additional usage should be from the size of the new node and the size of references
        // to this new node. This is multiplied by 2 to approximate the storage in the `levels`
        // vector. `mem::size_of` and `mem::size_of_val` does not actually get the size of vectors
        // since vectors are allocated to the heap and only a pointer is stored in the field.
        self.approximate_mem_usage.fetch_add(
            mem::size_of::<SkipNode<K, V>>() + (2 * mem::size_of::<Link<K, V>>() * new_node_height),
            atomic::Ordering::AcqRel,
        );
        self.inc_length();

        /*
        `Box::leak` is called so that the node does not get deallocated at the end of the function.
        The `SkipList::remove` method will ensure to reform the box from the pointer so that the
        node is de-allocated on removal.
        */
        Box::leak(new_node);
    }

    /// Return a reference to the key and value of the first node with a key that is greater than
    /// or equal to the target key.
    pub fn find_greater_or_equal(&self, target: &K) -> Option<(&K, &V)> {
        if self.is_empty() {
            return None;
        }

        self.find_greater_or_equal_node(target)
            .map(|node| (node.key.as_ref().unwrap(), node.value.as_ref().unwrap()))
    }

    /// Return a reference to the key and value of the last node with a key that is less than the
    /// target key.
    pub fn find_less_than(&self, target: &K) -> Option<(&K, &V)> {
        self.find_less_than_node(target).and_then(|node| {
            // Only the head node has empty keys and values, so this means the search
            // stayed on the head node. This can happen if we are searching for a target
            // with less than all of the keys in the skip list. Return `None` in this case.
            node.key.as_ref()?;

            Some((node.key.as_ref().unwrap(), node.value.as_ref().unwrap()))
        })
    }

    /// Return a reference to the key and value of the first node in the skip list if there is a
    /// node. Otherwise, it returns `None`.
    pub fn first(&self) -> Option<(&K, &V)> {
        self.head()
            .next()
            .map(|node| (node.key.as_ref().unwrap(), node.value.as_ref().unwrap()))
    }

    /// Return a reference to the key and value of the last node in the skip list if there is a
    /// node. Otherwise, it returns `None`.
    pub fn last(&self) -> Option<(&K, &V)> {
        if self.is_empty() {
            return None;
        }

        let mut current_node = self.head();
        for level_idx in (0..self.height()).rev() {
            let mut maybe_next_node = current_node.next_at_level(level_idx);
            while let Some(next_node) = maybe_next_node {
                current_node = next_node;
                maybe_next_node = next_node.next_at_level(level_idx);
            }
        }

        Some((
            current_node.key.as_ref().unwrap(),
            current_node.value.as_ref().unwrap(),
        ))
    }

    /// The number of elements in the skip list.
    pub fn len(&self) -> usize {
        self.length.load(atomic::Ordering::Acquire)
    }

    /// Returns true if the skip list does not hold any elements; otherwise false.
    pub fn is_empty(&self) -> bool {
        self.length.load(atomic::Ordering::Acquire) == 0
    }

    /// Get the approximate amount of memory used in number of bytes.
    pub fn get_approx_mem_usage(&self) -> usize {
        self.approximate_mem_usage.load(atomic::Ordering::Acquire)
    }

    /// An iterator visiting each node in order.
    ///
    /// Returns values of (&'a K, &'a V)
    pub fn iter(&self) -> NodeIterHelper<'_, K, V> {
        if self.is_empty() {
            return NodeIterHelper { next: None };
        }

        let next = self.head().next();

        NodeIterHelper { next }
    }
}

/// Implementation for keys and values that implement `Clone`
impl<K, V> ConcurrentSkipList<K, V>
where
    K: Ord + Debug + Clone,
    V: Clone,
{
    /// Eagerly returns the entries stored in the skip list as `Vec<(K,V)>` with cloned values.
    ///
    /// # Examples
    /// ```
    /// use nerdondon_hopscotch::skiplist::ConcurrentSkiplist;
    ///
    /// let skiplist = ConcurrentSkiplist::<i32, String>::new(None);
    /// skiplist.insert(2, "banana".to_string());
    /// skiplist.insert(3, "orange".to_string());
    /// skiplist.insert(1, "apple".to_string());
    ///
    /// let entries = skiplist.entries();
    /// assert_eq!(
    ///   entries,
    ///   [
    ///     (1, "apple".to_string()),
    ///     (2, "banana".to_string()),
    ///     (3, "orange".to_string())
    ///   ]
    /// );
    /// ```
    pub fn entries(&self) -> Vec<(K, V)> {
        let mut kv_pairs = Vec::<(K, V)>::with_capacity(self.len());
        for (key, value) in self.iter() {
            let cloned_key = key.clone();
            let cloned_value = value.clone();
            kv_pairs.push((cloned_key, cloned_value));
        }

        kv_pairs
    }

    /// Print out the keys of elements in the skip list.
    pub fn print_keys(&self) {
        let entries = self.entries();
        for (key, _value) in entries {
            println!("Key: {:?}", key);
        }
    }
}

// Private methods of SkipList
impl<K: Ord + Debug, V: Clone> ConcurrentSkipList<K, V> {
    /// Return a reference to the head node.
    fn head(&self) -> &SkipNode<K, V> {
        // SAFTEY: This is safe because the head is always guaranteed to exist.
        unsafe { self.head_ptr.as_ref() }
    }

    /// Return a mutable reference to the head node.
    ///
    /// FIXME: I'm just doing this to keep my other project going. We don't really care about a
    /// data race because we assert that a lock is around insert. Porting C++ to Rust kinda hurts :/
    #[allow(clippy::mut_from_ref)]
    fn head_mut(&self) -> &mut SkipNode<K, V> {
        // SAFTEY: This is safe because the head is always guaranteed to exist.
        unsafe { &mut *self.head_ptr.as_ptr() }
    }

    /// The current maximum height of the skip list.
    fn height(&self) -> usize {
        self.head().levels.len()
    }

    /// Generates a random height according to a geometric distribution.
    fn random_height(&self) -> usize {
        // 0x6261746d616e6e => Batmann for my dog :)
        let mut rng = StdRng::seed_from_u64(0x6261746d616e6e);
        let distribution = Geometric::new(self.probability).unwrap();
        let sample = distribution.sample(&mut rng);

        if sample == 0 || sample > (self.height() + 3).try_into().unwrap() {
            // Avoid zero height and only increase the height by one if the number drawn is much
            // more than the current maximum height. This is just an arbitrary cap on the growth in
            // height of the skip list.
            self.height() + 1
        } else {
            sample as usize
        }
    }

    /// Adjust the levels stored in head to match a new height.
    fn adjust_head(&self, new_height: usize) {
        if self.height() >= new_height {
            return;
        }

        let height_difference = new_height - self.height();
        for _ in 0..height_difference {
            self.head_mut().levels.push(None);
        }
    }

    /// Increment length by 1.
    fn inc_length(&self) {
        self.length.fetch_add(1, atomic::Ordering::AcqRel);
    }

    /// Return a reference to the last node with a key that is less than the target key.
    fn find_less_than_node(&self, target: &K) -> Option<&SkipNode<K, V>> {
        if self.is_empty() {
            return None;
        }

        let mut current_node = self.head();
        // Start iteration at the top of the skip list "towers" and iterate through pointers at the
        // current level. If we skipped past our key, move down a level.
        for level_idx in (0..self.height()).rev() {
            // Get an optional of the next node
            let mut maybe_next_node = current_node.next_at_level(level_idx);

            while let Some(next_node) = maybe_next_node {
                match next_node.key.as_ref().unwrap().cmp(target) {
                    std::cmp::Ordering::Less => {
                        current_node = next_node;
                        maybe_next_node = next_node.next_at_level(level_idx);
                    }
                    _ => break,
                }
            }
        }

        Some(current_node)
    }

    /// Return a reference to the first node with a key that is greater than or equal to the target
    /// key.
    fn find_greater_or_equal_node(&self, target: &K) -> Option<&SkipNode<K, V>> {
        self.find_less_than_node(target).and_then(|prev_node| {
            // We potentially found a node right before the first node with a key greater than or
            // equal to the our target key. Move the pointer forward one time to check if we
            // actually have the node we are looking for or if we hit the end of the levels without
            // finding anything.
            prev_node.next()
        })
    }
}

/// An iterator adapter over the nodes of a `SkipList`.
///
/// This `struct` is created by the [`iter`] method.
///
/// [`iter`]: SkipList::iter
pub struct NodeIterHelper<'a, K, V>
where
    K: Ord + Debug,
    V: Clone,
{
    next: Option<&'a SkipNode<K, V>>,
}

impl<'a, K, V> Iterator for NodeIterHelper<'a, K, V>
where
    K: Ord + Debug,
    V: Clone,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let wrapped_current_node = self.next;

        // Short-circuit return `None`
        wrapped_current_node?;

        let current_node = wrapped_current_node.unwrap();
        self.next = current_node.next();

        Some((
            current_node.key.as_ref().unwrap(),
            current_node.value.as_ref().unwrap(),
        ))
    }
}

impl<'a, K, V> IntoIterator for &'a ConcurrentSkipList<K, V>
where
    K: Ord + Debug,
    V: Clone,
{
    type Item = (&'a K, &'a V);
    type IntoIter = NodeIterHelper<'a, K, V>;

    fn into_iter(self) -> NodeIterHelper<'a, K, V> {
        self.iter()
    }
}

impl<K, V> FusedIterator for NodeIterHelper<'_, K, V>
where
    K: Ord + Debug,
    V: Clone,
{
}

/// SAFETY: This is safe for because atomic operations are used when changing pointers.
unsafe impl<K, V> Send for ConcurrentSkipList<K, V>
where
    K: Ord + Debug,
    V: Clone,
{
}

/// SAFETY: This is safe for because atomic operations are used when changing pointers.
unsafe impl<K, V> Sync for ConcurrentSkipList<K, V>
where
    K: Ord + Debug,
    V: Clone,
{
}

impl<K, V> Drop for ConcurrentSkipList<K, V>
where
    K: Ord + Debug,
    V: Clone,
{
    fn drop(&mut self) {
        if self.is_empty() {
            return;
        }

        // SAFETY: This is safe be cause the head is guaranteed to always exist.
        let mut head = unsafe { Box::from_raw(self.head_ptr.as_ptr()) };

        let mut maybe_node_ptr = head.as_mut().levels[0].as_mut().map(|node_ptr| {
            // Get atomic access to the underlying pointer
            let atomic_ptr: AtomicPtr<SkipNode<K, V>> = AtomicPtr::from(*node_ptr);
            atomic_ptr.load(atomic::Ordering::Acquire)
        });

        while maybe_node_ptr.is_some() {
            let current_node_ptr = maybe_node_ptr.unwrap();

            if current_node_ptr.is_null() {
                break;
            }

            /*
            Re-box the allocation the pointer represents so that it can get dropped. Insert's
            leak the boxed node when it is created.

            It is ok to leave pointers in the dropped node's levels vector dangling because all
            nodes are getting dropped.
            */
            let mut current_node = unsafe {
                /*
                SAFETY:
                We check that there is a pointer in the option before entering the loop which
                guarantees that the node was initialized. We also check that the pointer is not
                null.
                */
                Box::from_raw(current_node_ptr)
            };

            maybe_node_ptr = current_node.levels[0].as_mut().map(|node_ptr| {
                // Get atomic access to the underlying pointer
                let atomic_ptr: AtomicPtr<SkipNode<K, V>> = AtomicPtr::from(*node_ptr);
                atomic_ptr.load(atomic::Ordering::Acquire)
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn with_an_empty_skiplist_get_returns_none() {
        let skiplist = ConcurrentSkipList::<i32, String>::new(None);
        assert_eq!(skiplist.get(&10), None);
    }

    #[test]
    fn with_an_empty_skiplist_is_empty_returns_true() {
        let skiplist = ConcurrentSkipList::<i32, String>::new(None);
        assert_eq!(skiplist.is_empty(), true);
    }

    #[test]
    fn with_an_empty_skiplist_len_returns_zero() {
        let skiplist = ConcurrentSkipList::<i32, String>::new(None);
        assert_eq!(skiplist.len(), 0);
    }

    #[test]
    fn with_an_empty_skiplist_insert_can_add_an_element() {
        let skiplist = ConcurrentSkipList::<i32, String>::new(None);

        // SAFETY: Single thread insert
        unsafe { skiplist.insert(1, "apple".to_string()) };

        assert_eq!(skiplist.len(), 1);
    }

    #[test]
    fn insert_can_add_an_element_after_an_existing_element() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(1, "apple".to_string());

            skiplist.insert(2, "banana".to_string());

            assert_eq!(skiplist.len(), 2);
        }
    }

    #[test]
    fn insert_can_add_an_element_before_an_existing_element() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());

            skiplist.insert(1, "apple".to_string());

            assert_eq!(skiplist.len(), 2);
        }
    }

    #[test]
    fn insert_can_add_an_element_between_existing_elements() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(1, "apple".to_string());

            skiplist.insert(2, "banana".to_string());

            assert_eq!(skiplist.len(), 3);
        }
    }

    #[test]
    fn get_an_element_at_the_head() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(1, "apple".to_string());
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(2, "banana".to_string());
            let expected_value = "apple".to_string();

            let actual_value = skiplist.get(&1).unwrap();

            assert_eq!(&expected_value, actual_value);
        }
    }

    #[test]
    fn get_an_element_in_the_middle() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());
            skiplist.insert(1, "apple".to_string());
            skiplist.insert(3, "orange".to_string());
            let expected_value = "banana".to_string();

            let actual_value = skiplist.get(&2).unwrap();

            assert_eq!(&expected_value, actual_value);
        }
    }

    #[test]
    fn get_an_element_at_the_tail() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(1, "apple".to_string());
            let expected_value = "orange".to_string();

            let actual_value = skiplist.get(&3).unwrap();

            assert_eq!(&expected_value, actual_value);
        }
    }

    #[test]
    fn with_a_non_empty_skiplist_getting_a_non_existent_element_returns_none() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(1, "apple".to_string());

            let actual_value = skiplist.get(&0);

            assert_eq!(None, actual_value);
        }
    }

    #[test]
    fn with_a_non_empty_skiplist_is_empty_returns_false() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(1, "apple".to_string());

            assert_eq!(skiplist.is_empty(), false);
        }
    }

    #[test]
    fn with_an_empty_skiplist_collect_returns_an_empty_vec() {
        let skiplist = ConcurrentSkipList::<i32, String>::new(None);

        let actual_value = skiplist.entries();

        assert_eq!(actual_value.len(), 0);
        assert_eq!(actual_value, []);
    }

    #[test]
    fn entries_returns_a_vec_with_the_key_value_pairs_of_elements() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(1, "apple".to_string());

            let actual_value = skiplist.entries();

            assert_eq!(actual_value.len(), 3);
            assert_eq!(
                actual_value,
                [
                    (1, "apple".to_string()),
                    (2, "banana".to_string()),
                    (3, "orange".to_string())
                ]
            );
        }
    }

    #[test]
    fn get_approx_mem_usage_provides_decent_estimates() {
        // SAFETY: Single thread insert
        unsafe {
            // Note that these estimates have only just some basis in reality. We do not attempt to get
            // too crazy with the estimates. Just make sure the numbers are somewhat sane.

            // Approximated initial usage
            // size of head node
            //   = 1 (None) + 1 (None) + 3 (vec pointer) + 0 (empty vec actual size) = 26
            // size of SkipList = 8 (length) + 8 (probability) + 8 (approx_mem_usage)  = 24
            let approx_initial_usage: usize = 50;

            // Approximate node size
            // size of `levels` actually = height of the skiplist * size of `Link`
            let base_node_usage: usize = mem::size_of::<SkipNode<u16, String>>();
            let link_size = mem::size_of::<Link<u16, String>>();

            let mut usage_approximation = approx_initial_usage;

            let skiplist = ConcurrentSkipList::<u16, String>::new(None);
            assert!(skiplist.get_approx_mem_usage() >= usage_approximation);

            skiplist.insert(1, "apple".to_string());
            usage_approximation += base_node_usage + 7 + skiplist.height() * link_size;
            assert!(skiplist.get_approx_mem_usage() > usage_approximation);

            skiplist.insert(2, "banana".to_string());
            usage_approximation += base_node_usage + 8 + skiplist.height() * link_size;
            assert!(skiplist.get_approx_mem_usage() > usage_approximation);
        }
    }

    #[test]
    fn with_a_non_empty_skiplist_first_returns_references_to_the_first_key_value_pair() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(1, "apple".to_string());
            skiplist.insert(4, "strawberry".to_string());
            skiplist.insert(5, "watermelon".to_string());

            assert_eq!(skiplist.first(), Some((&1, &"apple".to_string())));
        }
    }

    #[test]
    fn with_an_empty_skiplist_first_returns_none() {
        let skiplist = ConcurrentSkipList::<i32, String>::new(None);

        assert_eq!(skiplist.first(), None);
    }

    #[test]
    fn with_a_non_empty_skiplist_last_returns_references_to_the_last_key_value_pair() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(1, "apple".to_string());
            skiplist.insert(4, "strawberry".to_string());
            skiplist.insert(5, "watermelon".to_string());

            assert_eq!(skiplist.last(), Some((&5, &"watermelon".to_string())));
        }
    }

    #[test]
    fn with_an_empty_skiplist_last_returns_none() {
        let skiplist = ConcurrentSkipList::<i32, String>::new(None);

        assert_eq!(skiplist.last(), None);
    }

    #[test]
    fn with_a_non_empty_skiplist_find_greater_or_equal_returns_correct_responses() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(1, "apple".to_string());
            skiplist.insert(4, "strawberry".to_string());
            skiplist.insert(5, "watermelon".to_string());
            skiplist.insert(11, "grapefruit".to_string());
            skiplist.insert(12, "mango".to_string());

            // First element exists so it is found
            assert_eq!(
                skiplist.find_greater_or_equal(&1),
                Some((&1, &"apple".to_string()))
            );

            // Middle element exists so it is found
            assert_eq!(
                skiplist.find_greater_or_equal(&3),
                Some((&3, &"orange".to_string()))
            );

            // Target doesn't exist so it finds a greatest
            assert_eq!(
                skiplist.find_greater_or_equal(&7),
                Some((&11, &"grapefruit".to_string()))
            );

            // Last element exists so it is found
            assert_eq!(
                skiplist.find_greater_or_equal(&12),
                Some((&12, &"mango".to_string()))
            );

            // Greater than last element so it returns `None`
            assert_eq!(skiplist.find_greater_or_equal(&20), None);
        }
    }

    #[test]
    fn with_a_non_empty_skiplist_find_less_than_returns_correct_responses() {
        // SAFETY: Single thread insert
        unsafe {
            let skiplist = ConcurrentSkipList::<i32, String>::new(None);
            skiplist.insert(2, "banana".to_string());
            skiplist.insert(3, "orange".to_string());
            skiplist.insert(1, "apple".to_string());
            skiplist.insert(4, "strawberry".to_string());
            skiplist.insert(5, "watermelon".to_string());
            skiplist.insert(11, "grapefruit".to_string());
            skiplist.insert(12, "mango".to_string());

            // Finding a target less than every element in the list returns `None`
            assert_eq!(skiplist.find_less_than(&0), None);

            // Finding a target less than the first element returns `None`
            assert_eq!(skiplist.find_less_than(&1), None);

            // Finding a target less than an existing middle element
            assert_eq!(
                skiplist.find_less_than(&3),
                Some((&2, &"banana".to_string()))
            );

            // Finding a target less than a non-existent middle element
            assert_eq!(
                skiplist.find_less_than(&7),
                Some((&5, &"watermelon".to_string()))
            );

            // Finding a target greater than all elements returns the last element
            assert_eq!(
                skiplist.find_less_than(&20),
                Some((&12, &"mango".to_string()))
            );
        }
    }
}

#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use rand::RngCore;
    use std::iter;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    const NUM_KEYS: usize = 4;

    /**
    This test attempts to mimic the [concurrent test] used for the skip list in LevelDB.

    We want to make sure that with a single writer and multiple concurrent readers, the readers
    always observe all data that was present in the skip list when a reader was spawned. Because
    insertions happen concurrently, we may observe new values.

    Keys will be a tuple of (key: usize, generation: usize) where key will be in a range [0...K-1]
    and generation is some monotonically increasing number.

    Insertions will pick a random key and set the generation to 1 + the last generation number
    inserted for that key.

    At the beginning of a read, a snapshot of last inserted generation numbers for each key is
    taken. Read activities are then commenced. For every key encountered, we check that it is
    either expected given the initial snapshot or has been added since the reader started reading.

    [concurrent test]: https://github.com/google/leveldb/blob/e426c83e88c4babc785098d905c2dcb4f4e884af/db/skiplist_test.cc#L128
    */
    #[test]
    fn with_concurrent_threads_readers_see_correct_values() {}

    #[derive(Clone)]
    struct Snapshot {
        generations: [Arc<AtomicUsize>; NUM_KEYS],
    }

    impl Snapshot {
        /// Create a new instance of [`Snapshot`].
        fn new() -> Self {
            let generations: [Arc<AtomicUsize>; NUM_KEYS] =
                iter::repeat_with(|| Arc::new(AtomicUsize::new(0)))
                    .take(4)
                    .collect::<Vec<Arc<AtomicUsize>>>()
                    .try_into()
                    .unwrap();

            Self { generations }
        }

        /// Set the generation number of a key.
        fn set(&mut self, k: usize, generation: usize) {
            self.generations[k].store(generation, atomic::Ordering::Release);
        }

        /// Get the generation number of a key.
        fn get(&self, k: usize) -> usize {
            self.generations[k].load(atomic::Ordering::Acquire)
        }
    }

    /// The state of a reader thread.
    enum ReaderState {
        Started,
        Running,
        Done,
    }

    struct TestHarness {
        stop_flag: Arc<AtomicBool>,
        random_seed: u64,
        skiplist: Arc<ConcurrentSkipList<(usize, usize), String>>,
        current_snapshot: Snapshot,
    }

    impl TestHarness {
        fn new() -> Self {
            let stop_flag = Arc::new(AtomicBool::new(false));
            // 0x726f62696e => Robin, Batmann's real life companion :)
            let random_seed = 0x726f62696e;
            let skiplist = Arc::new(ConcurrentSkipList::<(usize, usize), String>::new(None));
            let current_snapshot = Snapshot::new();

            Self {
                stop_flag,
                random_seed,
                skiplist,
                current_snapshot,
            }
        }

        fn concurrent_reader(harness: Arc<TestHarness>, thread_seed: u64) {
            while !harness.stop_flag.load(atomic::Ordering::Acquire) {
                harness.read_step(thread_seed);
            }
        }

        fn read_step(&self, seed: u64) {
            let mut rng = StdRng::seed_from_u64(seed);

            // Remember the initial state of the skip list
            let mut snapshot = Snapshot::new();
            for key in 0..NUM_KEYS {
                snapshot.set(key, self.current_snapshot.get(key));
            }

            let zero_gen = 0.to_string();
            let mut start_read_range = TestHarness::get_random_start_position(&mut rng);
            let mut iterator = self.skiplist.iter().peekable();
            let mut end_of_read_range: &(usize, usize);

            // Move iterator to the start of the read range
            iterator.position(|(key, _)| key >= &start_read_range);

            loop {
                // Set end of read range to the last element if our starting position is at the last
                // element.
                end_of_read_range = self
                    .skiplist
                    .find_greater_or_equal(&start_read_range)
                    .or(Some((&(NUM_KEYS, 0), &zero_gen)))
                    .unwrap()
                    .0;

                assert!(
                    &start_read_range <= end_of_read_range,
                    "The end of the range cannot go backwards"
                );

                /*
                Verify that every thing from [start_read_range, end_of_read_range) was not
                present in the initial state. Generation number 0 is never inserted so the end of
                the range was the greatest at the point in time. Anything within the range was
                added by a concurrent writer.
                */
                while (&start_read_range < end_of_read_range) {}

                if iterator.peek().is_none() {
                    break;
                }
            }
        }

        fn write_step(&mut self, seed: u64) {
            let mut rng = StdRng::seed_from_u64(seed);
            let key: usize = (rng.next_u64() % (NUM_KEYS as u64)) as usize;
            let generation_number = self.current_snapshot.get(key) + 1;

            // SAFETY: Only one thread is writing
            unsafe {
                self.skiplist
                    .insert((key, generation_number), generation_number.to_string());
            }

            self.current_snapshot.set(key, generation_number);
        }

        fn get_random_start_position(rng: &mut StdRng) -> (usize, usize) {
            match rng.next_u32() % 10 {
                0 => {
                    // Start at the beginning
                    (0, 0)
                }
                1 => {
                    // Start at the end
                    (NUM_KEYS, 0)
                }
                _ => {
                    // Start somewhere in the middle
                    ((rng.next_u32() % 10) as usize, 0)
                }
            }
        }
    }
}
