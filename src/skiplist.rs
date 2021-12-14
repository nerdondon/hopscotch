use rand::thread_rng;
use rand_distr::{Distribution, Geometric};
use std::convert::TryInto;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::FusedIterator;
use std::mem;
use std::ptr::NonNull;

type Link<K, V> = Option<NonNull<SkipNode<K, V>>>;

#[derive(Debug)]
struct SkipNode<K: Ord + Hash + Debug, V: Clone> {
    /// The key should only be `None` for the `head` node.
    key: Option<K>,
    /// The Value should only be `None` for the `head` node.
    value: Option<V>,
    levels: Vec<Link<K, V>>,
}

impl<K: Ord + Hash + Debug, V: Clone> SkipNode<K, V> {
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

        self.levels[level].as_ref().map(|node_ptr| {
            unsafe {
                /*
                SAFETY:
                This is safe because links are guaranteed to exist. If the link did not exist, it
                would be a `None` in the tower and execution would not have gotten here.
                */
                node_ptr.as_ref()
            }
        })
    }
}

/// A skip list.
///
/// # Safety
///
/// Invariants:
///
/// - If an Link exists it must be valid to dereference to a `SkipNode`.
#[derive(Debug)]
pub struct SkipList<K: Ord + Hash + Debug, V: Clone> {
    head: Box<SkipNode<K, V>>,
    length: usize,
    /// The probability of success used in probability distribution for determining height of a new
    /// node. Defaults to 0.25.
    probability: f64,
    /// Approximate memory used by the skip list in number of bytes.
    approximate_mem_usage: usize,
}

// Public methods of SkipList
impl<K: Ord + Hash + Debug, V: Clone> SkipList<K, V> {
    /// Create a new skip list.
    ///
    /// `probability` is the probability of success used in probability distribution for determining
    /// height of a new node. Defaults to 0.25.
    ///
    /// # Examples
    /// ```
    /// use nerdondon_hopscotch::skiplist::SkipList;
    ///
    /// let skiplist = SkipList::<i32, String>::new(None);
    /// ```
    pub fn new(probability: Option<f64>) -> Self {
        let head = Box::new(SkipNode::head());
        let mut skiplist = SkipList {
            head,
            /// The number of elements in the skip list.
            length: 0,
            probability: probability.unwrap_or(0.25),
            approximate_mem_usage: 0,
        };

        // TODO: Make size tracking a feature?
        let size = mem::size_of_val(&skiplist) + mem::size_of::<SkipNode<K, V>>();
        skiplist.approximate_mem_usage = size;

        skiplist
    }

    /// Get an immutable reference to the value corresponding to the specified `key`.
    ///
    /// Returns `Some(V)` if found and `None` if not
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.is_empty() {
            return None;
        }

        let mut current_node = &*self.head;

        // Start iteration at the top of the skip list "towers" and iterate through pointers at the
        // current level. If we skipped past our key, move down a level.
        for level_idx in (0..self.height()).rev() {
            // Get an optional of the next node
            let mut maybe_next_node = current_node.levels[level_idx].as_ref();

            while maybe_next_node.is_some() {
                let next_node_ptr = maybe_next_node.unwrap();
                let next_node = unsafe {
                    // SAFETY: next_node_ptr is guaranteed to exist by the condition for the `while`
                    next_node_ptr.as_ref()
                };
                match next_node.key.as_ref().unwrap().cmp(key) {
                    std::cmp::Ordering::Less => {
                        current_node = next_node;
                        maybe_next_node = next_node.levels[level_idx].as_ref();
                    }
                    _ => break,
                }
            }
        }

        // The while loop uses a less than comparator and stops at the node that is potentially just
        // prior to the node we are looking for. We need to move the pointer forward one time and
        // check we actually arrived at our node or if we hit the end of the levels without finding
        // anything.
        let potential_record = current_node.levels[0].as_ref();
        if potential_record.is_some() {
            let record = unsafe {
                /*
                SAFETY:
                The `is_some` check ensures that the pointer exists and globally all links are
                guaranteed to be valid.
                */
                potential_record.as_ref().unwrap().as_ref()
            };

            if record.key.as_ref().unwrap().eq(key) {
                return record.value.as_ref();
            }
        }

        None
    }

    /// Insert a key-value pair.
    ///
    /// # Examples
    /// ```
    /// use nerdondon_hopscotch::skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::<i32, String>::new(None);
    /// skiplist.insert(2, "banana".to_string());
    /// skiplist.insert(3, "orange".to_string());
    /// skiplist.insert(1, "apple".to_string());
    ///
    /// let some_value = skiplist.get(&2).unwrap();
    /// assert_eq!(some_value, "banana");
    /// ```
    pub fn insert(&mut self, key: K, value: V) {
        let new_node_height = self.random_height();
        if new_node_height > self.height() {
            self.adjust_head(new_node_height);
        }

        // Track where we end on each level
        let mut nodes_to_update: Vec<Option<NonNull<SkipNode<K, V>>>> = vec![None; self.height()];
        let list_height = self.height();
        let mut current_node = self.head.as_ref();

        // Start iteration at the top of the skip list "towers" and find the insert position at each
        // level
        for level_idx in (0..list_height).rev() {
            // Get an optional of the next node
            let mut maybe_next_node = current_node.levels[level_idx].as_ref();

            while maybe_next_node.is_some() {
                let next_node_ptr = maybe_next_node.unwrap();
                let next_node = unsafe {
                    // SAFETY: next_node_ptr is guaranteed to exist by the condition for the `while`
                    next_node_ptr.as_ref()
                };
                match next_node.key.as_ref().unwrap().cmp(&key) {
                    std::cmp::Ordering::Less => {
                        current_node = next_node;
                        maybe_next_node = next_node.levels[level_idx].as_ref();
                    }
                    _ => break,
                }
            }

            // Keep track of the node we stopped at. This is either the node right before our new
            // node or head node of the level if no lesser node was found.
            nodes_to_update[level_idx] = Some(current_node.into());
        }

        let mut new_node = Box::new(SkipNode::new(key, value, new_node_height));
        /*
        `.unwrap` is called explicity after the `NonNull::new` because `None` is produced on
        failure and we want to be explicit about the existence of the value stored in the
        levels vector.
        */
        let new_node_ptr = NonNull::new(new_node.as_mut()).unwrap();
        for level_idx in (0..new_node_height).rev() {
            let previous_node = unsafe {
                /*
                SAFETY:
                `nodes_to_update` is populated above with `current_node`, which is checked for
                existence.
                */
                nodes_to_update[level_idx].as_mut().unwrap().as_mut()
            };

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
        self.approximate_mem_usage +=
            mem::size_of::<SkipNode<K, V>>() + (2 * mem::size_of::<Link<K, V>>() * new_node_height);
        self.inc_length();

        /*
        `Box::leak` is called so that the node does not get deallocated at the end of the function.
        The `SkipList::remove` method will ensure to reform the box from the pointer so that the
        node is de-allocated on removal.
        */
        Box::leak(new_node);
    }

    /// Remove a key-value pair.
    ///
    /// Returns the value at the key if the key was in the map.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.is_empty() {
            return None;
        }

        // Track where we end on each level
        let mut nodes_to_update: Vec<Option<NonNull<SkipNode<K, V>>>> = vec![None; self.height()];
        let list_height = self.height();
        let mut current_node = self.head.as_ref();

        // Start iteration at the top of the skip list "towers" and find the removal position at each
        // level
        for level_idx in (0..list_height).rev() {
            // Get an optional of the next node
            let mut maybe_next_node = current_node.levels[level_idx].as_ref();

            while maybe_next_node.is_some() {
                let next_node_ptr = maybe_next_node.unwrap();
                let next_node = unsafe {
                    // SAFETY: next_node_ptr is guaranteed to exist by the condition for the `while`
                    next_node_ptr.as_ref()
                };
                match next_node.key.as_ref().unwrap().cmp(key) {
                    std::cmp::Ordering::Less => {
                        current_node = next_node;
                        maybe_next_node = next_node.levels[level_idx].as_ref();
                    }
                    _ => break,
                }
            }

            // Keep track of the node we stopped at. This is either the node right before our new
            // node or head node of the level if no lesser node was found.
            nodes_to_update[level_idx] = Some(current_node.into());
        }

        // Our comparator uses a less than condition so the last node we stopped at might be just in
        // front of the node we are looking to remove
        let found_node_ptr = *current_node.levels[0].as_ref().unwrap();
        let found_node = unsafe {
            // SAFETY: All links are guaranteed to be valid
            found_node_ptr.as_ref()
        };
        if found_node.key.as_ref().unwrap().ne(key) {
            // No-op if we didn't find the key in the skip list
            return None;
        }

        let mut num_nodes_adjusted: usize = 0;
        for level_idx in (0..self.height()).rev() {
            let previous_node = unsafe {
                /*
                SAFETY:
                `nodes_to_update` is populated above with `current_node_ptr` that are checked for
                existence.
                */
                nodes_to_update[level_idx].take().unwrap().as_mut()
            };
            let maybe_next_node = previous_node.levels[level_idx].as_mut();

            // If the next pointer of the node we ended at on this level is the node for the search
            // key, adjust the next pointer to point to the node after the node we are removing.
            if let Some(next_node_ptr) = maybe_next_node {
                // SAFETY: Links are guaranteed to exist if not `None`
                let next_node = unsafe { next_node_ptr.as_mut() };
                if next_node.key.as_ref().unwrap().eq(key) {
                    previous_node.levels[level_idx] = next_node.levels[level_idx].take();
                    num_nodes_adjusted += 1;
                }
            }
        }

        // Book keeping for size
        // See [`Skiplist::insert`] for reasoning behind the approximate usage removed.
        self.approximate_mem_usage -= mem::size_of::<SkipNode<K, V>>()
            + (2 * mem::size_of::<Link<K, V>>() * num_nodes_adjusted);
        self.dec_length();

        // Re-box the allocation the pointer represents so that it can get dropped.
        // Strategy from std::linked_list: https://github.com/rust-lang/rust/blob/6f40fa4353a9075288f74ecc3553010b34c65baa/library/alloc/src/collections/linked_list.rs#L186
        let boxed_found_node = unsafe {
            /*
            SAFETY:
            The other references to this pointer were removed when removing the links above, so
            `found_node_ptr` should be the last reference to this node. All links should be valid so
            `found_node_ptr`, which is a link at level 0, should be valid.
            */
            Box::from_raw(found_node_ptr.as_ptr())
        };

        Some(boxed_found_node.value.unwrap())
    }

    /// The number of elements in the skip list.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the skip list does not hold any elements; otherwise false.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the approximate amount of memory used in number of bytes.
    pub fn get_approx_mem_usage(&self) -> usize {
        self.approximate_mem_usage
    }

    /// An iterator visiting each node in order.
    ///
    /// Returns values of (&'a K, &'a V)
    pub fn iter(&self) -> NodeIterHelper<'_, K, V> {
        if self.is_empty() {
            return NodeIterHelper { next: None };
        }

        let next = self.head.levels[0].as_ref().map(|node_ptr| unsafe {
            // SAFETY: All links are valid if they exist.
            node_ptr.as_ref()
        });

        NodeIterHelper { next }
    }

    /// An iterator visiting each node in order with mutable references to values.
    ///
    /// Returns values of (&'a K, &'a mut V)
    pub fn iter_mut(&mut self) -> NodeIterMutHelper<'_, K, V> {
        if self.is_empty() {
            return NodeIterMutHelper { next: None };
        }

        let next = self.head.levels[0].as_mut().map(|node_ptr| unsafe {
            // SAFETY: All links are valid if they exist.
            node_ptr.as_mut()
        });

        NodeIterMutHelper { next }
    }

    /// Return a reference to the key and value of the first node in the skip list if there is a
    /// node. Otherwise, it returns `None`.
    pub fn first(&self) -> Option<(&K, &V)> {
        self.head
            .next()
            .map(|node| (node.key.as_ref().unwrap(), node.value.as_ref().unwrap()))
    }

    /// Return a reference to the key and value of the last node in the skip list if there is a
    /// node. Otherwise, it returns `None`.
    pub fn last(&self) -> Option<(&K, &V)> {
        if self.is_empty() {
            return None;
        }

        let mut current_node = &*self.head;
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
}

/// Implementation for keys and values that implement `Clone`
impl<K, V> SkipList<K, V>
where
    K: Ord + Hash + Debug + Clone,
    V: Clone,
{
    /// Eagerly returns the entries stored in the skip list as `Vec<(K,V)>` with cloned values.
    ///
    /// # Examples
    /// ```
    /// use nerdondon_hopscotch::skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::<i32, String>::new(None);
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
impl<K: Ord + Hash + Debug, V: Clone> SkipList<K, V> {
    /// The current maximum height of the skip list.
    fn height(&self) -> usize {
        self.head.levels.len()
    }

    /// Generates a random height according to a geometric distribution.
    fn random_height(&self) -> usize {
        let mut rng = thread_rng();
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
    fn adjust_head(&mut self, new_height: usize) {
        if self.height() >= new_height {
            return;
        }

        let height_difference = new_height - self.height();
        for _ in 0..height_difference {
            self.head.levels.push(None);
        }
    }

    /// Increment length by 1.
    fn inc_length(&mut self) {
        self.length += 1;
    }

    /// Decrement length by 1.
    fn dec_length(&mut self) {
        self.length -= 1;
    }

    /// Return a reference to the first node with a key that is greater than or equal to the target
    /// key.
    fn find_greater_or_equal_node(&self, target: &K) -> Option<&SkipNode<K, V>> {
        if self.is_empty() {
            return None;
        }

        let mut current_node = &*self.head;
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
                    std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => {
                        if level_idx == 0 {
                            // We are at the bottom of the tower, so this is closest node greater
                            // than or equal to our target.
                            return Some(next_node);
                        }

                        // We found a node greater than or equal to our target. See if this is the
                        // the first greatest node after our target by breaking and moving one
                        // level down in the tower.
                        break;
                    }
                }
            }
        }

        // This is reached when the target is greater than all of the nodes in the skip list.
        None
    }

    /// Return a reference to the last node with a key that is less than the target key.
    fn find_less_than_node(&self, target: &K) -> Option<&SkipNode<K, V>> {
        if self.is_empty() {
            return None;
        }

        let mut current_node = &*self.head;
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
}

/// An iterator adapter over the nodes of a `SkipList`.
///
/// This `struct` is created by the [`iter`] method.
///
/// [`iter`]: SkipList::iter
pub struct NodeIterHelper<'a, K, V>
where
    K: Ord + Hash + Debug,
    V: Clone,
{
    next: Option<&'a SkipNode<K, V>>,
}

impl<'a, K, V> Iterator for NodeIterHelper<'a, K, V>
where
    K: Ord + Hash + Debug,
    V: Clone,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let wrapped_current_node = self.next;

        // Short-circuit return `None`
        wrapped_current_node?;

        let current_node = wrapped_current_node.unwrap();
        self.next = unsafe {
            /*
            SAFETY:
            Links at level 0 are always valid. No mutations can happen because the only way to get a
            `NodeIterator` is via [`SkipList::iter`] which borrows an immutable reference.
            */
            current_node.levels[0]
                .as_ref()
                .map(|node_ptr| node_ptr.as_ref())
        };

        Some((
            current_node.key.as_ref().unwrap(),
            current_node.value.as_ref().unwrap(),
        ))
    }
}

impl<'a, K, V> IntoIterator for &'a SkipList<K, V>
where
    K: Ord + Hash + Debug,
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
    K: Ord + Hash + Debug,
    V: Clone,
{
}

/// An mutable iterator adapter over the nodes of a `SkipList`.
///
/// This `struct` is created by the [`iter_mut`] method.
///
/// [`iter_mut`]: SkipList::iter_mut
pub struct NodeIterMutHelper<'a, K, V>
where
    K: Ord + Hash + Debug,
    V: Clone,
{
    next: Option<&'a mut SkipNode<K, V>>,
}

impl<'a, K, V> Iterator for NodeIterMutHelper<'a, K, V>
where
    K: Ord + Hash + Debug,
    V: Clone,
{
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        match self.next.take() {
            None => None,
            Some(current_node) => {
                self.next = unsafe {
                    /*
                    SAFETY:
                    Links at level 0 are always valid. No mutations can happen because the only way to get a
                    `NodeIterator` is via [`SkipList::iter`] which borrows an immutable reference.
                    */
                    current_node.levels[0]
                        .as_mut()
                        .map(|node_ptr| node_ptr.as_mut())
                };

                return Some((
                    current_node.key.as_ref().unwrap(),
                    current_node.value.as_mut().unwrap(),
                ));
            }
        }
    }
}

impl<K, V> Drop for SkipList<K, V>
where
    K: Ord + Hash + Debug,
    V: Clone,
{
    fn drop(&mut self) {
        if self.is_empty() {
            return;
        }

        let mut maybe_node_ptr = self.head.as_mut().levels[0];

        while maybe_node_ptr.is_some() {
            let mut current_node_ptr = maybe_node_ptr.unwrap();

            /*
            Re-box the allocation the pointer represents so that it can get dropped. Insert's
            leak the boxed node when it is created.

            It is ok to leave pointers in the dropped node's levels vector dangling because all
            nodes are getting dropped.
            */
            let current_node = unsafe {
                // SAFETY: All links are guaranteed to be valid nodes.
                Box::from_raw(current_node_ptr.as_mut())
            };

            maybe_node_ptr = current_node.levels[0];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn with_an_empty_skiplist_get_returns_none() {
        let skiplist = SkipList::<i32, String>::new(None);
        assert_eq!(skiplist.get(&10), None);
    }

    #[test]
    fn with_an_empty_skiplist_is_empty_returns_true() {
        let skiplist = SkipList::<i32, String>::new(None);
        assert_eq!(skiplist.is_empty(), true);
    }

    #[test]
    fn with_an_empty_skiplist_len_returns_zero() {
        let skiplist = SkipList::<i32, String>::new(None);
        assert_eq!(skiplist.len(), 0);
    }

    #[test]
    fn with_an_empty_skiplist_insert_can_add_an_element() {
        let mut skiplist = SkipList::<i32, String>::new(None);

        skiplist.insert(1, "apple".to_string());

        assert_eq!(skiplist.len(), 1);
    }

    #[test]
    fn insert_can_add_an_element_after_an_existing_element() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(1, "apple".to_string());

        skiplist.insert(2, "banana".to_string());

        assert_eq!(skiplist.len(), 2);
    }

    #[test]
    fn insert_can_add_an_element_before_an_existing_element() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());

        skiplist.insert(1, "apple".to_string());

        assert_eq!(skiplist.len(), 2);
    }

    #[test]
    fn insert_can_add_an_element_between_existing_elements() {
        // TODO: Mock distribution
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());

        skiplist.insert(2, "banana".to_string());

        assert_eq!(skiplist.len(), 3);
    }

    #[test]
    fn get_an_element_at_the_head() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(1, "apple".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(2, "banana".to_string());
        let expected_value = "apple".to_string();

        let actual_value = skiplist.get(&1).unwrap();

        assert_eq!(&expected_value, actual_value);
    }

    #[test]
    fn get_an_element_in_the_middle() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(1, "apple".to_string());
        skiplist.insert(3, "orange".to_string());
        let expected_value = "banana".to_string();

        let actual_value = skiplist.get(&2).unwrap();

        assert_eq!(&expected_value, actual_value);
    }

    #[test]
    fn get_an_element_at_the_tail() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());
        let expected_value = "orange".to_string();

        let actual_value = skiplist.get(&3).unwrap();

        assert_eq!(&expected_value, actual_value);
    }

    #[test]
    fn with_a_non_empty_skiplist_getting_a_non_existent_element_returns_none() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());

        let actual_value = skiplist.get(&0);

        assert_eq!(None, actual_value);
    }

    #[test]
    fn with_a_non_empty_skiplist_is_empty_returns_false() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());

        assert_eq!(skiplist.is_empty(), false);
    }

    #[test]
    fn with_an_empty_skiplist_collect_returns_an_empty_vec() {
        let skiplist = SkipList::<i32, String>::new(None);

        let actual_value = skiplist.entries();

        assert_eq!(actual_value.len(), 0);
        assert_eq!(actual_value, []);
    }

    #[test]
    fn entries_returns_a_vec_with_the_key_value_pairs_of_elements() {
        let mut skiplist = SkipList::<i32, String>::new(None);
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

    #[test]
    fn remove_can_remove_an_item_from_the_front_of_the_skip_list() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());

        let removed_value = skiplist.remove(&1);

        assert_eq!(skiplist.len(), 2);
        assert_eq!(removed_value.unwrap(), "apple".to_string());
        assert_eq!(skiplist.get(&1), None);
        assert_eq!(
            skiplist.entries(),
            [(2, "banana".to_string()), (3, "orange".to_string())]
        );
    }

    #[test]
    fn remove_can_remove_an_item_from_the_middle_of_the_skip_list() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());

        let removed_value = skiplist.remove(&2);

        assert_eq!(skiplist.len(), 2);
        assert_eq!(removed_value.unwrap(), "banana".to_string());
        assert_eq!(skiplist.get(&2), None);
        assert_eq!(
            skiplist.entries(),
            [(1, "apple".to_string()), (3, "orange".to_string())]
        );
    }

    #[test]
    fn remove_can_remove_an_item_from_the_back_of_the_skip_list() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());

        let removed_value = skiplist.remove(&3);

        assert_eq!(skiplist.len(), 2);
        assert_eq!(removed_value.unwrap(), "orange".to_string());
        assert_eq!(skiplist.get(&3), None);
        assert_eq!(
            skiplist.entries(),
            [(1, "apple".to_string()), (2, "banana".to_string())]
        );
    }

    #[test]
    fn remove_can_remove_all_elements_from_the_skip_list() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());

        skiplist.remove(&3);
        skiplist.remove(&1);
        skiplist.remove(&2);

        assert_eq!(skiplist.len(), 0);
        assert_eq!(skiplist.entries(), []);
    }

    #[test]
    fn with_an_empty_skiplist_remove_does_nothing() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        assert_eq!(skiplist.is_empty(), true);

        let removed_value = skiplist.remove(&30);

        assert_eq!(skiplist.is_empty(), true);
        assert_eq!(removed_value, None);
    }

    #[test]
    fn get_approx_mem_usage_provides_decent_estimates() {
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

        let mut skiplist = SkipList::<u16, String>::new(None);
        assert!(skiplist.get_approx_mem_usage() >= usage_approximation);

        skiplist.insert(1, "apple".to_string());
        usage_approximation += base_node_usage + 7 + skiplist.height() * link_size;
        assert!(skiplist.get_approx_mem_usage() > usage_approximation);

        skiplist.insert(2, "banana".to_string());
        usage_approximation += base_node_usage + 8 + skiplist.height() * link_size;
        assert!(skiplist.get_approx_mem_usage() > usage_approximation);

        skiplist.remove(&1);
        usage_approximation -= base_node_usage + 7 + skiplist.height() * link_size;
        assert!(skiplist.get_approx_mem_usage() > usage_approximation);

        skiplist.remove(&2);
        usage_approximation -= base_node_usage + 8 + skiplist.height() * link_size;
        assert!(skiplist.get_approx_mem_usage() >= usage_approximation);
    }

    #[test]
    fn with_a_non_empty_skiplist_first_returns_references_to_the_first_key_value_pair() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());
        skiplist.insert(4, "strawberry".to_string());
        skiplist.insert(5, "watermelon".to_string());

        assert_eq!(skiplist.first(), Some((&1, &"apple".to_string())));
    }

    #[test]
    fn with_an_empty_skiplist_first_returns_none() {
        let skiplist = SkipList::<i32, String>::new(None);

        assert_eq!(skiplist.first(), None);
    }

    #[test]
    fn with_a_non_empty_skiplist_last_returns_references_to_the_last_key_value_pair() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());
        skiplist.insert(4, "strawberry".to_string());
        skiplist.insert(5, "watermelon".to_string());

        assert_eq!(skiplist.last(), Some((&5, &"watermelon".to_string())));
    }

    #[test]
    fn with_an_empty_skiplist_last_returns_none() {
        let skiplist = SkipList::<i32, String>::new(None);

        assert_eq!(skiplist.last(), None);
    }

    #[test]
    fn with_a_non_empty_skiplist_find_greater_or_equal_returns_correct_responses() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());
        skiplist.insert(4, "strawberry".to_string());
        skiplist.insert(5, "watermelon".to_string());
        skiplist.insert(11, "grapefruit".to_string());
        skiplist.insert(12, "mango".to_string());

        assert_eq!(
            skiplist.find_greater_or_equal(&1),
            Some((&1, &"apple".to_string())),
            "The target is the first element so it should be found"
        );

        assert_eq!(
            skiplist.find_greater_or_equal(&3),
            Some((&3, &"orange".to_string())),
            "The middle element exists so it should be found"
        );

        assert_eq!(
            skiplist.find_greater_or_equal(&7),
            Some((&11, &"grapefruit".to_string())),
            "The target does not exist but there is a greater node so it should return that node"
        );

        assert_eq!(
            skiplist.find_greater_or_equal(&12),
            Some((&12, &"mango".to_string())),
            "THe last element exists so it should be found"
        );

        assert_eq!(
            skiplist.find_greater_or_equal(&20),
            None,
            "The target is greater than every element in the list so `None` should be returned"
        );
    }

    #[test]
    fn with_a_non_empty_skiplist_find_less_than_returns_correct_responses() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());
        skiplist.insert(4, "strawberry".to_string());
        skiplist.insert(5, "watermelon".to_string());
        skiplist.insert(11, "grapefruit".to_string());
        skiplist.insert(12, "mango".to_string());

        assert_eq!(
            skiplist.find_less_than(&0),
            None,
            "Finding a target less than every element in the list returns `None`"
        );

        assert_eq!(
            skiplist.find_less_than(&1),
            None,
            "Finding a target less than the first element returns `None`"
        );

        assert_eq!(
            skiplist.find_less_than(&3),
            Some((&2, &"banana".to_string())),
            "Finding a target less than an existing middle element"
        );

        assert_eq!(
            skiplist.find_less_than(&7),
            Some((&5, &"watermelon".to_string())),
            "Finding a target less than a non-existent middle element"
        );

        assert_eq!(
            skiplist.find_less_than(&20),
            Some((&12, &"mango".to_string())),
            "Finding a target greater than all elements returns the last element"
        );
    }
}
