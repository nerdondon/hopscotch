use rand::thread_rng;
use rand_distr::{Distribution, Geometric};
use std::cell::RefCell;
use std::convert::TryInto;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::rc::Rc;

type Link<K, V> = Option<Rc<RefCell<SkipNode<K, V>>>>;

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
}

#[derive(Debug)]
pub struct SkipList<K: Ord + Hash + Debug, V: Clone> {
    head: Link<K, V>,
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
        let head = SkipNode::head();
        let mut skiplist = SkipList {
            head: Some(Rc::new(RefCell::new(head))),
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
    pub fn get(&self, key: &K) -> Option<V> {
        if self.is_empty() {
            return None;
        }

        let mut wrapped_current_node = self.head.as_ref().map(Rc::clone);

        // Start iteration at the top of the skip list "towers" and iterate through pointers at the
        // current level. If we skipped past our key, move down a level.
        for level_idx in (0..self.height()).rev() {
            // Get an optional of the next node
            let mut maybe_next_node = wrapped_current_node.as_ref().unwrap().borrow().levels
                [level_idx]
                .as_ref()
                .map(Rc::clone);

            while maybe_next_node.is_some() {
                let current_node = maybe_next_node.unwrap();
                let borrowed_node = current_node.borrow();
                match borrowed_node.key.as_ref().unwrap().cmp(key) {
                    std::cmp::Ordering::Less => {
                        wrapped_current_node = Some(Rc::clone(&current_node));
                        maybe_next_node = borrowed_node.levels[level_idx].as_ref().map(Rc::clone);
                    }
                    _ => break,
                }
            }
        }

        // The while loop uses a less than comparator and stops at the node that is potentially just
        // prior to the node we are looking for. We need to move the pointer forward one time and
        // check we actually arrived at our node or if we hit the end of the levels without finding
        // anything.
        let potential_record = wrapped_current_node.unwrap().borrow().levels[0]
            .as_ref()
            .map(Rc::clone);

        if potential_record.is_some() {
            let borrowed_record = potential_record.as_ref().unwrap().borrow();
            if borrowed_record.key.as_ref().unwrap().eq(key) {
                return borrowed_record.value.as_ref().cloned();
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
    /// assert_eq!(some_value, "banana".to_string());
    /// ```
    pub fn insert(&mut self, key: K, value: V) {
        let new_node_height = self.random_height();
        if new_node_height > self.height() {
            self.adjust_head(new_node_height);
        }

        // Track the end of each level
        let mut nodes_to_update: Vec<Link<K, V>> = vec![None; self.height()];
        let mut wrapped_current_node = self.head.as_ref().map(Rc::clone);

        // Start iteration at the top of the skip list "towers" and find the insert position at each
        // level
        for level_idx in (0..self.height()).rev() {
            // Get an optional of the next node
            let mut maybe_next_node = wrapped_current_node.as_ref().unwrap().borrow().levels
                [level_idx]
                .as_ref()
                .map(Rc::clone);

            while maybe_next_node.is_some() {
                let current_node = maybe_next_node.unwrap();
                let borrowed_node = current_node.borrow();
                match borrowed_node.key.as_ref().unwrap().cmp(&key) {
                    std::cmp::Ordering::Less => {
                        wrapped_current_node = Some(Rc::clone(&current_node));
                        maybe_next_node = borrowed_node.levels[level_idx].as_ref().map(Rc::clone);
                    }
                    _ => break,
                }
            }

            // Keep track of the node we stopped at. This is either the node right before our new
            // node or end of the level if no lesser node was found.
            nodes_to_update[level_idx] = wrapped_current_node.as_ref().map(Rc::clone);
        }

        let new_node = Rc::new(RefCell::new(SkipNode::new(key, value, new_node_height)));
        for level_idx in (0..new_node_height).rev() {
            let previous_node = nodes_to_update[level_idx].as_ref().map(Rc::clone);

            // Set the next pointer for this level on the new node i.e. `previous_node`'s next
            // node at this level becomes `new_node`'s next node at this level
            new_node.borrow_mut().levels[level_idx] =
                previous_node.as_ref().unwrap().borrow_mut().levels[level_idx].take();

            // Set the next pointer of the previous node to the new node
            previous_node.unwrap().borrow_mut().levels[level_idx] = Some(Rc::clone(&new_node));
        }

        // Book keeping for size
        // The additional usage should be from the size of the new node and the size of references
        // to this new node. This is multiplied by 2 to approximate the storage in the `levels`
        // vector. `mem::size_of` and `mem::size_of_val` does not actually get the size of vectors
        // since vectors are allocated to the heap and only a pointer is stored in the field.
        self.approximate_mem_usage +=
            mem::size_of::<SkipNode<K, V>>() + (2 * mem::size_of::<Link<K, V>>() * new_node_height);
        self.inc_length();
    }

    /// Remove a key-value pair.
    ///
    /// Returns the value at the key if the key was in the map.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.is_empty() {
            return None;
        }

        // Track the end of each level
        let mut nodes_to_update: Vec<Link<K, V>> = vec![None; self.height()];
        let mut wrapped_current_node = self.head.as_ref().map(Rc::clone);

        // Start iteration at the top of the skip list "towers" and find the removal position at each
        // level
        for level_idx in (0..self.height()).rev() {
            // Get an optional of the next node
            let mut maybe_next_node = wrapped_current_node.as_ref().unwrap().borrow().levels
                [level_idx]
                .as_ref()
                .map(Rc::clone);

            while maybe_next_node.is_some() {
                let current_node = maybe_next_node.unwrap();
                let borrowed_node = current_node.borrow();
                match borrowed_node.key.as_ref().unwrap().cmp(key) {
                    std::cmp::Ordering::Less => {
                        wrapped_current_node = Some(Rc::clone(&current_node));
                        maybe_next_node = borrowed_node.levels[level_idx].as_ref().map(Rc::clone);
                    }
                    _ => break,
                }
            }

            // Keep track of the node we stopped at. This is either the node right before our new
            // node or end of the level if no lesser node was found.
            nodes_to_update[level_idx] = wrapped_current_node.as_ref().map(Rc::clone);
        }

        // Our comparator uses a less than condition so the last node we stopped at might be just in
        // front of the node we are looking to remove
        let maybe_found_node = wrapped_current_node.unwrap().borrow().levels[0]
            .as_ref()
            .map(Rc::clone)
            .unwrap();
        if maybe_found_node.borrow().key.as_ref().unwrap().ne(key) {
            // No-op if we didn't find the key in the skip list
            return None;
        }

        let mut num_nodes_adjusted: usize = 0;
        for level_idx in (0..self.height()).rev() {
            let previous_node = nodes_to_update[level_idx].as_ref().map(Rc::clone).unwrap();
            let maybe_next_node = previous_node.borrow().levels[level_idx]
                .as_ref()
                .map(Rc::clone);

            // If the next pointer of the node we ended at on this level is the node for the search
            // key, adjust the next pointer to point to the node after the node we are removing.
            if maybe_next_node.is_some() {
                let mut borrowed_next_node = maybe_next_node.as_ref().unwrap().borrow_mut();
                if borrowed_next_node.key.as_ref().unwrap().eq(key) {
                    previous_node.borrow_mut().levels[level_idx] =
                        borrowed_next_node.levels[level_idx].take();
                    num_nodes_adjusted += 1;
                }
            }
        }

        // Book keeping for size
        // See [`Skiplist::insert`] for reasoning behind the approximate usage removed.
        self.approximate_mem_usage -= mem::size_of::<SkipNode<K, V>>()
            + (2 * mem::size_of::<Link<K, V>>() * num_nodes_adjusted);
        self.dec_length();

        let mutable_found_node = maybe_found_node.borrow_mut();
        Some(mutable_found_node.value.as_ref().cloned().unwrap())
    }

    /// The number of elements in the skip list.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the skip list does not hold any elements; otherwise false.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Print out the keys of elements in the skip list.
    pub fn print_keys(&self) {
        let mut node = self.head.as_ref().map(Rc::clone);
        while node.is_some() {
            let unwrapped_node = node.unwrap();
            let borrowed_node = unwrapped_node.borrow();
            let wrapped_key = borrowed_node.key.as_ref();
            if wrapped_key.is_some() {
                println!("Key: {:?}", wrapped_key.unwrap());
            } else {
                println!("Head");
            }

            node = borrowed_node.levels[0].as_ref().map(Rc::clone);
        }
    }

    /// Get the approximate amount of memory used in number of bytes.
    pub fn get_approx_mem_usage(&self) -> usize {
        self.approximate_mem_usage
    }
}

/// Implementation for keys and values that implement `Clone`
impl<K, V> SkipList<K, V>
where
    K: Ord + Hash + Debug + Clone,
    V: Clone,
{
    /// Returns the entries stored in the skip list as `Vec<(K,V)>` with cloned values.
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
        for node in self.iter() {
            let borrowed_node = node.borrow();
            let cloned_key = borrowed_node.key.as_ref().cloned().unwrap();
            let cloned_value = borrowed_node.value.as_ref().cloned().unwrap();
            kv_pairs.push((cloned_key, cloned_value));
        }

        kv_pairs
    }
}

// Private methods of SkipList
impl<K: Ord + Hash + Debug, V: Clone> SkipList<K, V> {
    /// The current maximum height of the skip list.
    fn height(&self) -> usize {
        self.head.as_ref().unwrap().borrow().levels.len()
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
        let mut head_node = self.head.as_ref().unwrap().borrow_mut();
        for _ in 0..height_difference {
            head_node.levels.push(None);
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

    /// An iterator visiting each node
    fn iter(&self) -> Iter<K, V> {
        if self.is_empty() {
            return Iter { next: None };
        }

        Iter {
            next: self.head.as_ref().unwrap().borrow().levels[0]
                .as_ref()
                .map(Rc::clone),
        }
    }
}

/// An iterator over the entries of a `SkipList`.
///
/// This `struct` is created by the [`iter`] method.
///
/// [`iter`]: SkipList::iter
struct Iter<K, V>
where
    K: Ord + Hash + Debug,
    V: Clone,
{
    next: Option<Rc<RefCell<SkipNode<K, V>>>>,
}

impl<K, V> Iterator for Iter<K, V>
where
    K: Ord + Hash + Debug,
    V: Clone,
{
    type Item = Rc<RefCell<SkipNode<K, V>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let wrapped_current_node = self.next.as_ref();

        // Short-circuit return `None`
        wrapped_current_node?;

        let current_node = wrapped_current_node.map(Rc::clone).unwrap();
        self.next = current_node.borrow().levels[0].as_ref().map(Rc::clone);
        Some(current_node)
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

        assert_eq!(expected_value, actual_value);
    }

    #[test]
    fn get_an_element_in_the_middle() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(1, "apple".to_string());
        skiplist.insert(3, "orange".to_string());
        let expected_value = "banana".to_string();

        let actual_value = skiplist.get(&2).unwrap();

        assert_eq!(expected_value, actual_value);
    }

    #[test]
    fn get_an_element_at_the_tail() {
        let mut skiplist = SkipList::<i32, String>::new(None);
        skiplist.insert(2, "banana".to_string());
        skiplist.insert(3, "orange".to_string());
        skiplist.insert(1, "apple".to_string());
        let expected_value = "orange".to_string();

        let actual_value = skiplist.get(&3).unwrap();

        assert_eq!(expected_value, actual_value);
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
    fn collect_returns_a_vec_with_the_key_value_pairs_of_elements() {
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
}
