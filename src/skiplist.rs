use rand::thread_rng;
use rand_distr::{Distribution, Geometric};
use std::cell::RefCell;
use std::hash::Hash;
use std::rc::{Rc, Weak};

type Link<K, V> = Option<Rc<RefCell<SkipNode<K, V>>>>;

struct SkipNode<K: Ord + Hash, V: Clone> {
    /// The key should only be `None` for the `head` node.
    key: Option<K>,
    /// The Value should only be `None` for the `head` node.
    value: Option<V>,
    levels: Vec<Link<K, V>>,
}

impl<K: Ord + Hash, V: Clone> SkipNode<K, V> {
    fn new(key: K, value: V) -> Self {
        SkipNode {
            key: Some(key),
            value: Some(value),
            levels: vec![],
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

pub struct SkipList<K: Ord + Hash, V: Clone> {
    head: Link<K, V>,
    length: u64,
    /// The probability of success used in probability distribution for determining height of a new
    /// node. Defaults to 0.25.
    probability: f64,
    /// The cuurent maximum height of the skip list.
    height: u64,
}

// Public methods of SkipList
impl<K: Ord + Hash, V: Clone> SkipList<K, V> {
    /// Create a new skip list.
    ///
    /// `probability` is the probability of success used in probability distribution for determining
    /// height of a new node. Defaults to 0.25.
    pub fn new(probability: Option<f64>) -> Self {
        let head = SkipNode::head();
        SkipList {
            head: Some(Rc::new(RefCell::new(head))),
            length: 0,
            probability: probability.unwrap_or(0.25),
            height: 1,
        }
    }

    /// Get an immutable reference to the value corresponding to the `key`.
    ///
    /// Returns `Some(V)` if found and `None` if not
    pub fn get(&self, key: &K) -> Option<V> {
        let mut wrapped_current_node = self.head.as_ref().map(Rc::clone);

        // Start iteration at the top of the skip list "towers"
        for level_idx in (0..self.height()).rev() {
            // Iterate through pointers at the current level. If we skipped past our key, move down
            // a level.
            while wrapped_current_node.is_some() {
                // Move the pointer forward
                wrapped_current_node = wrapped_current_node.unwrap().borrow().levels[level_idx]
                    .as_ref()
                    .map(Rc::clone);

                // `.take()` here instead of `as_ref()` on `wrapped_current_node` because we do not
                // want to borrow a reference. If we borrow a reference to `wrapped_current_node`
                // we will not be able to re-assign a new value to it. `.take()` will create a new
                // `Option` from the existing one.
                let current_node = wrapped_current_node.take().unwrap();
                let borrowed_node = current_node.borrow();
                match borrowed_node.key.as_ref().unwrap().cmp(key) {
                    std::cmp::Ordering::Less => {
                        wrapped_current_node =
                            borrowed_node.levels[level_idx].as_ref().map(Rc::clone);
                    }
                    _ => break,
                }
            }
        }

        let potential_record = wrapped_current_node.unwrap().borrow().levels[0]
            .as_ref()
            .map(Rc::clone);

        match potential_record {
            Some(ref node) => node.borrow().value.as_ref().cloned(),
            None => None,
        }
    }

    /// Get a mutable reference to the value with `key` from within the skip list.
    ///
    /// Returns `Some(V)` if found and `None` if not
    pub fn get_mut(&self, key: &K) -> Option<V> {
        // TODO: Figure out a better way to do `get` and `get_mut` rather than having so much duplicate code.
        None
    }

    /// Insert a key-value pair.
    pub fn insert(&self, key: K, value: V) {}

    /// Remove a key-value pair.
    pub fn remove(&self, key: K, value: V) {}

    /// The number of elements in the skip list.
    pub fn len(&self) -> u64 {
        self.length
    }

    /// Returns true if the skip list does not hold any elements; otherwise false.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    }
}

// Private methods of SkipList
impl<K: Ord + Hash, V: Clone> SkipList<K, V> {
    fn randomHeight(&self) -> u64 {
        let mut rng = thread_rng();
        let distribution = Geometric::new(self.probability).unwrap();
        let sample = distribution.sample(&mut rng);

        if sample > &self.height + 3 {
            // Only increase the height by one if the number drawn is much more than the current
            // maximum height. This is just an arbitrary cap on the growth in height of the skip
            // list.
            &self.height + 1
        } else {
            sample
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
