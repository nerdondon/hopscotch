//! # Hopscotch - a skip list implemented in rust
//! What it says. Cuz it skips.

pub mod skiplist;

// Re-export the SkipList struct and show at the top level of docs
#[doc(inline)]
pub use crate::skiplist::SkipList;
