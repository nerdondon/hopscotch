//! # Hopscotch - a skip list implemented in rust
//! What it says. Cuz it skips.
//!
//! Features:
//!
//! `concurrent` - Enables the concurrent skiplist

pub mod skiplist;
// Re-export the SkipList struct and show at the top level of docs
#[doc(inline)]
pub use crate::skiplist::SkipList;

mod size;
#[doc(inline)]
pub use crate::size::Sizeable;

#[cfg(feature = "concurrent")]
pub mod concurrent_skiplist;
