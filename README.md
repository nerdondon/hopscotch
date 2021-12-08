# Hopscotch - Skip list implemented in Rust

What it says. Cuz it skips.

[![Build Status](https://github.com/nerdondon/hopscotch/actions/workflows/ci.yaml/badge.svg)](https://github.com/nerdondon/hopscotch/actions/workflows/ci.yaml)
[![Crates.io](https://img.shields.io/crates/v/nerdondon-hopscotch?color=orange)](https://crates.io/crates/nerdondon-hopscotch)

## Motivation

Note this is a toy project and is not meant for production usage...yet(?). It's primary purpose will
be as part of a database internals teaching project.

## Details

The implementation of the algorithms in v1 adheres somewhat faithfully to the algorithms as laid out
in the original paper by Pugh.

Uses a geometric distribution for determining if a new key is part of a level (fancy for saying we
flip a coin). The geometric distrubution actually defaults to p = 0.25 but this is configurable.

### Concurrency

A version of the skip list that allows for lock-free concurrent reads is now available by turning on
the `concurrent` feature. This skip list has a couple major feature gaps:

1. Callers must get a lock (e.g. `Mutex` or `RwLock`) over the skip list before insertions can be
   done.

1. Delete has not been implemented yet because my use case does not require delete.

## Other art

- [subway](https://github.com/sushrut141/subway)
- [rust-skiplist](https://github.com/JP-Ellis/rust-skiplist)
- [crossbeam-skiplist](https://github.com/crossbeam-rs/crossbeam/tree/master/crossbeam-skiplist)

## References

- [Blog - Notes and References on Skip Lists](https://blog.nerdondon.com/skip-list/)
- [OpenDSA - 15.1 Skip Lists](https://opendsa-server.cs.vt.edu/OpenDSA/Books/CS3/html/SkipList.html)
- [Learn Rust With Entirely Too Many Linked Lists](https://rust-unofficial.github.io/too-many-lists/)
- [Advanced Lifetimes](http://web.mit.edu/rust-lang_v1.25/arch/amd64_ubuntu1404/share/doc/rust/html/book/second-edition/ch19-02-advanced-lifetimes.html)
- [std::linked_list](https://github.com/rust-lang/rust/blob/master/library/alloc/src/collections/linked_list.rs)
- Great reference on creating an iterator in Rust:
  [Creating an Iterator in Rust](https://aloso.github.io/2021/03/09/creating-an-iterator)
