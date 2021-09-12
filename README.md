# Hopscotch - Skip list implemented in Rust

What it says. Cuz it skips.

## Details

Uses a geometric distribution for determining if a new key is part of a level (fancy for saying we
flip a coin). The geometric distrubution actually defaults to p = 0.25 but this is configurable.

### Concurrency

The first version of this will be a non-concurrent version, requiring an external lock for
concurrent writes. The hope is to iteratively add concurrency features with `Arc`/`RwLock` first and
then lock-free methods following.

## TODO's and Considerations

- Use archery package to parameterize `Rc` vs `Arc` usage
- The non-concurrent version can be done with the normal `Rc`/`RefCell` construction or using
  `unsafe`/raw pointers much like the standard library's linked list implementation

## References

- [OpenDSA - 15.1 Skip Lists](https://opendsa-server.cs.vt.edu/OpenDSA/Books/CS3/html/SkipList.html)
- [Learn Rust With Entirely Too Many Linked Lists](https://rust-unofficial.github.io/too-many-lists/)
