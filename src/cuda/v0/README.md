This first version is built on top of the flattened kd-tree implemented in the vectorization part. From that, data is moved to the GPU, computation is done and output is copied back on host.

## Technical details

The data structure used is the flattened kd-tree, meaning an array of nodes and other data where each node contains the children idxs, so that one can navigate the tree using the idxs in a BFS way. 

Data copied to GPU:

- tgt nodes
- query arrays (qx, qy, qz)
- best indexes (bi)
- best distances (bd2)

Data copied back to Host:

- best indexes
- best distances

Then host computes the matches by its own.

## Performance

This version parallelizes the slowest part of the code: the NN search. The speedup is huge, and at the end the time taken by this search is much less than the rest of the code, which tells me that the parallelization was good and that I can achieve a bigger speedup focusing on the rest of the code. 
