Basic implementation of the ICP algorithm using a KD-tree as data structure to save the tgt pointcloud. 

The use of a kd-tree makes the search much more efficient, going from a complexity of O(nm) for the brute-force implementation to 
an O(nlogm). The makefile creates a baseline executable, which does not have any optimization flag, and a serial one, with
optimization flags for the compiler. The serial version has a speedup of almost 2x. 

In this implementation, I have not focused on code and memory optimization but rather on having a working baseline
to work with and to compare with future parallel and vectorized versions.
