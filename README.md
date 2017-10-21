# Designer-Motion
Designing mechanical meta material with specific input/output behavior

First Commit: I wrote Random-Dynamical-Matrix.py
This basically generates a random sparse matrix which is supposed to represent the rigidity
matrix of a spring network. Then from that I minimize the cost(spring-constants) = Energy of desired motion/lowest energy of system.

Second Commit, 10/5/2017 - Thursday: I'm commiting changes to latticeMaking

- This has a bunch of methods for creating a square lattice with diagonal bonds.
- There are also options to randomize the lattice.
- We also create rigidity and dynamical matrices and a few other useful methods. 
- Still working on energy minimization. 

Third Commit 10/10/2017 - Tuesday: Changes to EnergyFunctions and InputOutput:

- Energy functions calculates the energy given a set of boundary displacements. 
- Minimizint the cost function with respect to the spring constants will result in low energy for the desired boundary displacements.
- InputOutput: is method for generating a mechanical structure that has desired input/output relations. 
- For example, imagine you want have a robotic arm that responds mechanically on the hand side based on mechanical input in the shoulder side
- I am commiting changes to it now because so far I've only been controlling the spring constants. And that's not enough to generate the correct
IO relations
- Next I will modify this script to also minimize with respect to the positions of the interior points of the structure.


Fourth Commit, 10/21/2017 Saturday. The script little_square:

- Made a lot of changes. 
- The main one right now is little_square. It seems to be designing the proper motion.
- We want to make the square vertices move in a certain way.
- We add in as many vertices as we want (about 10) and connect every vertex to every other.
- We then optimize with respect to the spring constants and positions of extra vertices to make the motion happen.
- It seems to be working the only problem is that the two lowest energy modes are near each other. 
