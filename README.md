# PHYS7688 - FEA Project - FEM Simulation of Superconducting CPW

hello Prof. Maxson, apologies for the code being quite messy!

# AI Usage:

AI was most useful for me in a few situations:
- Devloping test cases. For verification purposes, it suggested and provided code for various simple scenarios that I could verify it got right, and plug into my simulation.
- Plotting/interfacing with APIs. Matplotlib plots are not trivial with these libraries, so it saved me a lot of time with boilerplate plotting code with high accuracy (and again, easy to verify that it was doing the correct things).
- Debugging. The most significant was moving from Dirchlet BC's for A to using the volumetric transport current. I could not figure out why accuracy decreased in the thin film regime, and it helpfully identified this as the problem and got me started on working towards the solution.

I discovered it was often quite bad at the physics. When trying to debug other pieces, it often gave incorrect or nonsensical corrections to physics I was confident was correct. Mainly, it _really_ insisted that I could in fact use PEC boundary conditions for the superconductor, even when telling it I explicitly did not wish to.
