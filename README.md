# Magneto_Kresling_Truss

### Project name: Computational models for magneto-Kresling truss structures

### Project type: Research project

### Date: 2020-2021

Project description: The Kresling truss structure, derived from Kresling origami, has been widely studied for its bi-stability and various other properties that are useful for diverse engineering applications. The stable states of Kresling trusses are governed by their geometry and elastic response, which involves a limited design space that has been well-explored in previous studies. In this work, we present a magneto-Kresling truss design that involves embedding nodal magnets in the structure, which results in a more complex energy landscape, and consequently, greater tunability under mechanical deformation. We explore this energy landscape first along the zero-torque folding path and then release the restraint on the path to explore the complete two-degree-of-freedom behavior for various structural geometries and magnet strengths. We show that the magnetic interaction could alter the potential energy landscape by either changing the stable configuration, adjusting the energy well depth, or both. Energy wells with different minima endow this magneto-elastic structure with an outstanding energy storage capacity. More interestingly, proper design of the magneto-Kresling truss system yields a tri-stable structure, which is not possible in the absence of magnets. We also demonstrate various loading paths that can induce desired conformational changes of the structure. The proposed magneto-Kresling truss design sets the stage for fabricating tunable, scalable magneto-elastic multi-stable systems that can be easily utilized for applications in energy harvesting, storage, vibration control, as well as active structures with shape-shifting capability.

File and script descriptions: 
<br>(1) funcs.py - python package with user-defined functions, from creating Kresling truss model to calculating its folding paths and visualizing it.
<br>(2) MKT_modeling_and_visualization.ipynb - Jupyter notebook
<br>	Modeled Kresling truss without magnets.
<br>	Modeled a bi-stable magneto-Kresling truss and compared its folding paths with those of purely elastic Kresling truss.
<br>	Gave a tri-stable magneto-Kresling truss example with energy storing capacity.
<br>(3) find_minima_serial.py - python scripts for serially finding potential energy minima for a given magneto- or nonmagneto-Kresling truss.
<br>(4) find_minima_mpi - MPI veriion of finding potential energy minima for a (or many) given magneto- or nonmagneto-Kresling truss(es).
<br>	minima_mpi.py -  python scripts.
<br>	minima_mpi.out - output file by running minima_mpi.py.
<br>	quest_run_this.sh - job submit bash file (sbatch).
<br>(5) T0_folding_truss-and-magnet.mp4 & T0_folding_truss-only.mp4 - videos showing folding a magneto-Kresling truss and a purely elastic truss, respectively.
<br>Other files (protein structure and data) are too large to upload here. They are available per request.

Publication: Yang, X., and Keten, S. (July 29, 2021). "Multi-Stability Property of Magneto-Kresling Truss Structures." ASME. J. Appl. Mech. September 2021; 88(9): 091009. https://doi.org/10.1115/1.4051705
