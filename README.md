# Magneto_Kresling_Truss

Project name: Computational models for magneto-Kresling truss structures

Project type: Research project

Date: 2020-2021

Project description: In this study, we conducted RMSD-based Targeted Molecular Dynamics (TMD) simulations on glycosylated S-protein head protomers to complete the opening and closing transitions in a short simulation time. PCA was performed on the conformations sampled in 50-ns MD simulations, from which the initial and target structures used in TMD runs were generated. PC1 is characterized by the RBD rotating up and down. PC2 captures the NTD-SD1/SD2-RBD pocket closing and drifting. In TMD simulations, each initial structure finally reached the target conformation with an RMSD around 0.5 Å. The path completion was also validated in the 2D PC1-PC2 space. We defined an RBD orientation dihedral to quantify the global movement of RBD in the protomer, which increased or decreased nearly monotonically in the opening and closing transitions, respectively. The intra- and inter-domain salt bridges related to the protomer conformational change were investigated. Also detected were some critical backbone dihedrals reluctant to undergo the transition and remaining in the initial state.

File and script descriptions: 
	(1) funcs.py - python package with user-defined functions, from creating Kresling truss model to calculating its folding paths and visualizing it.
	(2) MKT_modeling_and_visualization.ipynb - Jupyter notebook
		Modeled Kresling truss without magnets.
		Modeled a bi-stable magneto-Kresling truss and compared its folding paths with those of purely elastic Kresling truss.
		Gave a tri-stable magneto-Kresling truss example with energy storing capacity.
	(3) find_minima_serial.py - python scripts for serially finding potential energy minima for a given magneto- or nonmagneto-Kresling truss.
	(4) find_minima_mpi - MPI veriion of finding potential energy minima for a (or many) given magneto- or nonmagneto-Kresling truss(es).
		minima_mpi.py -  python scripts.
		minima_mpi.out - output file by running minima_mpi.py.
		quest_run_this.sh - job submit bash file (sbatch).
	(5) T0_folding_truss-and-magnet.mp4 & T0_folding_truss-only.mp4 - videos showing folding a magneto-Kresling truss and a purely elastic truss, respectively.
	Other files (protein structure and data) are too large to upload here. They are available per request.

Publication: Yang, X., and Keten, S. (July 29, 2021). "Multi-Stability Property of Magneto-Kresling Truss Structures." ASME. J. Appl. Mech. September 2021; 88(9): 091009. https://doi.org/10.1115/1.4051705
