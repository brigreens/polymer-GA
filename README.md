# Polymer-GA

This repository contains the code for the Hutchison Group's polymer genetic algorithm (GA). This GA will perform crossover, mutation, etc. on oligomers of 2 monomers with a length of 6. The default sequence is ABABAB. 

The fitness function can be changed for your desired scoring property. In this code, we have scored on polarizability, optical bandgap, and solvation energy ratio between water and hexane, assuming the semiempirical calculations for your search space are done previsouly (`Calculations/GFN2`, `Calculations/sTDDFTxtb`, and xTB in `Calculations/solvation_water` and `Calculations/solvation_hexane`)

The main GA code is `GA_code/GA_main.py`. The following hyperparameters should be changed in the `main` function:

- 'run_label': Sets the initial random state and allows for replication. Can be set as 'A', 'B', 'C', etc.
- `chem_property`: If you are using our fitness functions, the options are 'polar', 'opt_bg', and 'solv_eng'.
- `pop_size`: This is the size of the populations. The default is 32 oligomers.
- `selection_method`: Can be 'random', k-way tournament with k=2 as'tournament_2', k=3 as 'tournament_3', and k=4 as'tournament_4', 'roulette', 'rank', or stochastic universal sampling as 'SUS'. Default is 'tournament_3'.
- `mutation_rate`: Sets the chance each oligomer in a generation will undergo mutation. Can be between 0 and 1. Default is 0.4.
- `elitism_perc`: Percentage of the top oligomers that will pass to the next generation. Can be between 0 and 1. Default is 0.5.
- `spear_thresh`: Spearman coefficient threshold - spearman coefficent must be greater than this value to trip the convergence counter. Default is 0.8.
- `conv_gen`: Convergence threshold - # of consecutive generations that need to meet Spearman coefficient threshold before termination. Default is 50.

VERY IMPORTANT! Change the `run_name` for every GA run


A csv containing SMILES is supplied (`monomer_SMILES.csv`) and can be changed for your dataset.
