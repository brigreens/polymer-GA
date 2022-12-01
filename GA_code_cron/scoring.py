import numpy as np
import utils
import gzip
import os
import pybel
import subprocess


def parse_GFN2(filename):
    '''
    Parses through GFN2-xTB output files

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    outputs: list
        [dipole_moment, polarizability]
    '''
    
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        while line:
            if 'molecular dipole' in line:
                line = file.readline()
                line = file.readline()
                line = file.readline()
                    
                line_list = line.split()
                dipole_moment = float(line_list[-1])
                
            elif 'Mol. C8AA' in line:
                line = file.readline()
                line_list = line.split()
                
                polarizability = float(line_list[-1])

            line = file.readline()  
        line = file.readline()

        outputs = [dipole_moment, polarizability]
        
        return outputs

def parse_GFN2_gzip(filename):
    '''
    Parses through gzipped GFN2-xTB output files

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    outputs: list
        [dipole_moment, polarizability]
    '''
    with gzip.open(filename, 'rt') as file:
        line = file.readline()
        while line:
            if 'molecular dipole' in line:
                line = file.readline()
                line = file.readline()
                line = file.readline()
                    
                line_list = line.split()
                dipole_moment = float(line_list[-1])
                
            elif 'Mol. C8AA' in line:
                line = file.readline()
                line_list = line.split()
                
                polarizability = float(line_list[-1])

            line = file.readline()  
        line = file.readline()

        outputs = [dipole_moment, polarizability]
        
        return outputs
    
def make_calc_directories():
    '''
    Makes directories to store all calculation files (input and optimized mol, GFN2 output, etc.)
    TODO: Change names of directories to what better suits your calculations
    '''

    # Check for standard output directories & create them if necessary
    directories = ['input_mol', 'opt_mol', 'output_GFN2']
    for dir in directories:
        exists = os.path.isdir('../%s' % dir)
        if not exists:
            os.makedirs('../%s' % dir)



def run_calculations(polymer, unit_list, gen_counter, run_name, run_label):
    """
    Run the calculations needed for fitness function
    For polarizability, run GFN2-xTB

    Parameters
    ----------
    polymer: list (specific format)
        [(#,#,...), A, B]
    unit_list: dataframe
        contains 1 column of monomer SMILES
    """
    # run GFN2 geometry optimization if not already done (needed for all properties)
    filename = utils.make_file_name(polymer)
    GFN2_file = '../output_GFN2/%s.out.gz' % (filename)

    exists = os.path.isfile(GFN2_file)
    if not exists:
        # make polymer into SMILES string
        poly_smiles = utils.make_polymer_smi(polymer, unit_list)

        # make polymer string into pybel molecule object
        mol = pybel.readstring('smi', poly_smiles)
        utils.make3D(mol)

        # write polymer .mol file to containing folder
        mol.write('mol', '../input_mol/%s.mol' % (filename), overwrite=True)

        # run xTB with a slurm script (TODO: CHANGE SLURM SCRIPT PATHS TO YOUR OWN)
        subprocess.call('(cd ../input_mol && sbatch -J %s --export=run_name=%s,gen_counter=%s,run_label=%s GA_run_GFN2.slurm)' %(filename, run_name, gen_counter, run_label), shell=True)
        
    return

def fitness_function(population, unit_list):
    """
    Calculates the score of a fitness property (polarizability in this case) and ranks the population


    Parameters
    ----------
    population: list
        list of polymers that each have the format [(seq: #,#,#,#,#,#), monomer_index1, monomer_index2]
    unit_list: dataframe
        contains 1 column of monomer SMILES

    Return
    ------
    ranked_population: nested list
        lists of NFAs and their PCE ranked in order of highest PCE first. Also contains the best donor
        [ranked_NFA_names, ranked_PCE, ranked_best_donor]
    """

    score_list = []

    for x in range(len(population)):
        polymer = population[x]
        filename = utils.make_file_name(polymer)

        # parse for polarizability
        try:
            GFN2_file = '../output_GFN2/%s.out.gz' % filename
            GFN2_props = parse_GFN2_gzip(GFN2_file) #dipole_moment, polarizability
            polarizability = GFN2_props[1]
        except:
            print('error with GFN2 file')
            print(filename)
            polarizability = 0
        score_list.append(polarizability)


    ranked_score = []
    ranked_poly_population = []

    # make list of indicies of polymers in population, sorted based on score
    ranked_indices = np.argsort(score_list)

    for x in ranked_indices:
        ranked_score.append(score_list[x])
        ranked_poly_population.append(population[x])

    
    ranked_population = [ranked_score, ranked_poly_population]

    return ranked_population


def fitness_individual(polymer, scoring_prop):
    '''
    Returns the fitness score of an individual 
    
    Parameters
    ----------
    polymer: list
        specific order [(#,#,...), A, B]
    scoring_prop: str
        can be 'polar', 'opt_bg', or 'solv_eng'

    Returns
    -------
    Returns the score depending on the property (polarizability, optical bandgap, or solvation ratio)
    '''

    filename = utils.make_file_name(polymer)
    if scoring_prop == 'polar':
        # parse out polarizability
        try:
            GFN2_file = '../output_GFN2/%s.out.gz' % filename
            GFN2_props = parse_GFN2_gzip(GFN2_file) #dipole_moment, polarizability
            polarizability = GFN2_props[1]
        except:
            print('error with GFN2 file')
            print(filename)
            polarizability = 0
        return polarizability




