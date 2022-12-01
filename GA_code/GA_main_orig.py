# Imports
import csv
import random
import numpy as np
from copy import deepcopy
import pickle
import pandas as pd
from scipy.stats import spearmanr
import argparse
from itertools import product
import os
import subprocess



import utils
import scoring


def main(pop_size, selection_method, mutation_rate, elitism_perc, spear_thresh, conv_gen, run_label, run_name, restart, init_restart):

    # Create list of possible building block unit SMILES in specific format
    unit_list = utils.make_unit_list()

    # Create all possible numerical sequences for hexamers with 1 or 2 monomer species
    sequence_list = list(product(range(2), repeat=6))

    # Check for standard output directories & create them if necessary
    directories = ['quick_files', 'full_files', 'mono_freq', 'last_gen_params', 'rand_states', 'initial_randstates']
    for dir in directories:
        exists = os.path.isdir('../%s' % dir)
        if not exists:
            os.makedirs('../%s' % dir)

    # Check for scoring-function related directories & create them if necessary
    scoring.make_calc_directories()

    if restart == True:
        # reload parameters and random state from restart file
        last_gen_filename = '../last_gen_params/last_gen_' + run_name + '.p'
        open_params = open(last_gen_filename, 'rb')
        params = pickle.load(open_params)
        open_params.close()

        randstate_filename = '../rand_states/randstate_' + run_name + '.p'
        open_rand = open(randstate_filename, 'rb')
        randstate = pickle.load(open_rand)
        random.setstate(randstate)
        open_rand.close()

    else:
        if init_restart == False:
            # sets initial state
            randstate = random.getstate()
            initial_randstate = '../initial_randstates/initial_randstate_' + run_label +'.p'
            rand_file = open(initial_randstate, 'wb')
            pickle.dump(randstate, rand_file)
            rand_file.close()
        else:
            # re-opens intial state during troubleshooting
            initial_randstate = '../initial_randstates/initial_randstate_' + run_label + '.p'
            open_rand = open(initial_randstate, 'rb')
            randstate = pickle.load(open_rand)
            random.setstate(randstate)
            open_rand.close()

        # run initial generation if NOT loading from restart file
        params = init_gen(pop_size, selection_method, mutation_rate, elitism_perc, run_name, unit_list, sequence_list, spear_thresh)

        # pickle parameters needed for restart
        last_gen_filename = '../last_gen_params/last_gen_' + run_name + '.p'
        params_file = open(last_gen_filename, 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = '../rand_states/randstate_' + run_name + '.p'
        rand_file = open(randstate_filename, 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()

    # get convergence counter from inital parameters
    spear_counter = params[11]

    while spear_counter < conv_gen:
        # run next generation of GA
        params = next_gen(params)

        # pickle parameters needed for restart
        last_gen_filename = '../last_gen_params/last_gen_' + run_name + '.p'
        params_file = open(last_gen_filename, 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = '../rand_states/randstate' + run_name + '.p'
        rand_file = open(randstate_filename, 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()

        # update convergence counter
        spear_counter = params[11]

def next_gen(params):
    '''
    Runs the next generation of the GA

    Paramaters
    ----------
    params: list
        list with specific order
        params = [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name]
    
    Returns
    --------
    params: list
        list with specific order
        params = [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name]
    '''
    pop_size = params[0]
    unit_list = params[1]
    sequence_list = params[2]
    gen_counter = params[3]
    fitness_list = params[4]
    selection_method = params[5]
    mutation_rate = params[6]
    elitism_perc = params[7]
    run_name = params[8]
    mono_df = params[9]
    spear_thresh = params[10]
    spear_counter = params[11]


    gen_counter +=1
    ranked_population = fitness_list[1]
    ranked_scores = fitness_list[0]

    # Reversing the list so the best scores are first
    ranked_population.reverse()
    ranked_scores.reverse()

    # Select percentage of top performers for next genration - "elitism"
    elitist_population = elitist_select(ranked_population, elitism_perc)

    # Selection, Crossover & Mutation - create children to repopulate bottom 50% of NFAs in population
    new_population = select_crossover_mutate(ranked_population, ranked_scores, elitist_population, selection_method, mutation_rate, pop_size, unit_list, sequence_list)

    fitness_list = scoring.fitness_function(new_population, unit_list) # [ranked_score, ranked_poly_population]

    min_score = fitness_list[0][0]
    median = int((len(fitness_list[0])-1)/2)
    med_score = fitness_list[0][median]
    max_score = fitness_list[0][-1]

    # calculate new monomer frequencies
    old_freq = deepcopy(mono_df)
    mono_df = utils.get_monomer_freq(ranked_population, mono_df, gen_counter)

    # generate monomer frequency csv
    mono_df.to_csv('../monomer_frequency.csv')

    # get ranked index arrays for Spearman correlation
    old_ranks = utils.get_ranked_idx(old_freq, gen_counter-1)[:10]
    new_ranks = utils.get_ranked_idx(mono_df, gen_counter)[:10]


    # calculate Spearman correlation coefficient
    spear = spearmanr(old_ranks, new_ranks)[0]

    # keep track of number of successive generations meeting Spearman criterion
    if spear > spear_thresh:
        spear_counter += 1
    else:
        spear_counter = 0

    quick_filename = '../quick_files/quick_analysis_' + run_name + '.csv'
    with open(quick_filename, mode='a+') as quick_file:
        # write to quick analysis file
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([gen_counter, min_score, med_score, max_score, spear, spear_counter])

    for x in range(len(fitness_list[0])):
        poly = fitness_list[1][x]
        score = fitness_list[0][x]

        # write full analysis file
        full_filename = '../full_files/full_analysis_' + run_name + '.csv'
        with open(full_filename, mode='a+') as full_file:
            full_writer = csv.writer(full_file)
            full_writer.writerow([gen_counter, poly, score])

    

    params = [pop_size, unit_list, sequence_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, mono_df, spear_thresh, spear_counter]
    
    return(params)

def parent_select(ranked_population, ranked_scores, selection_method):
    '''
    Selects two parents. Method of selection depends on selection_method
    
    Parameters
    ----------
    ranked_population: list
        ordered list of polymer names, ranked from best to worst
    ranked_scores: list
        ordered list of scores, ranked from best to worst
    selection_method: str
        Type of selection operation. Options are 'random', 'random_top50', 'tournament', 'roulette', 'SUS', or 'rank'

    Returns
    -------
    parents: list
        list of length 2 containing the two parent indicies
    '''
    # randomly select parents
    if selection_method == 'random':
        parents = []
        # randomly select two parents (as indexes from population) to cross
        parent_a = random.randint(0, len(ranked_population) - 1)
        parent_b = random.randint(0, len(ranked_population) - 1)
        # ensure parents are unique indiviudals
        if len(ranked_population) > 1:
            while parent_b == parent_a:
                parent_b = random.randint(0, len(ranked_population) - 1)

        parents.append(ranked_population[parent_a])
        parents.append(ranked_population[parent_b])

    # randomly select parents from top 50% of population
    elif selection_method == 'random_top50':
        parents = []
        # randomly select two parents (as indexes from population) to cross
        parent_a = random.randint(0, len(ranked_population)/2 - 1)
        parent_b = random.randint(0, len(ranked_population)/2 - 1)
        # ensure parents are unique indiviudals
        if len(ranked_population) > 1:
            while parent_b == parent_a:
                parent_b = random.randint(0, len(ranked_population)/2 - 1)

        parents.append(ranked_population[parent_a])
        parents.append(ranked_population[parent_b])

    # 3-way tournament selection
    elif selection_method == 'tournament_3':
        parents = []
        # select 2 parents
        while len(parents) < 2:
            individuals = []
            # select random individual 1
            individual_1 =  random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_1)

            # select random individual 2
            individual_2 =  random.randint(0, len(ranked_population) - 1)
            while individual_1 == individual_2:
                individual_2 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_2)

            # select random individual 3
            individual_3 =  random.randint(0, len(ranked_population) - 1)
            while ((individual_3 == individual_1) or (individual_3 == individual_2)):
                individual_3 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_3)
            # Make list of their fitness
            scores = [ranked_scores[individual_1], ranked_scores[individual_2], ranked_scores[individual_3]]
            
            # find the index of the best fitness score
            best_index = np.argmax(scores)


            best_individual = individuals[best_index]
            parent = ranked_population[best_individual]

            if parent not in parents:
                parents.append(parent)

    # 4-way tournament selection
    elif selection_method == 'tournament_4':
        parents = []
        # select 2 parents
        while len(parents) < 2:
            individuals = []
            # select random individual 1
            individual_1 =  random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_1)

            # select random individual 2
            individual_2 =  random.randint(0, len(ranked_population) - 1)
            while individual_1 == individual_2:
                individual_2 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_2)

            # select random individual 3
            individual_3 =  random.randint(0, len(ranked_population) - 1)
            while ((individual_3 == individual_1) or (individual_3 == individual_2)):
                individual_3 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_3)

            # select random individual 4
            individual_4 =  random.randint(0, len(ranked_population) - 1)
            while ((individual_4 == individual_1) or (individual_4 == individual_2) or (individual_4 == individual_3)):
                individual_4 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_4)
            # Make list of their fitness
            scores = [ranked_scores[individual_1], ranked_scores[individual_2], ranked_scores[individual_3], ranked_scores[individual_4]]
            
            # find the index of the best fitness score
            best_index = np.argmax(scores)


            best_individual = individuals[best_index]
            parent = ranked_population[best_individual]

            if parent not in parents:
                parents.append(parent)

    elif selection_method == 'tournament_2':
        parents = []
        # select 2 parents
        while len(parents) < 2:
            individuals = []
            # select random individual 1
            individual_1 =  random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_1)

            # select random individual 2
            individual_2 =  random.randint(0, len(ranked_population) - 1)
            while individual_1 == individual_2:
                individual_2 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_2)

            # Make list of their fitness
            scores = [ranked_scores[individual_1], ranked_scores[individual_2]]
            
            # find the index of the best fitness score
            best_index = np.argmax(scores)

            best_individual = individuals[best_index]
            parent = ranked_population[best_individual]

            if parent not in parents:
                parents.append(parent)

    elif selection_method == 'roulette': #ranked_population, ranked_scores, selection_method
        parents = []
        # create wheel 
        wheel = []
        # bottom limit
        limit = 0

        # sum of scores
        total = sum(ranked_scores)
        for x in range(len(ranked_scores)):
            # fitness proportion
            fitness = ranked_scores[x]/total
            # appends the bottom and top limits of the pie, the score, and the polymer
            wheel.append((limit, limit+fitness, ranked_scores[x], ranked_population[x]))
            limit += fitness

        # random number between 0 and 1
        r = random.random()
        # search for polymer in that pie
        score, polymer = utils.binSearch(wheel, r)
        parents.append(polymer)
        while len(parents) < 2:
            r = random.random()
            score, polymer = utils.binSearch(wheel, r)
            if polymer not in parents:
                parents.append(polymer)
            
    
    elif selection_method == 'SUS': #stochastic universal sampling
        parents = []
        # create wheel 
        wheel = []
        # bottom limit
        limit = 0

        # sum of scores
        total = sum(ranked_scores)
        for x in range(len(ranked_scores)):
            # fitness proportion
            fitness = ranked_scores[x]/total
            # appends the bottom and top limits of the pie, the score, and the polymer
            wheel.append((limit, limit+fitness, ranked_scores[x], ranked_population[x]))
            limit += fitness

        # separation between selected points on wheel
        stepSize = 0.5
        # random number between 0 and 1
        r = random.random()

        # search for polymer in that pie
        score, polymer = utils.binSearch(wheel, r)
        parents.append(polymer)
        while len(parents) < 2:
            r += stepSize
            if r>1:
                r %= 1
            score, polymer = utils.binSearch(wheel, r)
            parents.append(polymer)

    elif selection_method == 'rank':
        parents = []
        # reverse population so worst scores get smallest slice of pie
        ranked_population.reverse()

        # creates wheel 
        wheel = []
        # total sum of the indicies 
        total = sum(range(1, len(ranked_population)+1))
        top = 0
        for p in range(len(ranked_population)):
            # adds 1 so the first element does not have a pie slice of size 0
            x = p+1
            # fraction of total pie
            f = x/total
            wheel.append((top, top+f, ranked_population[p]))
            top += f

        # pick random parents from wheel
        while len(parents) < 2:    
            r = random.random()
            polymer = utils.rank_binSearch(wheel, r)

            if polymer not in parents:
                parents.append(polymer)
    else:
        print('not a valid selection method')

    return parents



def select_crossover_mutate(ranked_population, ranked_scores, elitist_population, selection_method, mutation_rate, pop_size, unit_list, sequence_list):
    '''
    Perform selection, crossover, and mutation operations

    Parameters
    ----------
    ranked_population: list
        ordered list containing lists of polymers of format [mon_1_index, mon_2_index]
    ranked_scores: list
        ordered list of scores of population
    elitist_population: list
        list of elite polymers to pass to next generation
    selection_method: str
        Type of selection operation. Options are 'random', 'tournament', 'roulette', 'SUS', or 'rank'
    mutation_rate: int
        Chance of polymer to undergo mutation
    pop_size: int
        number of individuals in the population
    unit_list: Dataframe
        dataframe containing the SMILES of all monomers
    sequence_list: list
        list containing all possible sequences
    
    Returns
    -------
    new_pop: list
        list of new polymers of format [mon_1_index, mon_2_index]
    new_pop_smiles: list
        list of the SMILES of the new polymers
    '''
    
    new_pop = deepcopy(elitist_population)

    # loop until enough children have been added to reach population size
    while len(new_pop) < pop_size:

        # select two parents
        parents = parent_select(ranked_population, ranked_scores, selection_method)
        
        # create hybrid child
        temp_child = []

        # randomly determine which parent's sequence will be used
        par_seq = random.randint(0, 1)

        # give child appropriate parent's sequence
        temp_child.append(parents[par_seq][0])

        # take first unit from parent 1 and second unit from parent 2
        temp_child.append(parents[0][1])
        temp_child.append(parents[1][2])

        # give child opportunity for mutation
        temp_child = mutate(temp_child, unit_list, sequence_list, mutation_rate)


        # check for duplication
        if temp_child in new_pop:
            pass
        elif utils.not_valid(temp_child) == True:
            pass
        else:
            new_pop.append(temp_child)       

    return new_pop


def mutate(temp_child, unit_list, sequence_list, mut_rate):
    '''
    Mutation operator. Replaces one monomer with a new one

    Parameters
    -----------
    temp_child: list
        format is [mon_1_index, mon_2_index]
    unit_list: Dataframe
        dataframe containing the SMILES of all monomers
    mut_rate: int
        Chance of polymer to undergo mutation
    
    Return
    ------
    temp_child: list
        format is [mon_1_index, mon_2_index]
    '''

    # determine whether to mutate based on mutation rate
    rand = random.randint(1, 100)
    if rand <= (mut_rate * 100):
        pass
    else:
        return temp_child

   # choose point of mutation
    point = random.randint(0, 2)

    # replace sequence
    if point == 0:
        temp_child[point] = sequence_list[random.randint(
            0, len(sequence_list) - 1)]
    # or replace specific monomer
    else:
        new_mono = random.randint(0, len(unit_list) - 1)
        temp_child[point] = new_mono

    return temp_child

def elitist_select(ranked_population, elitism_perc):
    '''
    Selects a percentage of the top polymers to ensure in next generation
    
    Parameters
    ----------
    ranked_population: list
        ordered list containing lists of polymers of format [mon_1_index, mon_2_index]
    elitism_perc: float
        percentage of generation to pass to next generation. Can range from 0-1 (although 1 is the entire generation)

    Returns
    -------
    elitist_list: list
        list of polymers each of format [mon_1_index, mon_2_index]
    '''

    # find number of parents to pass to next generation
    elitist_count = int(len(ranked_population) * elitism_perc)
    elitist_list = []

    for x in range(elitist_count):
        elitist_list.append(ranked_population[x])

    return elitist_list
    

def init_gen(pop_size, selection_method, mutation_rate, elitism_perc, run_name, unit_list, sequence_list, spear_thresh):
    '''
    Create initial population

    Parameters
    -----------
    pop_size: int
        number of individuals in the population
    selection_method: str
        Type of selection operation. Options are 'random', 'tournament', 'roulette', 'SUS', or 'rank'
    mutation_rate: int
        Chance of polymer to undergo mutation
    elitism_perc: float
        percentage of generation to pass to next generation. Can range from 0-1 (although 1 is the entire generation)
    run_name: str
        name of this GA run
    unit_list: Dataframe
        dataframe containing the SMILES of all monomers
    sequence_list: list
        list containing all possible sequences
    spear_thresh: float
        minimum spearman correlation coefficient to trip convergence counter
    
    Returns
    -------
    params: list
        format is [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, mono_df, spear_thresh, spear_counter]
    '''
    # initialize generation counter
    gen_counter = 1

    # initialize convergence counter
    spear_counter = 0

    # create monomer frequency df [freq_0]
    mono_df = pd.DataFrame(0, index=np.arange(len(unit_list)), columns=np.arange(1))
    mono_df = mono_df.rename_axis('mono_idx')
    mono_df = mono_df.rename(columns={0:'freq_0'})

    # create inital population as list of polymers
    population = []

    while len(population) < pop_size:
        temp_poly = []

        # select sequence type for polymer
        poly_seq = sequence_list[random.randint(0, len(sequence_list) - 1)]
        temp_poly.append(poly_seq)

        # select monomer types for polymer
        for num in range(2):
            # randomly select a monomer index
            poly_monomer = random.randint(0, len(unit_list) - 1)
            temp_poly.append(poly_monomer)

        # check for duplication
        if temp_poly in population:
            continue
        else:
            population.append(temp_poly)

    # calculate new monomer frequencies
    mono_df = utils.get_monomer_freq(population, mono_df, gen_counter)

    # create new analysis files
    quick_filename = '../quick_files/quick_analysis_' + run_name + '.csv'
    with open(quick_filename, mode='w+') as quick:
        quick_writer = csv.writer(quick)
        quick_writer.writerow(['gen', 'min_score', 'med_score', 'max_score', 'spearman', 'conv_counter'])

    full_filename = '../full_files/full_analysis_' + run_name + '.csv'
    with open(full_filename, mode='w+') as full:
        full_writer = csv.writer(full)
        full_writer.writerow(['gen', 'filename', 'score'])



    fitness_list = scoring.fitness_function(population, unit_list) # [ranked_score, ranked_poly_population] 

    min_score = fitness_list[0][0]
    median = int((len(fitness_list[0])-1)/2)
    med_score = fitness_list[0][median]
    max_score = fitness_list[0][-1]

    # inital spearman coefficent for output file
    spear = 0

    quick_filename = '../quick_files/quick_analysis_' + run_name + '.csv'
    with open(quick_filename, mode='a+') as quick_file:
        # write to quick analysis file
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([1, min_score, med_score, max_score, spear, spear_counter])

    for x in range(len(fitness_list[0])):
        poly = fitness_list[1][x]
        score = fitness_list[0][x]

        # write full analysis file
        full_filename = '../full_files/full_analysis_' + run_name + '.csv'
        with open(full_filename, mode='a+') as full_file:
            full_writer = csv.writer(full_file)
            full_writer.writerow([gen_counter, poly, score])

    params = [pop_size, unit_list, sequence_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, mono_df, spear_thresh, spear_counter]
    
    return(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')

    param_type = subparser.add_parser('param_type')
    param_type.add_argument('--pop_size', type=int, help='size of the population')
    param_type.add_argument('--selection_method', type=str, help='options are random, random_top50, tournament_2, tournament_3, tournament_4, roulette, rank, or SUS')
    param_type.add_argument('--mutation_rate', type=float, help='any number between 0 and 1')
    param_type.add_argument('--elitism_perc', type=float, help='any number between 0 and 1')
    param_type.add_argument('--spear_thresh', type=float, help='spearman coefficent must be greater than this value to trip the convergence counter')
    param_type.add_argument('--conv_gen', type=int, help='# of consecutive generations that need to meet Spearman coefficient threshold before termination')

    # 'A', 'B', 'C', 'D', or 'E'
    parser.add_argument('--run_label', type=str, required=True, help='If you want to reuse the same initial state, input the same letter. Can be any letter, such as A, B, C, etc..')
    # name of GA run
    parser.add_argument('--name', type=str, required=True, help='the name you want to refer this GA run by')
    
    # are you restarting?
    parser.add_argument('--restart', action='store_true', help='use this flag if you are restarting the GA from NOT the first generation')
    parser.add_argument('--init_restart', action='store_true', help='use this flag if you are restarting the GA from the first generation')

    args = parser.parse_args()

    # setting default parameters if none are given
    if args.pop_size == None:
        pop_size = 32
    else:
        pop_size = args.pop_size

    if args.selection_method == None:
        selection_method = 'tournament_3'
    else:
        selection_method = args.selection_method  

    if args.mutation_rate == None:
        mutation_rate = 0.4
    else:
        mutation_rate = args.mutation_rate 

    if args.elitism_perc == None:
        elitism_perc = 0.5
    else:
        elitism_perc = args.elitism_perc     

    if args.spear_thresh == None:
        spear_thresh = 0.8
    else:
        spear_thresh = args.spear_thresh    

    if args.conv_gen == None:
        conv_gen = 50
    else:
        conv_gen = args.conv_gen 


    main(pop_size, selection_method, mutation_rate, elitism_perc, spear_thresh, conv_gen, args.run_label, args.name, args.restart, args.init_restart)
