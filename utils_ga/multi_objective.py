from pylab import *

from inspyred.ec.emo import NSGA2
from inspyred.ec import terminators, variators, replacers, selectors
from inspyred.ec import EvolutionaryComputation

from utils_ga.inspyred_utils import *
from utils_ga.plot_utils import *

def custom_mutation(random, candidates, args):
    """Custom mutation for a vector of integer values."""
    mutation_rate = args.get('mutation_rate', 0.1)  # Probability of mutating each gene
    mutation_range = args.get('mutation_range', (-10, 10))  # Range for mutation values

    mutant = candidates[:]  # Make a copy of the candidate
    for i in range(len(mutant)):
        if random.random() < mutation_rate:
            # Apply a random mutation within the specified range
            mutant[i] += random.randint(*mutation_range)
            # Ensure the mutated value stays within bounds (if applicable)
            if 'lower_bound' in args and 'upper_bound' in args:
                lower_bound = args['lower_bound']
                upper_bound = args['upper_bound']
                mutant[i] = max(lower_bound[i], min(mutant[i], upper_bound[i]))
    return mutant


def run_nsga2(random, problem, display=False, num_vars=0, use_bounder=True,
        variator=None, **kwargs) :
    """ run NSGA2 on the given problem """
    
    #create dictionaries to store data about initial population, and lines
    initial_pop_storage = {}
 
    algorithm = NSGA2(random)
    algorithm.terminator = terminators.generation_termination 
    if variator is None :     
        # algorithm.variator = [variators.partially_matched_crossover,
        #                       variators.random_reset_mutation,]
        algorithm.variator = [variators.blend_crossover,
                              variators.gaussian_mutation]
    else :
        algorithm.variator = variator
    
    kwargs["num_selected"]=kwargs["pop_size"]  
    if use_bounder :
        kwargs["bounder"]=problem.bounder
    
    if display:
        if num_vars == 2 :
            algorithm.observer = [initial_pop_observer]
        if "csv_file" in kwargs :
            algorithm.observer = [csv_pop_observer]
        
    final_pop = algorithm.evolve(evaluator=problem.evaluator,  
                          maximize=problem.maximize,
                          initial_pop_storage=initial_pop_storage,
                          num_vars=num_vars, 
                          generator=problem.generator,
                          **kwargs)         
    
    best_guy = final_pop[0].candidate[0:num_vars]
    best_fitness = final_pop[0].fitness
    #final_pop_fitnesses = asarray([guy.fitness for guy in algorithm.archive])
    #final_pop_candidates = asarray([guy.candidate[0:num_vars] for guy in algorithm.archive])
    final_pop_fitnesses = asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = asarray([guy.candidate[0:num_vars] for guy in final_pop])

    if display :
        # Plot the parent and the offspring on the fitness landscape 
        # (only for 1D or 2D functions)
        if num_vars == 1 :
            plot_results_multi_objective_1D(problem, 
                                  initial_pop_storage["individuals"], 
                                  initial_pop_storage["fitnesses"], 
                                  final_pop_candidates, final_pop_fitnesses,
                                  'Initial Population', 'Final Population',
                                  len(final_pop_fitnesses[0]), kwargs)
    
        elif num_vars == 2 :
            plot_results_multi_objective_2D(problem, 
                                  initial_pop_storage["individuals"], 
                                  final_pop_candidates, 'Initial Population',
                                  'Final Population',
                                  len(final_pop_fitnesses[0]), kwargs)

        plot_results_multi_objective_PF(final_pop, kwargs['fig_title'] + ' (Pareto front)')
    
    return final_pop_candidates, final_pop_fitnesses

def run_ga(random,problem, obj, display=False, num_vars=0, 
           maximize=False, use_bounder=True, **kwargs) :
    """ run a GA on the given problem """
    
    #create dictionaries to store data about initial population, and lines
    initial_pop_storage = {}
    
    algorithm = EvolutionaryComputation(random)
    algorithm.terminator = terminators.generation_termination
    algorithm.replacer = replacers.generational_replacement    
    algorithm.variator = [variators.uniform_crossover, 
                          variators.gaussian_mutation]
    algorithm.selector = selectors.tournament_selection
    if display:
        if num_vars == 2:
            algorithm.observer = [plot_observer, initial_pop_observer]
        else :
            algorithm.observer = [plot_observer]
        # if "csv_file" in kwargs :
        #     algorithm.observer = [csv_pop_observer]
    
    kwargs["num_selected"]=kwargs["pop_size"]  
    if use_bounder :
        kwargs["bounder"]=problem.bounder
    if "pop_init_range" in kwargs :
        kwargs["generator"]=generator
    else :
        kwargs["generator"]=problem.generator
    
    kwargs["problem"] = problem
    kwargs["class"] = obj
    final_pop = algorithm.evolve(evaluator=single_objective_evaluator,
                                 maximize=problem.maximize,
                                 initial_pop_storage=initial_pop_storage,
                                 num_vars=num_vars, 
                                 **kwargs)                          

    best_guy = final_pop[0].candidate
    best_fitness = final_pop[0].fitness.fitness
    final_pop_fitnesses = asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = asarray([guy.candidate for guy in final_pop])
    
    if display :
        # Plot the parent and the offspring on the fitness landscape 
        # (only for 1D or 2D functions)
        if num_vars == 1 :
            plot_results_multi_objective_1D(problem, 
                                  initial_pop_storage["individuals"], 
                                  initial_pop_storage["fitnesses"], 
                                  final_pop_candidates, final_pop_fitnesses,
                                  'Initial Population', 'Final Population',
                                  len(final_pop_fitnesses[0]), kwargs)
    
        elif num_vars == 2 :
            plot_results_multi_objective_2D(problem, 
                                  initial_pop_storage["individuals"], 
                                  final_pop_candidates, 'Initial Population',
                                  'Final Population',
                                  len(final_pop_fitnesses[0]), kwargs)
   
    return best_guy, best_fitness
