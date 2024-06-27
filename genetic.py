from tqdm import tqdm
from model import *
from utils import *
from inspyred import ec
from inspyred.benchmarks import Benchmark
from utils_ga import multi_objective
from utils_ga.inspyred_utils import NumpyRandomWrapper
from inspyred.ec.emo import Pareto
from pylab import *

def genetic_quant(model, device, loader, criterion, neuron_opt=False): 

    stats = gatherStats(model, loader, device)

    vars = [1] + [module.out_features if neuron_opt else 1 for module in model.modules() if isinstance(module, nn.Linear)]
    print("Number of variables: ", sum(vars))

    benchmark = BitsBenchmark(vars, model, stats, loader, device, criterion)

    # parameters for the GA
    args = {}
    args["pop_size"] = 100
    args["tournament_size"] = 5
    args["max_generations"] = 5

    args['mutation_rate'] = 0.5
    args['gaussian_mean'] = 0
    args['gaussian_stdev'] = 2

    args["fig_title"] = 'GA'
    args["csv_file"] = 'neuron50_20_41.csv'

    # make sure that this array has the same size as num_objs
    seed=41
    rng = NumpyRandomWrapper(seed)

    final_pop, final_pop_fitnesses = multi_objective.run_nsga2(rng, benchmark,
                                    display=True, num_vars=sum(vars),
                                    **args)


    print("Final Population\n", final_pop)
    print()
    print("Final Population Fitnesses\n", final_pop_fitnesses)  

class BitsBenchmark(Benchmark):
    def __init__(self, vars, net, stats, loader, device, criterion):
        self.net = net
        self.vars = vars
        self.dimensions = sum(vars)
        super(BitsBenchmark, self).__init__(self.dimensions)
        self.bounder = ec.Bounder(1, 8)
        self.device = device
        self.criterion = criterion
        self.loader = loader
        self.stats = stats
        self.maximize = False
        

    def generator(self, random, args):
        
        a =  [random.randint(1, 8) for _ in range(self.dimensions)]
        return a

    def evaluator(self, candidates, args):
        fitness = []
        self.net.eval().to(self.device)               

        for candidate in tqdm(candidates, desc='Evaluating', leave=True): 
            cand = candidate.copy()
            stats = self.stats 
            for i, (k, _ )in enumerate(stats.items()):
                if self.vars[i] == 1:
                    stats[k]['bits'] = int(cand.pop(0))
                else:
                    popped = [int(elem) for elem in cand[:self.vars[i]]]
                    cand = cand[self.vars[i]:]
                        
                    stats[k]['bits'] = popped
                    stats[k]['bits_out'] = stats['input']['bits']

            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.net.Qforward(data, stats)
                    test_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.loader.dataset)

            err = 1 -  correct / len(self.loader.dataset)
        
            m_size = model_size(stats) / 1024

            fitness.append(Pareto([err, m_size]))

        return fitness
    
