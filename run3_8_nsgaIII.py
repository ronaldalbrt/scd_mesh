from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np
from objectives import *
from powerflow import *
import pickle
from tqdm import tqdm
from pathlib import Path

class RCM_prob(Problem):
    def __init__(self, n_var, n_obj, n_rcm, pop_size, xl, xu):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
        self.n_rcm = n_rcm
        self.pop_size = pop_size
    
    def _evaluate(self, x, out, *args, **kwargs):
        output = None
        for i in range(self.pop_size):
            result = RCM(list(x[i, :].flatten()), self.n_rcm)
            if output is None:
                output = np.column_stack(result).astype(float)
            else:
                output = np.vstack([output, np.column_stack(result)]).astype(float)
                
        out['F'] = output

def main():
    Path("result").mkdir(parents=False, exist_ok=True)
    
    num_runs = 30
    rcm = 43
    rcm_func = lambda x: RCM(x, rcm)

    objectives_dim = 2
    position_dim = 34
    position_max_value = np.array([1] * position_dim)
    position_min_value = np.array([0] * position_dim)
    population_size = 100
    n_partitions = 12

    print(f"Running NSGA3 on RCM{rcm}")

    result = {}
    combined = None
    for i in tqdm(range(num_runs)):
        ref_dirs = get_reference_directions("das-dennis", objectives_dim, n_partitions=n_partitions)
        nsga3 = NSGA3(pop_size=population_size, ref_dirs=ref_dirs)

        res = minimize(RCM_prob(n_var=position_dim, n_obj=objectives_dim, n_rcm=rcm, pop_size=population_size, xl=position_min_value, xu=position_max_value), nsga3, seed=1, termination=('n_gen', 600))

        get_population = lambda p: p.X
        
        result[i+1] = {"F":res.F, "P":[get_population(p) for p in res.pop]}
        if combined is None:
            combined = res.F
        else:
            combined = np.vstack((combined, res.F))

        result["combined"] = combined

    with open(f'result/NSGA3_RCM{rcm}.pkl', 'wb') as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    main()