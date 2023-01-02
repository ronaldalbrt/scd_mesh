from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
import numpy as np
import datetime
from objectives import *
from MESH import *
import pygmo as pg
import pickle
from tqdm import tqdm
from pathlib import Path
import itertools

optimizationMap = {
    1: 'DTLZ1',
    2: 'DTLZ2',
    3: 'DTLZ3',
    4: 'DTLZ4',
    5: 'DTLZ5',
    6: 'DTLZ6',
    7: 'DTLZ7',
    11: 'ZDT1',
    12: 'ZDT2',
    13: 'ZDT3',
    14: 'ZDT4',
    15: 'ZDT5',
    16: 'ZDT6',
    21: 'WFG1',
    22: 'WFG2',
    23: 'WFG3',
    24: 'WFG4',
    25: 'WFG5',
    26: 'WFG6',
    27: 'WFG7',
    28: 'WFG8',
    29: 'WFG9'
}

def main():
    Path("result").mkdir(parents=False, exist_ok=True)
    
    for func_n in [21, 22, 23, 24, 25, 26, 27, 28, 29]:
        num_runs = 30

        objectives_dim = 3
        otimizations_type = [False] * objectives_dim
        max_iterations = 0
        max_fitness_eval = 15000
        position_dim = 10
        population_size = 100
        num_final_solutions = 50
        n_partitions = 12

        print(f"Running NSGA2 on {optimizationMap[func_n]}")

        result = {}
        combined = None
        for i in tqdm(range(num_runs)):
            nsga2 = NSGA2(pop_size=population_size)

            res = minimize(get_problem(optimizationMap[func_n], n_var=position_dim, n_obj=objectives_dim), nsga2, termination=('n_eval', max_fitness_eval))

            get_population = lambda p: p.X
            
            result[i+1] = {"F":res.F, "P":[get_population(p) for p in res.pop]}
            if combined is None:
                combined = res.F
            else:
                combined = np.vstack((combined, res.F))

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=combined)
        n = num_final_solutions
        if len(ndf[0]) < num_final_solutions:
            n = len(ndf[0])
        best_idx = pg.sort_population_mo(points = combined)[:n]
        result['combined'] = (best_idx, combined[best_idx])

        with open(f'result/NSGA2_{optimizationMap[func_n]}_{objectives_dim}obj.pkl', 'wb') as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    main()