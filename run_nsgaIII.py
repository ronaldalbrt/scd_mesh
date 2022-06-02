from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
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
    21: 'ZDT2',
    31: 'ZDT3',
    41: 'ZDT4',
    51: 'ZDT5',
    61: 'ZDT6'
}

def main():
    Path("result").mkdir(parents=False, exist_ok=True)
    
    for func_n in [1, 2, 3, 4, 5, 6, 7]:
        num_runs = 30

        objectives_dim = 3
        otimizations_type = [False] * objectives_dim
        max_iterations = 0
        max_fitness_eval = 15000
        position_dim = 5
        population_size = 100
        num_final_solutions = 50
        n_partitions = 12

        print(f"Running NSGA3 on {optimizationMap[func_n]}")

        result = {}
        combined = None
        for i in tqdm(range(num_runs)):
            ref_dirs = get_reference_directions("das-dennis", objectives_dim, n_partitions=n_partitions)
            nsga3 = NSGA3(pop_size=population_size, ref_dirs=ref_dirs)

            res = minimize(get_problem(optimizationMap[func_n], n_var=position_dim, n_obj=objectives_dim), nsga3, seed=1, termination=('n_gen', 600))

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

        with open(f'result/NSGA3_{optimizationMap[func_n]}_{objectives_dim}obj.pkl', 'wb') as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    main()