
import numpy as np
import pygmo as pg
import pickle
from pathlib import Path


def main():
    Path("result").mkdir(parents=False, exist_ok=True)
    rcm = 50

    objectives_dim = 3
    otimizations_type = [False] * objectives_dim
    max_iterations = 0
    max_fitness_eval = 15000
    position_dim = 5
    population_size = 100
    num_final_solutions = 50
    n_partitions = 12

    with open(f'result/NSGA3_RCM{rcm}.pkl', 'rb') as f:
        result = pickle.load(f)
    
    combined = result["combined"]

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=combined)
    n = num_final_solutions
    if len(ndf[0]) < num_final_solutions:
        n = len(ndf[0])
    best_idx = pg.sort_population_mo(points = combined)[:n]
    result['combined'] = (best_idx, combined[best_idx])

    with open(f'result/NSGA3_RCM{rcm}.pkl', 'wb') as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    main()