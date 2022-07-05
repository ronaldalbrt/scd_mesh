import numpy as np
import datetime
import pygmo as pg
from MESH import *
import pickle
from tqdm import tqdm
from pathlib import Path

def main():
    Path("result").mkdir(parents=False, exist_ok=True)

    rcm = 50
    num_final_solutions = 100
    num_runs = 30

    global_best_attribution_type = 1 #G
    Xr_pool_type = 1                 #V
    DE_mutation_type = 0             #M
    config = f"E{global_best_attribution_type + 1}V{Xr_pool_type + 1}D{DE_mutation_type + 1}"
    
    result = {}
    combined = None
    for i in tqdm(range(num_runs)):
        log_memory = f"result/_{config}_{i}-RCM{rcm}-"

        F = open(log_memory+"fit.txt", 'r').read().split("\n")[-2]
        F = np.array([v.split() for v in F.split(",")], dtype=np.float64)

        P = open(log_memory+"pos.txt", 'r').read().split("\n")[-2]
        P = np.array([v.split() for v in P.split(",")], dtype=np.float64)

        result[i+1] = {"F":F, "P":P}
        if combined is None:
            combined = F
        else:
            combined = np.vstack((combined, F))

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=combined)
    n = num_final_solutions
    if len(ndf[0]) < num_final_solutions:
        n = len(ndf[0])
    best_idx = pg.sort_population_mo(points = combined)[:n]
    result['combined'] = (best_idx, combined[best_idx])

    with open(f'result/_{config}_RCM{rcm}.pkl', 'wb') as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    main()