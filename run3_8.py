import numpy as np
import datetime
from powerflow import *
from MESH import *
from tqdm import tqdm
from pathlib import Path


def main():
    Path("result").mkdir(parents=False, exist_ok=True)

    num_runs = 30
    rcm = 24
    rcm_func = lambda x: RCM(x, rcm)

    objectives_dim = 3
    otimizations_type = [False,False]
    max_iterations = 0
    max_fitness_eval = 15000
    position_dim = 9
    position_max_value = [10, 200, 100, 200, 2000000, 600, 600, 600, 900]
    position_min_value = [0, 0, 0, 0, 1000, 0, 100, 100, 100]
    population_size = 100
    memory_size = 100
    memory_update_type = 0

    communication_probability = 0.7 #0.5
    mutation_rate = 0.9
    personal_guide_array_size = 3

    global_best_attribution_type = 1 #G
    Xr_pool_type = 1                 #V
    DE_mutation_type = 0             #M
    config = f"E{global_best_attribution_type + 1}V{Xr_pool_type + 1}D{DE_mutation_type + 1}"
    
    print(f"Running E{global_best_attribution_type+1}V{Xr_pool_type+1}D{DE_mutation_type+1} on RCM{rcm}")

    for i in tqdm(range(num_runs)):

        params = MESH_Params(objectives_dim,otimizations_type,max_iterations,max_fitness_eval,position_dim,position_max_value,position_min_value,population_size,memory_size,memory_update_type,global_best_attribution_type,DE_mutation_type,Xr_pool_type,communication_probability,mutation_rate,personal_guide_array_size)

        MCDEEPSO = MESH(params,rcm_func)
        MCDEEPSO.log_memory = f"result/_{config}_{i}-RCM{rcm}-"
        MCDEEPSO.run()

if __name__ == '__main__':
    main()