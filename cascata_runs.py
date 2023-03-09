import datetime
import pickle
import time
from objectives import *
from MESH import *
from tqdm import tqdm
import pygmo as pg

num_runs = 30
objectives_dim = 3
otimizations_type = [False,False,False]
max_iterations = 0
max_fitness_eval = 15000
position_dim = 14
position_max_value = [140] * 14
position_min_value = [70] * 14
population_size = 100
memory_size = 100
memory_update_type = 0
communication_probability = 1
mutation_rate = 3
personal_guide_array_size = 3
num_final_solutions = 100

sys.argv.append("a")
sys.argv.append("1")
sys.argv.append("23")

if int(sys.argv[2]) == 0:
    global_best_attribution_type = 0  # G
    Xr_pool_type = 0  # V
    DE_mutation_type = 0  # M
    crowd_distance_type = 0             #C
    config = "G0V0M0"
    config_dir = "E1V1D1"
elif int(sys.argv[2]) == 1:
    global_best_attribution_type = 1 #G
    Xr_pool_type = 1                 #V
    DE_mutation_type = 0             #M
    crowd_distance_type = 0             #C
    config = "G1V1M0"
    config_dir = "E2V2D1"
elif int(sys.argv[2]) == 2:
    global_best_attribution_type = 1 #G
    Xr_pool_type = 1                 #V
    DE_mutation_type = 0             #M
    crowd_distance_type = 1             #C
    config = "G1V1M0"
    config_dir = "E2V2D1C1"


for hora in tqdm(range(0, 24)):
    afluentes_horas = [[102.83,208.30],[102.65,289.21],[103.22,297.96],[105.00,192.64],[221.29,201.45],[223.18,204.59],[110.80,218.98],[227.00,227.18],[114.06,343.05],[228.11,343.05],[227.75,228.87],[227.45,235.15],[341.35,228.87],[341.00,114.85],[340.62,114.86],[340.30,114.87],[339.41,114.81],[227.74,114.80],[236.17,230.89],[245.00,229.30],[253.80,115.34],[376.80,115.33],[376.81,109.53],[385.52,223.95]]
    demandas_horas = [[330, 264], [330, 264], [330, 264], [322, 258], [322, 258], [330, 264], [330, 264], [330, 264], [300, 240], [292, 234], [292, 234], [292, 234], [292, 234], [300, 240], [292, 234], [300, 240], [292, 234], [292, 234], [420, 336], [437, 343], [437, 343], [437, 343], [437, 343], [445, 349]]
    UWD_horas = [[705.81, 573.59], [705.81, 573.59], [705.81, 573.59], [686.02, 560.56], [686.02, 560.56], [705.81, 573.59], [705.81, 573.59], [705.81, 573.59], [662.67, 521.45], [662.67, 508.41], [662.67, 508.41], [662.67, 508.41], [662.67, 508.41], [662.67, 521.45], [662.67, 508.41], [662.67, 521.45], [662.67, 508.41], [662.67, 508.41], [898.30, 730.03], [934.66, 745.24], [943.66, 745.24], [934.66, 745.24], [934.66, 745.24], [951.77, 758.27]]

    volumes = []
    afluentes = []
    defluentes = []
    demandas = []
    UWD = []

    if hora == 0:
        volumes.append(np.ceil(19528 * 0.80))
        volumes.append(np.ceil(19528 * 0.80))
        afluentes.append(afluentes_horas[hora][0])
        afluentes.append(afluentes_horas[hora][1])
        defluentes.append(0)
        defluentes.append(0)
        defluentes.append(0)
        defluentes.append(0)
        demandas.append(demandas_horas[hora][0])
        demandas.append(demandas_horas[hora][1])
        UWD.append(UWD_horas[hora][0])
        UWD.append(UWD_horas[hora][1])
    else:
        volumes.append(curr_state[0][0])
        volumes.append(curr_state[0][1])
        afluentes.append(afluentes_horas[hora][0])
        afluentes.append(afluentes_horas[hora][1])
        defluentes.append(curr_state[2][0])
        defluentes.append(curr_state[2][1])
        defluentes.append(curr_state[2][2])
        defluentes.append(curr_state[2][3])
        demandas.append(demandas_horas[hora][0])
        demandas.append(demandas_horas[hora][1])
        UWD.append(UWD_horas[hora][0])
        UWD.append(UWD_horas[hora][1])

    curr_state = []
    curr_state.append(volumes)
    curr_state.append(afluentes)
    curr_state.append(defluentes)
    curr_state.append(demandas)
    curr_state.append(UWD)

    print(f"Running {config_dir} on UHE{hora}h")

    result = {}
    combined = None
    for i in tqdm(range(num_runs)):
        params = MESH_Params(objectives_dim,otimizations_type,max_iterations,max_fitness_eval,position_dim,position_max_value,position_min_value,population_size,memory_size,memory_update_type,global_best_attribution_type,DE_mutation_type,Xr_pool_type, crowd_distance_type,communication_probability,mutation_rate,personal_guide_array_size,True,curr_state)
        MCDEEPSO = MESH(params,cascata)
        MCDEEPSO.log_memory = "result/intermediate_results/"+config_dir+"-Cascata_"+str(hora)+"_"
        MCDEEPSO.run()
        
        F = open(MCDEEPSO.log_memory+"fit.txt", 'r').read().split("\n")[-2]
        F = np.array([v.split() for v in F.split(",")], dtype=np.float64)

        P = open(MCDEEPSO.log_memory+"pos.txt", 'r').read().split("\n")[-2]
        P = np.array([v.split() for v in P.split(",")], dtype=np.float64)

        result[i+1] = {"F":F, "P":P}
        if combined is None:
            combined = F
            combined_decision = P
        else:
            combined = np.vstack((combined, F))
            combined_decision = np.vstack((combined_decision, P))

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=combined)
    n = num_final_solutions
    if len(ndf[0]) < num_final_solutions:
        n = len(ndf[0])
    best_idx = pg.sort_population_mo(points = combined)[:n]
    result['combined'] = (best_idx, combined[best_idx])

    results = cascata([combined_decision[best_idx[int(len(best_idx)/2)]]] + curr_state)
    curr_state[0] = results[1]
    curr_state[2] = results[2]

    with open(f'result/{config_dir}_UHE{hora}h_{objectives_dim}obj.pkl', 'wb') as f:
        pickle.dump(result, f)

print("fim")