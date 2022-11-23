import datetime
import pickle
import time
from objectives import *
from MESH import *
from tqdm import tqdm


objectives_dim = 3
otimizations_type = [False,False,False]
max_iterations = 0
max_fitness_eval = 15000
position_dim = 14
position_max_value = [140] * 14
position_min_value = [70] * 14
population_size = 50
memory_size = 50
memory_update_type = 0
communication_probability = 1.1
mutation_rate = 0.8
personal_guide_array_size = 3

sys.argv.append("a")
sys.argv.append("1")
sys.argv.append("23")

if int(sys.argv[2]) == 0:
    global_best_attribution_type = 0  # G
    Xr_pool_type = 0  # V
    DE_mutation_type = 0  # M
    config = "G0V0M0"
    config_dir = "E1V1D1"
elif int(sys.argv[2]) == 1:
    global_best_attribution_type = 1 #G
    Xr_pool_type = 1                 #V
    DE_mutation_type = 0             #M
    config = "G1V1M0"
    config_dir = "E2V2D1"

for hora in tqdm(range(0, 24)):
    if hora > 0:
        #path_last_hour = "/home/loliveira/Cascata-"+config_dir+"/"+str(hora-1)+"/pickles/next_initial_state.pickle"
        #path_last_hour = "D:\\Dropbox\\Mestrado\\Dissertação\\Python\\MC-DEEPSO\\Cascata-E2V2D1\\11\\pickles\\next_initial_state.pickle"
        #path_last_hour = "D:\\Dropbox\\Mestrado\\Dissertação\\Python\\MC-DEEPSO\\Cascata-E2V2D1\\11\\pickles\\next_initial_state.pickle"
        path_last_hour = config+"-Cascata_"+str(hora - 1)+".pickle"
        file_last_hour = open(path_last_hour,'rb')
        last_hour = pickle.load(file_last_hour)

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
        volumes.append(initial_state[0][0])
        volumes.append(initial_state[0][1])
        afluentes.append(afluentes_horas[hora][0])
        afluentes.append(afluentes_horas[hora][1])
        defluentes.append(initial_state[2][0])
        defluentes.append(initial_state[2][1])
        defluentes.append(initial_state[2][2])
        defluentes.append(initial_state[2][3])
        demandas.append(demandas_horas[hora][0])
        demandas.append(demandas_horas[hora][1])
        UWD.append(UWD_horas[hora][0])
        UWD.append(UWD_horas[hora][1])

    initial_state = []
    initial_state.append(volumes)
    initial_state.append(afluentes)
    initial_state.append(defluentes)
    initial_state.append(demandas)
    initial_state.append(UWD)

    params = MESH_Params(objectives_dim,otimizations_type,max_iterations,max_fitness_eval,position_dim,position_max_value,position_min_value,population_size,memory_size,memory_update_type,global_best_attribution_type,DE_mutation_type,Xr_pool_type,communication_probability,mutation_rate,personal_guide_array_size,True,initial_state)

    time_string = str(datetime.datetime.now()).replace(":","-")

    MCDEEPSO = MESH(params,cascata)
    #MCDEEPSO.log_memory = "/home/loliveira/Cascata-"+config_dir+"/"+str(hora)+"/"+config+"-Cascata_"+str(hora)+"-"+str(sys.argv[1])+"-" + time_string
    #MCDEEPSO.log_memory = "D:\\Dropbox\\Mestrado\\Dissertação\\Python\\MC-DEEPSO\\Cascata-E2V2D1\\12\\"+config+"-Cascata_"+str(hora)+"-"+str(sys.argv[1])+"-" + time_string
    MCDEEPSO.log_memory = config+"-Cascata_"+str(hora)+"_"

    #MCDEEPSO.copy_pop = False
    start = time.time()
    MCDEEPSO.run()
    end = time.time()
    tempo = end - start
    print(tempo/60)

    export_memory = MCDEEPSO.memory

    #picklefile_name = "/home/ronald/Cascata-"+config_dir+"/"+str(hora)+"/pickles/"+config+"-Cascata_"+str(hora)+"-"+str(sys.argv[1])+"-" + time_string +"--" + str(MCDEEPSO.fitness_eval_count) + "--.pickle"
    picklefile_name = config+"-Cascata_"+str(hora) + ".pickle"
    picklefile = open(picklefile_name,'wb')
    pickle.dump(export_memory,picklefile)
    picklefile.close()

print("fim")