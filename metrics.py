import numpy as np


def hypervolume(pareto_front, ref):
    ideal = np.absolute(ref[1] - pareto_front[0][1])
    h = 0

    for idx, point in enumerate(pareto_front):
        if(idx > 0):
            ideal = pareto_front[idx - 1][1] - point[1]
        
        h += np.absolute((ref[0] - point[0]) * ideal)
    
    return h

