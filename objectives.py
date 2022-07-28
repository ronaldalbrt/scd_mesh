###########################################################################
# Lucas Braga, MS.c. (email: lucas.braga.deo@gmail.com )
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# Carolina Marcelino, PhD (email: carolimarc@ic.ufrj.br)
# June 16, 2021
###########################################################################
# Copyright (c) 2021, Lucas Braga, Gabriel Matos Leite, Carolina Marcelino
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS USING 
# THE CREATIVE COMMONS LICENSE: CC BY-NC-ND "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from deap import benchmarks
from pymoo.factory import get_problem

def get_function(number, n_dim = 3):
    if number == 1:
        return lambda x: DTLZ1(x, n_dim)
    if number == 2:
        return lambda x: DTLZ2(x, n_dim)
    if number == 3:
        return lambda x: DTLZ3(x, n_dim)
    if number == 4:
        return lambda x: DTLZ4(x, n_dim)
    if number == 5:
        return lambda x: DTLZ5(x, n_dim)
    if number == 6:
        return lambda x: DTLZ6(x, n_dim)
    if number == 7:
        return lambda x: DTLZ7(x, n_dim)
    if number == 11:
        return ZDT1
    if number == 12:
        return ZDT2
    if number == 13:
        return ZDT3
    if number == 14:
        return ZDT4
    if number == 15:
        return ZDT5
    if number == 16:
        return ZDT6
    if number == 21:
        return lambda x: WFG1(x, n_dim)
    if number == 22:
        return lambda x: WFG2(x, n_dim)
    if number == 23:
        return lambda x: WFG3(x, n_dim)
    if number == 24:
        return lambda x: WFG4(x, n_dim)
    if number == 25:
        return lambda x: WFG5(x, n_dim)
    if number == 26:
        return lambda x: WFG6(x, n_dim)
    if number == 27:
        return lambda x: WFG7(x, n_dim)
    if number == 28:
        return lambda x: WFG8(x, n_dim)
    if number == 29:
        return lambda x: WFG9(x, n_dim)
    return None

def rastrigin(x):
    sum_func = np.vectorize(lambda x: (x - .5)**2 - np.cos(20*np.pi*(x - .5)))

    return 100*(np.linalg.norm(x) + np.sum(sum_func(x)))

def WFG1(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("wfg1", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def WFG2(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("wfg2", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def WFG3(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("wfg3", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def WFG4(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("wfg4", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def WFG5(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("wfg5", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def WFG6(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("wfg6", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def WFG7(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("wfg7", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def WFG8(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("wfg8", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def WFG9(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("wfg9", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def DTLZ1(x, n_dim = 2):
    
    return benchmarks.dtlz1(x, n_dim)

def DTLZ2(x, n_dim = 2):
    
    return benchmarks.dtlz2(x, n_dim)

def DTLZ3(x, n_dim = 2): 
    return benchmarks.dtlz3(x, n_dim)

def DTLZ4(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("dtlz4", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def DTLZ5(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("dtlz5", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def DTLZ6(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("dtlz6", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def DTLZ7(x, n_dim = 2):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("dtlz7", n_var=x_dim, n_obj=n_dim).evaluate(np_x)

def ZDT1(x):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("zdt1", n_var=x_dim).evaluate(np_x)

def ZDT2(x):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("zdt2", n_var=x_dim).evaluate(np_x)
    
def ZDT3(x):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("zdt3", n_var=x_dim).evaluate(np_x)

def ZDT4(x):
    sum = 0
    for i in range(len(x)-1):
        sum += np.power(x[i+1],2) - 10*np.cos(4*np.pi*x[i+1])
    Gx = 1 + 10*(len(x)-1) + sum
    F1 = x[0]
    F2 = Gx * (1 - np.sqrt(F1/Gx))
    Fitness = []
    Fitness.append(F1)
    Fitness.append(F2)
    return Fitness

def ZDT5(x):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    zdt5 = get_problem("zdt5", n_var=x_dim)
    return zdt5.evaluate(np_x)

def ZDT6(x):
    np_x = np.array(x)
    x_dim = np_x.shape[0]
    
    return get_problem("zdt6", n_var=x_dim).evaluate(np_x)
