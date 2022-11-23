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


def cascata(input):
    ## Input:
    # Array 0 -> vazoes turbinadas | 0 - 7 > U1 || 8 - 13 > U2
    # Array 1 -> volumes | 0 > U1 || 1 > U2
    # Array 2 -> vazoes afluentes | 0 > U1 || 1 > U2
    # Array 3 -> vazoes defluentes | 0-1 > U1 || 2-3 > U2
    #                               [0] 1 hr atras -> [1] 2 hrs atras)
    # Array 4 -> demandas | 0 -> U1 || 1 -> U2
    # Array 5 -> vazão padrão da hidrelétrica | 0 -> U1 || 1 -> U2

    vazao_U1 = [] # input[0][0] + input[0][1] + input[0][2] + input[0][3] + input[0][4] + input[0][5] + input[0][6] + input[0][7]
    for i in range(8):
        vazao_U1.append(input[0][i])
    vazao_U2 = [] # input[0][8] + input[0][9] + input[0][10] + input[0][11] + input[0][12] + input[0][13]
    for i in range(6):
        vazao_U2.append(input[0][i+7])

    volume_U1 = input[1][0]
    volume_U2 = input[1][1]

    Qa_U1 = input[2][0]
    Qa_U2 = input[2][1]

    Qd_U1 = input[3][1]
    Qd_U2 = input[3][1]

    Qv_U1 = 0
    Qv_U2 = 0

    D_U1 = input[4][0]
    D_U2 = input[4][1]

    UWD_U1 = input[5][0]
    UWD_U2 = input[5][1]

    c_m3s_hm3 = 0.0036 # m^3/s^-1 para hm^3 em 1 hr
    V_max = 19528
    V_min = 4250

    ## Calculo Volumes
    #############################
    volume_U1 = volume_U1 + (c_m3s_hm3 * (Qa_U1))
    volume_U2 = volume_U2 + (c_m3s_hm3 * (Qa_U2 + Qd_U2))

    if volume_U1 > V_max:
        Qv_U1 = (V_max - volume_U1) * (1/c_m3s_hm3)

    if volume_U2 > V_max:
        Qv_U2 = (V_max - volume_U2) * (1/c_m3s_hm3)

    volume_U1 = volume_U1 - (c_m3s_hm3 * (Qv_U1 + np.sum(vazao_U1)))
    volume_U2 = volume_U2 - (c_m3s_hm3 * (Qv_U2 + np.sum(vazao_U2)))
    #############################

    ## Calculo cotas e queda bruta
    #############################
    Qd_U1_saida = np.sum(vazao_U1) + Qv_U1
    Qd_U2_saida = np.sum(vazao_U2) + Qv_U2

    montante1 = 5.30*(10**2) + volume_U1*6.08*(10**(-3)) - (np.power(volume_U1,2))*4.84*(10**(-7)) + (np.power(volume_U1,3))*2.20*(10**(-11)) - (np.power(volume_U1,4))*3.85*(10**(-16))
    montante2 = 5.30*(10**2) + volume_U2*6.08*(10**(-3)) - (np.power(volume_U2,2))*4.84*(10**(-7)) + (np.power(volume_U2,3))*2.20*(10**(-11)) - (np.power(volume_U2,4))*3.85*(10**(-16))

    jusante1 = 5.15*(10**2) + Qd_U1_saida*1.61*(10**(-3)) - (np.power(Qd_U1_saida,2))*2.55*(10**(-7)) + (np.power(Qd_U1_saida,3))*2.89*(10**(-11)) - (np.power(Qd_U1_saida,4))*1.18*(10**(-15))
    jusante2 = 5.15*(10**2) + Qd_U2_saida*1.61*(10**(-3)) - (np.power(Qd_U2_saida,2))*2.55*(10**(-7)) + (np.power(Qd_U2_saida,3))*2.89*(10**(-11)) - (np.power(Qd_U2_saida,4))*1.18*(10**(-15))

    quedaU1 = montante1 - jusante1
    quedaU2 = montante2 - jusante2
    #############################

    #constantes
    pi = 3.14
    g = 9.8
    l1 = 160.0
    l2a = 91.6
    l2b = 86.26
    l2c = 82.54
    l2d = 80.58
    l3 = 13.4
    d1 = 6.6
    d2 = 6.2
    r = 70000
    k = 0.2

    # CALCULANDO RUGOSIDADE TOTAL
    e1 = k / d1
    e2 = k / d2

    # CALCULANDO A AREA (m^2)
    a1 = pi * (np.power(d1, 4) / 4)
    a2 = pi * (np.power(d2, 4) / 4)

    # CALCULANDO FATOR DE ATRITO F (MAIOR SECAO)
    a = np.power((64 / r), 8)
    b = e1 / (3.7 * d1)
    c = (5.74 / np.power(r, 0.9)) * -1
    d = 2500 / (-1.0 * r)
    potd = np.power(d, 6)
    aux = b + c
    caln = np.log(aux)
    parteb = 9.5 * (np.power((caln - potd), -16))
    fatorf1 = np.power((a + parteb), 0.125)

    # CALCULANDO FATOR DE ATRITO F (MENOR SECAO)
    a22 = np.power((64 / r), 8)
    b22 = e2 / (3.7 * d2)
    c2 = (5.74 / np.power(r, 0.9)) * -1
    d22 = 2500 / (-1.0 * r)
    potd2 = np.power(d22, 6)
    aux2 = b22 + c2
    caln2 = np.log(aux2)
    parteb2 = 9.5 * (np.power((caln2 - potd2), -16))
    fatorf2 = np.power((a22 + parteb2), 0.125)

    # CALCULANDO FATOR DE ATRITO F TOTAL
    ftotal = fatorf1 + fatorf2

    ca1 = 0.080 + (0.080 * 0.03)  # Fator K para curva de 28
    ca2 = 0.100 + (0.100 * 0.03)  # Fator K para curva de 30

    cb1 = 0.030 + (0.030 * 0.03)  # Fator K para curva de 22
    cb2 = 0.020 + (0.020 * 0.03)  # Fator K para curva de 21

    cc1 = 0.051 + (0.051 * 0.03)  # Fator K para curva de 16
    cc2 = 0.047 + (0.047 * 0.02)  # Fator K para curva de 12

    cd1 = 0.0120 + (0.012 * 0.02)  # Fator K para curva de 4
    cd2 = 0.0118 + (0.0118 * 0.02)  # Fator K para curva de 3

    vm = np.zeros(14)

    p1a = np.zeros(14)
    p1b = np.zeros(14)
    p2a = np.zeros(14)
    p2b = np.zeros(14)
    p3a = np.zeros(14)
    p3b = np.zeros(14)
    p4a = np.zeros(14)
    p4b = np.zeros(14)
    p5a = np.zeros(14)
    p5b = np.zeros(14)
    p6a = np.zeros(14)
    p6b = np.zeros(14)
    p7a = np.zeros(14)
    p7b = np.zeros(14)

    pt1 = np.zeros(14)
    pt2 = np.zeros(14)
    pt3 = np.zeros(14)
    pt4 = np.zeros(14)
    pt5 = np.zeros(14)
    pt6 = np.zeros(14)
    pt7 = np.zeros(14)

    pcond18 = np.zeros(14)
    pcond27 = np.zeros(14)
    pcond36 = np.zeros(14)
    pcond45 = np.zeros(14)

    pc1 = np.zeros(14)
    pc2 = np.zeros(14)
    pc3 = np.zeros(14)
    pc4 = np.zeros(14)

    perda1 = np.zeros(14)
    perda2 = np.zeros(14)
    perda3 = np.zeros(14)
    perda4 = np.zeros(14)

    potencia = np.zeros(14)
    hl = 0
    potencia_total = 0

    #### POTENCIA USINA 1###
    for i in range(14):
        ## VM
        if i > 7:
            vm[i] = ((vazao_U2[i-8] / a1) + (vazao_U2[i-8] / a2)) / 2
        else:
            vm[i] = ((vazao_U1[i] / a1) + (vazao_U1[i] / a2)) / 2

        ## CALCULANDO PERDAS DE CARGA TUBO
        p1a[i] = ftotal * ((l1 / d1) * ((np.power(vm[i], 2) / 2) * g))
        p1b[i] = ftotal * ((l1 / d2) * ((np.power(vm[i], 2) / 2) * g))

        p2a[i] = ftotal * ((l1 / d1) * ((np.power(vm[i], 2) / 2) * g))
        p2b[i] = ftotal * ((l1 / d2) * ((np.power(vm[i], 2) / 2) * g))

        p3a[i] = ftotal * ((l2a / d1) * ((np.power(vm[i], 2) / 2) * g))
        p3b[i] = ftotal * ((l2a / d2) * ((np.power(vm[i], 2) / 2) * g))

        p4a[i] = ftotal * ((l2b / d1) * ((np.power(vm[i], 2) / 2) * g))
        p4b[i] = ftotal * ((l2b / d2) * ((np.power(vm[i], 2) / 2) * g))

        p5a[i] = ftotal * ((l2c / d1) * ((np.power(vm[i], 2) / 2) * g))
        p5b[i] = ftotal * ((l2c / d2) * ((np.power(vm[i], 2) / 2) * g))

        p6a[i] = ftotal * ((l2d / d1) * ((np.power(vm[i], 2) / 2) * g))
        p6b[i] = ftotal * ((l2d / d2) * ((np.power(vm[i], 2) / 2) * g))

        p7a[i] = ftotal * ((l3 / d1) * ((np.power(vm[i], 2) / 2) * g))
        p7b[i] = ftotal * ((l3 / d2) * ((np.power(vm[i], 2) / 2) * g))

        pt1[i] = p1a[i] + p1b[i]
        pt2[i] = p2a[i] + p2b[i]
        pt3[i] = p3a[i] + p3b[i]
        pt4[i] = p4a[i] + p4b[i]
        pt5[i] = p5a[i] + p5b[i]
        pt6[i] = p6a[i] + p6b[i]
        pt7[i] = p7a[i] + p7b[i]

        pcond18[i] = pt1[i] + pt2[i] + pt6[i] + pt7[i]
        pcond27[i] = pt1[i] + pt3[i] + pt6[i] + pt7[i]
        pcond36[i] = pt1[i] + pt4[i] + pt6[i] + pt7[i]
        pcond45[i] = pt1[i] + pt5[i] + pt6[i] + pt7[i]

        ## CALCULANDO PERDAS NAS CURVAS POR CONDUTOS

        pc1[i] = (ca1 * (np.power(vm[i], 2) / (2 * g))) + (ca2 * (np.power(vm[i], 2) / (2 * g)))
        pc2[i] = (cb1 * (np.power(vm[i], 2) / (2 * g))) + (cb2 * (np.power(vm[i], 2) / (2 * g)))
        pc3[i] = (cc1 * (np.power(vm[i], 2) / (2 * g))) + (cc2 * (np.power(vm[i], 2) / (2 * g)))
        pc4[i] = (cd1 * (np.power(vm[i], 2) / (2 * g))) + (cd2 * (np.power(vm[i], 2) / (2 * g)))

        perda1[i] = pcond18[i] + pc1[i]
        perda2[i] = pcond27[i] + pc2[i]
        perda3[i] = pcond36[i] + pc3[i]
        perda4[i] = pcond45[i] + pc4[i]

        if i > 7:
            queda = quedaU2
        else:
            queda = quedaU1

        if i == 0 or i == 1:
            hl = queda - perda1[i]
        if i == 2 or i == 3:
            hl = queda - perda2[i]
        if i == 4 or i == 5:
            hl = queda - perda3[i]
        if i == 6 or i == 7:
            hl = queda - perda4[i]

        if i == 8:
            hl = queda - perda1[i]
        if i == 9:
            hl = queda - perda2[i]
        if i == 10 or i == 11:
            hl = queda - perda3[i]
        if i == 12 or i == 13:
            hl = queda - perda4[i]


        # if (i == 0):
        #     ## saida_perdas[i] = perda1[i]
        #     hl = 54 - perda1[i]
        #
        # elif (i == 1):
        #     ## saida_perdas[i] = perda2[i]
        #     hl = 54 - perda2[i]
        #
        # elif (i == 2):
        #     ## saida_perdas[i] = perda3[i]
        #     hl = 54 - perda3[i]
        #
        # elif (i == 3):
        #     ## saida_perdas[i] = perda4[i]
        #     hl = 54 - perda4[i]
        #
        # elif (i == 4):
        #     ## saida_perdas[i] = perda4[i]
        #     hl = 54 - perda4[i]
        #
        # elif (i == 5):
        #     ## saida_perdas[i] = perda3[i]
        #     hl = 54 - perda3[i]

        if i > 7:
            potencia[i] = 0.0098 * (0.1463 + 0.018076 * hl + 0.0050502 * vazao_U2[i-8] - 0.000035254 * hl * vazao_U2[i-8] - 0.00012337 * np.power(hl,2) - 0.000014507 * np.power(vazao_U2[i-8], 2)) * hl * vazao_U2[i-8]
        else:
            potencia[i] = 0.0098 * (0.1463 + 0.018076 * hl + 0.0050502 * vazao_U1[i] - 0.000035254 * hl * vazao_U1[i] - 0.00012337 * np.power(hl,2) - 0.000014507 * np.power(vazao_U1[i], 2)) * hl * vazao_U1[i]
        potencia_total += potencia[i]
    ## penalidades
    Lambda = 1

    if abs(D_U1 - np.sum(potencia[:8])) < D_U1 * 0.005:
        FU1 = 0
    else:
        FU1 = Lambda * max(0, (np.power((D_U1 - np.sum(potencia[:8])), 2)))

    if abs(D_U2 - np.sum(potencia[8:])) < D_U2 * 0.005:
        FU2 = 0
    else:
        FU2 = Lambda * max(0, (np.power((D_U2 - np.sum(potencia[8:])), 2)))

    f1 = -1 * np.sum(potencia[:8]) + FU1
    f1 = f1 / np.sum(vazao_U1)

    f2 = -1 * np.sum(potencia[8:]) + FU2
    f2 = f2 / np.sum(vazao_U2)

    obj1 = (f1 + f2)/2
    obj2 = -1 * (((volume_U1/V_max) + (volume_U2/V_max)) / 2 - (FU1+FU2)/10)
    #obj2 = np.sum(vazao_U1) + np.sum(vazao_U2)
    obj3_U1 = np.sqrt(np.sum([(vazao_U1[i] - UWD_U1)**2 for i in range(len(vazao_U1))]))/np.sqrt(np.sum([(vazao_U1[i])**2 for i in range(len(vazao_U1))]))
    obj3_U2 = np.sqrt(np.sum([(vazao_U2[i] - UWD_U2)**2 for i in range(len(vazao_U2))]))/np.sqrt(np.sum([(vazao_U2[i])**2 for i in range(len(vazao_U2))]))
    obj3 = (obj3_U1 + obj3_U2)/2

    #c = 109.175
    #f2 = np.sqrt(np.power((vazoes[0] - c), 2) + np.power((vazoes[1] - c), 2) + np.power((vazoes[2] - c), 2) + np.power((vazoes[3] - c), 2) + np.power((vazoes[4] - c), 2) + np.power((vazoes[5] - c), 2))

    resultados = []

    objetivos = []
    objetivos.append(obj1)
    objetivos.append(obj2)
    objetivos.append(obj3)

    volumes = []
    volumes.append(volume_U1)
    volumes.append(volume_U2)

    defluencias = []
    defluencias.append(Qd_U1_saida)
    defluencias.append(input[3][0])
    defluencias.append(Qd_U2_saida)
    defluencias.append(input[3][2])

    resultados.append(objetivos)
    resultados.append(volumes)
    resultados.append(defluencias)

    return resultados