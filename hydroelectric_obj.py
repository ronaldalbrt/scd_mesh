import numpy as np


g = 9.8
k = 10**(-3)
a = [5.30e+02, 6.30e-03, -4.84e-07, 2.20e-11, -3.84e-16]
b = [5.15e+02, 1.61e-03, -2.55e-07, 2.89e-11, -1.18e-15] 
rho = [1.46e-01, 1.80e-02, 5.05e-03, -3.5205, -1.12e-03, -1.45e-05]


def ph_ujt(rho, hl, Qt, Hb_ut, DeltaH_ujt):
    rho0, rho1, rho2, rho3, rho4, rho5 = rho

    return g * k * (rho0 + rho1*hl + rho2*Qt + rho3*hl*Qt + rho4*(hl**2) + rho5*(Qt**2)) * (Hb_ut - DeltaH_ujt) * Qt

def hl_ujt(Hb_ut, DeltaH_ujt):
    return Hb_ut -  DeltaH_ujt

def Hb_ut(fcm, fcj):
    return fcm - fcj

def fcm_ut(a, Psi):
    a0, a1, a2, a3, a4 = a
    
    return a0 + a1*Psi + a2*(Psi**2) + a3*(Psi**3) + a4*(Psi**4)

def fcj_ut(b, Qt_overJ, Qv):
    b0, b1, b2, b3, b4 = b
    
    return b0 + b1*(Qt_overJ + Qv) + b2*((Qt_overJ + Qv)**2) + b3*((Qt_overJ + Qv)**3) + b4*((Qt_overJ + Qv)**4) 

def Psi(Psi_ant, Qa_ut, Qt_wtd, Qv_wtd, Qt_ut1, Qv_ut1, E_ut1, A_ut1):
    return Psi_ant + Qa_ut + Qt_wtd + Qv_wtd - Qt_ut1 + Qv_ut1 - (E_ut1 * A_ut1)

def F1(Qt_m, Psi_m, p, rho, a, b, DeltaH, Qv):
    U = len(Qt_m)

    Qt_sum = [np.sum(Qt_m[u]) for u in range(U)]
    Hb = [Hb_ut(fcm_ut(a, Psi_m[u]), fcj_ut(b, Qt_sum[u], Qv)) for u in range(U)]
    hl = [[hl_ujt(Hb[u], DeltaH[u][j]) for j in range(len(Qt_m[u]))] for u in range(U)]

    ph_ujt_sum = [np.sum([ph_ujt(rho, hl[u][j], Qt_m[u][j], Hb[u], DeltaH) for j in range(len(Qt_m[u]))]) for u in range(U)]
    
    return p * np.sum([np.max(0, ph_ujt_sum[u]/Qt_sum[u])**2 for u in range(U)])

def F2(Psi_m, Psi_max, p):
    U = len(Psi_m)
    
    return p * np.sum([np.max(0, Psi_m[u]/Psi_max)**2 for u in range(U)])

def calculate_UWD_U1(h):
    UWD = [
        705.81,
        705.81,
        705.81,
        686.02,
        686.02,
        705.81,
        705.81,
        705.81,
        662.67,
        662.67,
        662.67,
        662.67,
        662.67,
        662.67,
        662.67,
        662.67,
        662.67,
        662.67,
        898.30,
        934.66,
        943.66,
        934.66,
        934.66,
        951.77
    ]

    return UWD[h]

def calculate_UWD_U2(h):
    UWD = [
        572.69,
        572.51,
        572.86,
        560.38,
        560.33,
        572.86,
        572.68,
        572.68,
        521.09,
        507.77,
        507.77,
        507.77,
        507.77,
        521.09,
        507.77,
        521.09,
        507.77,
        507.77,
        727.85,
        744.07,
        744.04,
        743.92,
        744.05,
        757.31,
    ]

    return UWD[h]


def F3(Qt_m, h, calculate_UWD):
    U =  len(Qt_m)

    UWD = calculate_UWD(h)
    return np.sum([np.sqrt(np.sum([(Qt_m[u] - UWD)**2 for j in range(len(Qt_m[u]))])) for u in range(U)])
