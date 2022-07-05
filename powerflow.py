import matlab.engine

eng = matlab.engine.start_matlab()

def RCM(x, n):
    result = eng.CEC2021_func(matlab.double(vector=x), n)
    return list(result[0].toarray())

def ZDT(x, n):
    result = eng.ZDT(matlab.double(vector=x), n)
    return list(result[0].toarray())