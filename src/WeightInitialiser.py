import numpy as np


def isWeightInitialiser(str):
    if str in ["constantWI",
               "constantZerosWI",
               "constantOnesWI",
               "uniformWI",
               "normalWI",
               "leCunUniformWI",
               "leCunNormalWI",
               "xavierUniformWI",
               "xavierNormalWI",
               "heUniformWI",
               "heNormalWI"]:
        return True
    return False

def constantWI(input_units, output_units, value=1):
    return np.ones(input_units, output_units) * value


def constantZerosWI(input_units, output_units):
    return constantWI(input_units, output_units, 0)


def constantOnesWI(input_units, output_units):
    return constantWI(input_units, output_units, 1)


def uniformWI(input_units, output_units):
    low=-0.05
    high=0.05
    np.random.seed(42)
    return np.random.uniform(low, high, size=(input_units, output_units))


def normalWI(input_units, output_units):
    mean=0.0
    std=0.05
    np.random.seed(42)
    return np.random.normal(mean, std, size=(input_units, output_units))


def leCunUniformWI(in_units, out_units):
    limit = np.sqrt(3 / float(in_units))
    np.random.seed(42)
    return np.random.uniform(low=-limit, high=limit, size=(in_units, out_units))


def leCunNormalWI(in_units, out_units):
    limit = np.sqrt(1 / float(in_units))
    np.random.seed(42)
    return np.random.normal(0.0, limit, size=(in_units, out_units))
    

def xavierUniformWI(in_units, out_units):
    limit = np.sqrt(6 / float(in_units + out_units))
    np.random.seed(42)
    return np.random.uniform(-limit, limit, size=(in_units, out_units))
    

def xavierNormalWI(in_units, out_units):
    limit = np.sqrt(2 / float(in_units + out_units))
    np.random.seed(42)
    return np.random.normal(0.0, limit, size=(in_units, out_units))


def heUniformWI(in_units, out_units):
    limit = np.sqrt(6 / float(in_units))
    np.random.seed(42)
    return np.random.uniform(low=-limit, high=limit, size=(in_units, out_units))


def heNormalWI(in_units, out_units):
    limit = np.sqrt(2 / float(in_units))
    np.random.seed(42)
    return np.random.normal(0.0, limit, size=(in_units, out_units))
