import argparse
import os
import pickle

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.interpolate
import scipy.optimize
from numba import njit
from tqdm import tqdm

import fit_functions
import nekhoroshev_tools

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--index", type=int, required=True)
parser.add_argument("-k", "--kvalue", type=float, default=1.0)
parser.add_argument("-f", "--filename", type=str, required=True)
parser.add_argument("-l", "--label", type=str, required=True)
args = parser.parse_args()
idx = args.index
k_val = args.kvalue
filename = args.filename
label = args.label


with open(filename, "rb") as f:
    to_fit = pickle.load(f)

key = list(to_fit.keys())[idx]
print(label, key)
fit_x_list = to_fit[key][0]
fit_y_list = to_fit[key][1]

pars = lmfit.Parameters()
pars.add("I_star", value=10.0, min=0.0, max=200.0)
pars.add("k", value=0.33, min=0.0, max=k_val)
pars.add("c", value=1.0, vary=False)

fit_result = lmfit.minimize(
    fit_functions.resid_func, pars, args=(fit_x_list, fit_y_list)
)
fit_c1, fit_c2 = fit_functions.ana_current(fit_result.params, fit_x_list, fit_y_list)

with open(f"fit_result_{label}_{key}_single_kmax_{k_val}.pkl", "wb") as f:
    print("saved fit result", label)
    pickle.dump((fit_result, fit_c1, fit_c2, fit_x_list, fit_y_list), f)

print("Done!")

# pars = lmfit.Parameters()
# pars.add("I_star", value=10.0, min=0.0, max=200.0)
# pars.add("k", value=0.33, min=0.0, max=k_val)
# pars.add("c", value=1.0, vary=False)

# fit_result_list = []
# for i in tqdm(range(5, len(fit_x_list))):
#     fit_result = lmfit.minimize(
#         fit_functions.resid_func, pars, args=(fit_x_list[:i], fit_y_list[:i])
#     )
#     fit_c1, fit_c2 = fit_functions.ana_current(
#         fit_result.params, fit_x_list, fit_y_list
#     )
#     fit_result_list.append((fit_result, fit_c1, fit_c2))
#     pars = fit_result.params

#     with open(f"fit_result_{label}_{key}_iter_{i}_kmax_{k_val}.pkl", "wb") as f:
#         pickle.dump((fit_result, fit_c1, fit_c2), f)
