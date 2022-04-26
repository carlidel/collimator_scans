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

# create parser
parser = argparse.ArgumentParser(
    description="This script is used to find the best fit for the interpolation of the data."
)
parser.add_argument("-c", help="c value", required=True, type=float)
parser.add_argument(
    "-type", help="type of fit", required=True, choices=["single", "iterative"]
)
parser.add_argument(
    "-interpolation",
    help="interpolation type",
    required=True,
    choices=["spline", "fit", "basic"],
)

# parse arguments
args = parser.parse_args()
C = args.c
TYPE = args.type
INTERPOLATION = args.interpolation

# print chosen arguments
print("C:", C)
print("TYPE:", TYPE)
print("INTERPOLATION:", INTERPOLATION)


fixed_value = 1.8e-11

m_6052 = -97e-15
q_6052 = 6.2e-13
r2_6052 = 3.5e-26

m_6194 = -59e-15
q_6194 = 4.1e-13
r2_6194 = 0.8e-26

m_7221 = -3.4e-15
q_7221 = 2.8e-13
r2_7221 = 0.2e-26

m_7392 = -10e-15
q_7392 = 5.3e-13
r2_7392 = 0.4e-26


with open("colimator-scans-data.pkl", "rb") as f:
    data = pickle.load(f)


with open("colimator-scans-details.pkl", "rb") as f:
    details = pickle.load(f)

# # Grab some beam data properly


details[6052]["beam1"]["horizontal"]


details[6052]["beam1"]["vertical"]


data[6052]["TCP_IR7_B1H"]["lowres"].keys()


data[6052]["TCP_IR7_B1V"]["lowres"].keys()


plt.plot(
    data[6052]["TCP_IR7_B1H"]["lowres"]["timestamps"],
    data[6052]["TCP_IR7_B1H"]["lowres"]["TCP.C6L7.B1:MEAS_LVDT_LU"] / 0.28,
)

# plt.plot(
#     data[6052]["TCP_IR7_B1H"]["lowres"]["timestamps"],
#     data[6052]["TCP_IR7_B1H"]["lowres"]["TCP.C6R7.B2:MEAS_LVDT_LU"] / 0.28,
# )

plt.plot(
    data[6052]["TCP_IR7_B1V"]["lowres"]["timestamps"],
    data[6052]["TCP_IR7_B1V"]["lowres"]["TCP.D6L7.B1:MEAS_LVDT_LU"] / 0.20,
)

# plt.plot(
#     data[6052]["TCP_IR7_B1V"]["lowres"]["timestamps"],
#     data[6052]["TCP_IR7_B1V"]["lowres"]["TCP.D6R7.B2:MEAS_LVDT_LU"] / 0.28,
# )


data[6052]["TCP_IR7_B1V"]["hires"].keys()


d = data[6052]["TCP_IR7_B1V"]["hires"]["BLMTI.06L7.B1E10_TCP.D6L7.B1:LOSS_FAST"]
d[1][d[1] <= 0] = np.nan


plt.figure(figsize=(18, 6))

plt.plot(d[0], d[1] / fixed_value, linewidth=0.5)
plt.yscale("log")

plt.twinx()
plt.plot(
    data[6052]["TCP_IR7_B1V"]["lowres"]["timestamps"],
    data[6052]["TCP_IR7_B1V"]["lowres"]["TCP.D6L7.B1:MEAS_LVDT_LU"] / 0.20,
    c="C1",
)


plt.figure(figsize=(18, 6))

plt.plot(d[0], d[1] / fixed_value, linewidth=0.5)
plt.yscale("log")

plt.twinx()
plt.scatter(
    data[6052]["TCP_IR7_B1V"]["lowres"]["timestamps"],
    data[6052]["TCP_IR7_B1V"]["lowres"]["TCP.D6L7.B1:MEAS_LVDT_LU"] / 0.20,
    c="C1",
)

plt.xlim(55000 + 1.5020e9, 57000 + 1.5020e9)


func = scipy.interpolate.interp1d(
    data[6052]["TCP_IR7_B1V"]["lowres"]["timestamps"],
    data[6052]["TCP_IR7_B1V"]["lowres"]["TCP.D6L7.B1:MEAS_LVDT_LU"] / 0.20,
    kind="previous",
    fill_value="extrapolate",
)


plt.figure(figsize=(18, 6))

plt.plot(d[0], d[1] / fixed_value, linewidth=0.5)
plt.yscale("log")

plt.twinx()
plt.plot(d[0], func(d[0]), c="C1")

plt.xlim(55250 + 1.5020e9, 56750 + 1.5020e9)


plt.figure(figsize=(18, 6))

plt.plot(d[0], d[1] / fixed_value, linewidth=0.5)
plt.yscale("log")

plt.twinx()
plt.plot(d[0][:-1], np.diff(func(d[0])), c="C1")

plt.xlim(55250 + 1.5020e9, 56750 + 1.5020e9)


diffs = np.diff(func(d[0]))
idxs = np.where(diffs > 0.01)[0]
vals = d[0][np.where(diffs > 0.01)]


idxs


plt.figure(figsize=(18, 6))

plt.plot(d[0], d[1] / fixed_value, linewidth=0.5)
plt.yscale("log")

plt.twinx()
plt.plot(d[0][:-1], np.diff(func(d[0])), c="C1")

for v in vals:
    plt.axvline(v, c="C1", linestyle="--")

plt.xlim(55250 + 1.5020e9, 56750 + 1.5020e9)


slices = []

for i in range(len(vals) - 1):
    slices.append(
        (
            d[0][idxs[i] : idxs[i + 1]],
            d[1][idxs[i] : idxs[i + 1]],
            np.mean(func(d[0][idxs[i] + 1 : idxs[i + 1] - 1])),
            (idxs[i], idxs[i + 1]),
        )
    )


# plt.figure(figsize=(18, 6))
plt.figure(figsize=(12, 6))

for s in slices:
    plt.plot(s[0], s[1] / fixed_value, linewidth=0.5)
plt.yscale("log")

plt.twinx()
for s in slices:
    plt.plot(s[0], np.ones_like(s[0]) * s[2], c="black")


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


# plt.figure(figsize=(18, 6))
plt.figure(figsize=(12, 6))

for s in slices:
    plt.plot(s[0], s[1] / fixed_value, linewidth=0.5)
    plt.plot(s[0], moving_average(s[1] / fixed_value, n=30), linewidth=0.5, c="black")
plt.yscale("log")

plt.twinx()
for s in slices:
    plt.plot(s[0], np.ones_like(s[0]) * s[2], c="black")


slices = []

for i in range(len(vals) - 1):
    slices.append(
        (
            d[0][idxs[i] : idxs[i + 1]],
            d[1][idxs[i] : idxs[i + 1]],
            np.mean(func(d[0][idxs[i] + 1 : idxs[i + 1] - 1])),
            (idxs[i], idxs[i + 1]),
        )
    )


def reset_indexes(slices, slice_val=100):
    new_slices = []
    for i, s in enumerate(slices):
        if i == 0:
            new_left = np.argmin(s[1][:slice_val]) + s[3][0]
        else:
            data = np.concatenate(
                (slices[i - 1][1][-slice_val:], slices[i][1][:slice_val])
            )
            rel_index = np.argmin(data)
            if rel_index < slice_val:
                new_left = slices[i - 1][3][1] + rel_index
            else:
                new_left = slices[i][3][0] + rel_index - slice_val

        if i == len(slices) - 1:
            new_right = np.argmax(s[1][-slice_val:]) + s[3][1] - slice_val
        else:
            data = np.concatenate(
                (slices[i][1][-slice_val:], slices[i + 1][1][:slice_val])
            )
            rel_index = np.argmax(data)
            if rel_index < slice_val:
                new_right = slices[i][3][1] + rel_index - slice_val
            else:
                new_right = slices[i + 1][3][0] + rel_index - slice_val

        new_slices.append(
            (
                d[0][new_left:new_right],
                d[1][new_left:new_right],
                np.mean(func(d[0][idxs[i] + 1 : idxs[i + 1] - 1])),
                (new_left, new_right),
            )
        )
    return new_slices


new_slices = reset_indexes(slices, slice_val=50)


# plt.figure(figsize=(18, 6))
plt.figure(figsize=(12, 6))

for s in new_slices:
    plt.plot(s[0], s[1] / fixed_value, linewidth=0.5)
plt.yscale("log")

plt.twinx()
for s in new_slices:
    plt.plot(s[0], np.ones_like(s[0]) * s[2], c="black")


def gather_points(slices, end_sample):
    points = []
    for s in slices:
        points.append(
            (
                s[0][-1],
                np.nanmin(s[1][-end_sample:]),
                np.nanmean(s[1][-end_sample:]),
                np.nanmax(s[1][-end_sample:]),
            )
        )
    return points


import scipy.optimize

points = gather_points(new_slices, end_sample=50)

f_basic = scipy.interpolate.interp1d(
    np.array([p[0] for p in points]),
    np.array([p[3] for p in points]),
    kind="quadratic",
    bounds_error=False,
    fill_value=np.nan,
)


# plt.figure(figsize=(18, 6))
plt.figure(figsize=(12, 6))

for s in new_slices:
    plt.plot(s[0], s[1] / fixed_value, linewidth=0.5, c="C0")

    plt.plot(s[0], f_basic(s[0]) / fixed_value, c="C1")

plt.yscale("log")

plt.twinx()
for s in new_slices:
    plt.plot(s[0], np.ones_like(s[0]) * s[2], c="black")


t = np.array([p[0] for p in points])
v = np.array([p[3] for p in points])
t0 = t[0]


import lmfit


def res(params, x, y):
    a = params["a"].value
    b = params["b"].value
    c = params["c"].value
    return ((a * np.exp(-(np.power(10.0, b)) * (x - t0)) + c) - y) / y


params = lmfit.Parameters()
params.add("a", value=1)
params.add("b", value=-1)
params.add("c", value=0, vary=False)


result = lmfit.minimize(res, params, args=(t, v))


result


def fitted_function(x):
    a = result.params["a"].value
    b = result.params["b"].value
    c = result.params["c"].value
    return a * np.exp(-(np.power(10.0, b)) * (x - t0)) + c


f_spline = scipy.interpolate.UnivariateSpline(t, np.log10(v), k=3, s=1)


plt.figure(figsize=(12, 6))

for s in new_slices:
    plt.plot(s[0], s[1] / fixed_value, linewidth=0.5, c="C0", alpha=0.4)

plt.scatter(t, v / fixed_value, c="navy")

plt.plot(t, fitted_function(t) / fixed_value, c="C1")
plt.plot(t, f_basic(t) / fixed_value, c="C2")
plt.plot(t, np.power(10.0, f_spline(t)) / fixed_value, c="C3")

plt.yscale("log")

plt.tight_layout()


slice = 30
skipping = 300


fit_x_raw = []
fit_x_nonorm = []
fit_y_raw = []
fit_x_list = []
fit_y_list = []

for i in range(1, len(new_slices)):
    from_value = new_slices[i - 1][2]
    to_value = new_slices[i][2]
    times = new_slices[i][0] - new_slices[i][0][0]
    the_function = fitted_function(new_slices[i][0])
    values = new_slices[i][1] / the_function

    avg_times = moving_average(times, slice)
    avg_values = moving_average(values, slice)

    avg_times = avg_times[::skipping]
    avg_values = avg_values[::skipping]

    tmp = moving_average(new_slices[i][0], slice)[::skipping]

    tmp = tmp[~np.isnan(avg_values)]
    avg_times = avg_times[~np.isnan(avg_values)]
    avg_values = avg_values[~np.isnan(avg_values)]

    fit_x_raw.append(new_slices[i][0])
    fit_y_raw.append(values)

    fit_x_nonorm.append(tmp)
    fit_x_list.append(("forward", from_value, to_value, avg_times))
    fit_y_list.append(avg_values - 1)


plt.figure(figsize=(18, 6))

for x, y in zip(fit_x_raw, fit_y_raw):
    plt.plot(x, y - 1, linewidth=0.5, c="C0", alpha=0.4)

for x, y in zip(fit_x_nonorm, fit_y_list):
    plt.plot(x, y, c="navy")

plt.ylim(top=0.0)
plt.tight_layout()


basic_x_raw = []
basic_x_nonorm = []
basic_y_raw = []
basic_x_list = []
basic_y_list = []

for i in range(1, len(new_slices)):
    from_value = new_slices[i - 1][2]
    to_value = new_slices[i][2]
    times = new_slices[i][0] - new_slices[i][0][0]
    the_function = f_basic(new_slices[i][0])
    values = new_slices[i][1] / the_function

    avg_times = moving_average(times, slice)
    avg_values = moving_average(values, slice)

    avg_times = avg_times[::skipping]
    avg_values = avg_values[::skipping]

    tmp = moving_average(new_slices[i][0], slice)[::skipping]

    tmp = tmp[~np.isnan(avg_values)]
    avg_times = avg_times[~np.isnan(avg_values)]
    avg_values = avg_values[~np.isnan(avg_values)]

    basic_x_raw.append(new_slices[i][0])
    basic_y_raw.append(values)

    basic_x_nonorm.append(tmp)
    basic_x_list.append(("forward", from_value, to_value, avg_times))
    basic_y_list.append(avg_values - 1)


plt.figure(figsize=(18, 6))

for x, y in zip(basic_x_raw, basic_y_raw):
    plt.plot(x, y - 1, linewidth=0.5, c="C0", alpha=0.4)

for x, y in zip(basic_x_nonorm, basic_y_list):
    plt.plot(x, y, c="navy")

plt.ylim(top=0.0)

plt.tight_layout()


spline_x_raw = []
spline_x_nonorm = []
spline_y_raw = []
spline_x_list = []
spline_y_list = []

for i in range(1, len(new_slices)):
    from_value = new_slices[i - 1][2]
    to_value = new_slices[i][2]
    times = new_slices[i][0] - new_slices[i][0][0]
    the_function = np.power(10.0, f_spline(new_slices[i][0]))
    values = new_slices[i][1] / the_function

    avg_times = moving_average(times, slice)
    avg_values = moving_average(values, slice)

    avg_times = avg_times[::skipping]
    avg_values = avg_values[::skipping]

    tmp = moving_average(new_slices[i][0], slice)[::skipping]

    tmp = tmp[~np.isnan(avg_values)]
    avg_times = avg_times[~np.isnan(avg_values)]
    avg_values = avg_values[~np.isnan(avg_values)]

    spline_x_raw.append(new_slices[i][0])
    spline_y_raw.append(values)

    spline_x_nonorm.append(tmp)
    spline_x_list.append(("forward", from_value, to_value, avg_times))
    spline_y_list.append(avg_values - 1)


plt.figure(figsize=(18, 6))

for x, y in zip(basic_x_raw, basic_y_raw):
    plt.plot(x, y - 1, linewidth=0.5, c="C0", alpha=0.4)

for x, y in zip(basic_x_nonorm, basic_y_list):
    plt.plot(x, y, c="navy")

plt.ylim(top=0.0)
plt.tight_layout()


if INTERPOLATION == "spline":

    if TYPE == "single":
        pars = lmfit.Parameters()
        pars.add("I_star", value=5.0, min=0.0)
        pars.add("k", value=0.33, min=0.0)
        pars.add("c", value=C, vary=False)

        spline_result = lmfit.minimize(
            fit_functions.resid_func, pars, args=(spline_x_list, spline_y_list)
        )
        spline_c1, spline_c2 = fit_functions.ana_current(
            spline_result.params, spline_x_list, spline_y_list
        )

        with open("spline_result.pkl", "wb") as f:
            pickle.dump({"result": spline_result, "c1": spline_c1, "c2": spline_c2,}, f)

    elif TYPE == "iterative":
        pars = lmfit.Parameters()
        pars.add("I_star", value=5.0, min=0.0)
        pars.add("k", value=0.33, min=0.0)
        pars.add("c", value=C, vary=False)

        spline_result_list = []
        for i in tqdm(range(5, len(spline_x_list))):
            spline_result = lmfit.minimize(
                fit_functions.resid_func,
                pars,
                args=(spline_x_list[:i], spline_y_list[:i]),
            )
            spline_c1, spline_c2 = fit_functions.ana_current(
                spline_result.params, spline_x_list, spline_y_list
            )
            spline_result_list.append((spline_result, spline_c1, spline_c2))
            pars = spline_result.params

        with open("iterative_spline_result.pkl", "wb") as f:
            pickle.dump(spline_result_list, f)

elif INTERPOLATION == "fit":

    if TYPE == "single":
        pars = lmfit.Parameters()
        pars.add("I_star", value=5.0, min=0.0)
        pars.add("k", value=0.33, min=0.0)
        pars.add("c", value=C, vary=False)

        fit_result = lmfit.minimize(
            fit_functions.resid_func, pars, args=(fit_x_list, fit_y_list)
        )
        fit_c1, fit_c2 = fit_functions.ana_current(
            fit_result.params, fit_x_list, fit_y_list
        )

        with open("fit_result.pkl", "wb") as f:
            pickle.dump({"result": fit_result, "c1": fit_c1, "c2": fit_c2,}, f)

    elif TYPE == "iterative":
        pars = lmfit.Parameters()
        pars.add("I_star", value=5.0, min=0.0)
        pars.add("k", value=0.33, min=0.0)
        pars.add("c", value=C, vary=False)

        fit_result_list = []
        for i in tqdm(range(5, len(fit_x_list))):
            fit_result = lmfit.minimize(
                fit_functions.resid_func, pars, args=(fit_x_list[:i], fit_y_list[:i])
            )
            fit_c1, fit_c2 = fit_functions.ana_current(
                fit_result.params, fit_x_list, fit_y_list
            )
            fit_result_list.append((fit_result, fit_c1, fit_c2))
            pars = fit_result.params

        with open("iterative_fit_result.pkl", "wb") as f:
            pickle.dump(fit_result_list, f)

elif INTERPOLATION == "basic":

    if TYPE == "single":
        pars = lmfit.Parameters()
        pars.add("I_star", value=5.0, min=0.0)
        pars.add("k", value=0.33, min=0.0)
        pars.add("c", value=C, vary=False)

        basic_result = lmfit.minimize(
            fit_functions.resid_func, pars, args=(basic_x_list, basic_y_list)
        )
        basic_c1, basic_c2 = fit_functions.ana_current(
            basic_result.params, basic_x_list, basic_y_list
        )

        with open("basic_result.pkl", "wb") as f:
            pickle.dump({"result": basic_result, "c1": basic_c1, "c2": basic_c2,}, f)

    elif TYPE == "iterative":
        pars = lmfit.Parameters()
        pars.add("I_star", value=5.0, min=0.0)
        pars.add("k", value=0.33, min=0.0)
        pars.add("c", value=C, vary=False)

        basic_result_list = []
        for i in tqdm(range(5, len(basic_x_list))):
            basic_result = lmfit.minimize(
                fit_functions.resid_func,
                pars,
                args=(basic_x_list[:i], basic_y_list[:i]),
            )
            basic_c1, basic_c2 = fit_functions.ana_current(
                basic_result.params, basic_x_list, basic_y_list
            )
            basic_result_list.append((basic_result, basic_c1, basic_c2))
            pars = basic_result.params

        with open("iterative_basic_result.pkl", "wb") as f:
            pickle.dump(basic_result_list, f)

