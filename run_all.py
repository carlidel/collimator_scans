import itertools
import subprocess
from multiprocessing import Pool

C_LIST = [1.0]
TYPE_LIST = ["iterative"]
INTERP_LIST = ["spline", "fit", "basic", "forfun"]


def run_command(cmd):
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    return p


processes = []
for c, t, interp in itertools.product(C_LIST, TYPE_LIST, INTERP_LIST):
    command = (
        f"python possible_interpolations.py -c {c} -type {t} -interpolation {interp}"
    )
    processes.append(run_command(command))

for p in processes:
    p.wait()
