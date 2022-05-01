import itertools
import subprocess
from multiprocessing import Pool


def run_command(cmd):
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    return p


processes = []
for i in range(8):
    command = f"python just_fit.py -i {i} -k 2.1 -l b1_v -f to_fit_b1_v.pkl"
    processes.append(run_command(command))

for p in processes:
    p.wait()
processes = []

for i in range(8):
    command = f"python just_fit.py -i {i} -k 2.1 -l b2_v -f to_fit_b2_v.pkl"
    processes.append(run_command(command))

for p in processes:
    p.wait()
processes = []

for i in range(4):
    command = f"python just_fit.py -i {i} -k 2.1 -l b1_h -f to_fit_b1_h.pkl"
    processes.append(run_command(command))

    # for p in processes:
    #     p.wait()
    # processes = []

    # for i in range(1):
    command = f"python just_fit.py -i {i} -k 2.1 -l b2_h -f to_fit_b2_h.pkl"
    processes.append(run_command(command))

for p in processes:
    p.wait()
