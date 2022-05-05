import itertools
import subprocess
from multiprocessing import Pool


def run_command(cmd):
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    return p


modules_to_test = [750, 500, 250, 100, 1250, 1500, 1750, 2000]

for module in modules_to_test:
    processes = []
    for i in range(4):
        command = (
            f"python just_fit.py -i {i} -k 2.1 -l b1_v -f to_fit_b1_v.pkl -m {module}"
        )
        processes.append(run_command(command))

    for p in processes:
        p.wait()
