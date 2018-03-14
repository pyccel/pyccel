# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt

def matrix_product():
    procs = [1, 4, 8, 16, 28]
    times = [1194.849, 305.231, 69.174,37.145, 22.731]

    n_groups = len(procs)

    # ...
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.4
    rects1 = plt.bar(index, times, bar_width,
                     alpha=opacity,
                     color='b',
                     label='OpenMP')

    plt.xlabel('Number of Processors')
    plt.ylabel('CPU time')
    plt.title('Weak scaling')
    labels = [str(i) for i in procs]
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()

    plt.tight_layout()
    plt.savefig("matrix_product_scalability.png")
    plt.clf()
    # ...

    # ...
    speedup = [times[0]/b for b in times[1:]]
    n_groups = len(speedup)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.4
    rects1 = plt.bar(index, speedup, bar_width,
                     alpha=opacity,
                     color='b',
                     label='OpenMP')

    plt.xlabel('Number of Processors')
    plt.ylabel('Speedup')
    plt.title('Speedup')
    labels = [str(i) for i in procs[1:]]
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()

    plt.tight_layout()
    plt.savefig("matrix_product_speedup.png")
    plt.clf()
    # ...

matrix_product()
