import csv
import random
import numpy as np
import globals


def read_data():
    """
    Return list of k0 and S samples for a given sample. Sample needs to be
    manually specified here, because optimization algorithm packages don't allow
    for extra arguments in the objective function, other than the parameters
    to be optimized.

    """
    k0_list = []
    S_list = []

    path = globals.path

    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            k0_list.append(float(row[0]))
            S_list.append(float(row[1]))

    # Get the first 35 elements from each list. This is the sample.
    k0_list = k0_list[:35]
    S_list = S_list[:35]
    return(k0_list, S_list)
