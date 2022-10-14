import crf
import retention_model
import read_data as rd
import math
import json
import numpy as np
import chromatographic_response_funcions as of
import peak_and_width_detection as pwd

import globals
# HPLC system parameters
t_0 = globals.t_0
t_D = globals.t_D
N = globals.N


def chromosome_to_lists(chromosome):
    """
    Transform a candidate solution vector into separate lists for
    phi values, t_init and delta_t values respectively.
    """
    l = len(chromosome)
    segments = int((l - 2)/2)

    phi_list = []
    for i in range(segments + 1):
        phi_list.append(chromosome[i])

    t_init = chromosome[segments + 1]
    delta_t_list = []

    for i in range(segments + 2, 2*segments + 2):
        delta_t_list.append(chromosome[i])
    t_list = [0]

    for i, delta_t in enumerate(delta_t_list):
        t_list.append(t_list[i] + delta_t)
    return(phi_list, t_init, t_list)



# Input has to be a numpy array
def interface(chromosome):
    """
    This function serves as an interface between the Bayesian optimization,
    differential evolution, random search and grid search packages and the CRF
    function. This is necessary because the gradient profile specified by the
    candidate solution vector has to be transformed into a chromatogram
    (list of retention times and peak widths) for a given sample before the CRF
    score can be calculated. It does this using the following steps:

    1. Read in sample data using read_data.py
    2. Calculate retention times and peak widths for all sample compounds
       using the chromatographic simulator as implemented in retention_model.py
    3. Calculate and return the CRF score

    :chromosome: Gradient profile vector in the form of a numpy array.
    :return: CRF score for a chromatogram produced by the specified gradient
             profile.

    """
    with open('globals.json') as json_file:
        variables = json.load(json_file)

    crf_name = str(variables["crf_name"])
    wet = bool(variables["wet"])
    alg = str(variables["algorithm"])

    phi_list, t_init, t_list = chromosome_to_lists(chromosome)

    # Get lnk0 and S data
    k0_list, S_list = rd.read_data()
    #k0_list = [math.exp(lnk0) for lnk0 in lnk0_list]

    tR_list = []
    W_list = []

    # Calculate retention times and peak widths
    for i in range(len(k0_list)):
        k_0 = k0_list[i]
        S = S_list[i]

        tR, W = retention_model.retention_time_multisegment_gradient(k_0, S, t_0, t_D, t_init, phi_list, t_list, N)
        tR_list.append(tR)
        W_list.append(W)

    tlim = max(tR_list) + max(W_list)

    if(wet == True):
        # Wet signal
        x, signal = pwd.create_signal(np.array(tR_list), np.array(W_list), tlim)
        tR_list, W_list = pwd.detect_peaks(x, signal, height_thresh=0, plot=False)

    # Calculate crf

    print(crf_name)
    print(tR_list, W_list, phi_list)

    if(crf_name == "sum_of_res"):
        score = crf.capped_sum_of_resolutions(np.array(tR_list), np.array(W_list), phi_list)
    elif(crf_name == "prod_of_res"):
        score = crf.crf(tR_list, W_list, phi_list)
    elif(crf_name == "tyteca11"):
        score = crf.tyteca_eq_11(np.array(tR_list), np.array(W_list))
    elif(crf_name == "tyteca24"):
        score = crf.tyteca_eq_24(np.array(tR_list), np.array(W_list))

    return(-1 * score)


# Input has to be a numpy array
def interface_pygad(chromosome, solution_id):
    #print("Did an evaluaton.")
    """
    This function serves as an interface between the genetic algorithm package
    and the CRF function. This is necessary because the gradient profile
    specified by the candidate solution vector has to be transformed into a
    chromatogram (list of retention times and peak widths) for a given sample
    before the CRF score can be calculated. It does this using the following
    steps:

    1. Read in sample data using read_data.py
    2. Calculate retention times and peak widths for all sample compounds
       using the chromatographic simulator as implemented in retention_model.py
    3. Calculate and return the CRF score

    :chromosome: Gradient profile vector in the form of a numpy array.
    :solution_id: Solution ID required by PyGAD package.
    :return: CRF score for a chromatogram produced by the specified gradient
             profile.

    """

    with open('globals.json') as json_file:
        variables = json.load(json_file)

    crf_name = str(variables["crf_name"])
    wet = bool(variables["wet"])
    alg = str(variables["algorithm"])

    #print("Did an evaluaton.")
    phi_list, t_init, t_list = chromosome_to_lists(chromosome)

    # Get lnk0 and S data
    k0_list, S_list = rd.read_data()
    #k0_list = [math.exp(lnk0) for lnk0 in lnk0_list]

    tR_list = []
    W_list = []

    # Calculate retention times and peak widths
    for i in range(len(k0_list)):
        k_0 = k0_list[i]
        S = S_list[i]

        tR, W = retention_model.retention_time_multisegment_gradient(k_0, S, t_0, t_D, t_init, phi_list, t_list, N)
        tR_list.append(tR)
        W_list.append(W)

    tlim = max(tR_list) + max(W_list)

    if(wet == True):
        # Wet signal
        x, signal = pwd.create_signal(np.array(tR_list), np.array(W_list), tlim)
        tR_list, W_list = pwd.detect_peaks(x, signal, height_thresh=0, plot=False)

    # Calculate crf

    if(crf_name == "sum_of_res"):
        score = crf.capped_sum_of_resolutions(np.array(tR_list), np.array(W_list), phi_list)
    elif(crf_name == "prod_of_res"):
        score = crf.crf(tR_list, W_list, phi_list)
    elif(crf_name == "tyteca11"):
        score = crf.tyteca_eq_11(np.array(tR_list), np.array(W_list))
    elif(crf_name == "tyteca24"):
        score = crf.tyteca_eq_24(np.array(tR_list), np.array(W_list))

    return(score)
