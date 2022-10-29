import os
import csv
import sys
import time
import json
import diffevo
import bayesopt
import ga
import random_search
import grid_search
from tqdm import tqdm


def run_n_times(algorithm, segments, n, iters):
    """
    Perform a meta-experiment. The chosen algorithm is run n times for a given
    number of gradient segments. The CRF score per iteration and the cumulative
    runtime per iteration are written to csv files. Filepaths need to be specified
    manually and should indicate which sample was used in read_data.py. This has
    to be done manually because optimization algorithm packages don't allow
    for extra arguments in the objective function, other than the parameters
    to be optimized.

    :algorithm: Optimization algorithm. Choose from:
                BayesOpt/DiffEvo/GenAlgo/GridSearch/RandomSearch
    :segments: Number of gradient segments in the gradient profile.
    """
    with open('globals.json') as json_file:
        variables = json.load(json_file)

    sample = str(variables["sample_name"])
    wet = str(variables["wet"])
    crf_name = str(variables["crf_name"])

    if(wet == "True"):
        prefix = "results/wet/" + crf_name + "/" + str(segments) + "segments/" + sample + "/" + algorithm + "/"
    else:
        prefix = "results/dry/" + crf_name + "/" + str(segments) + "segments/" + sample + "/" + algorithm + "/"

    filename_score = "score" + ".csv"
    filename_runtime = "runtime" + ".csv"
    filename_solution = "solution" + ".csv"

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    filepath_score = prefix + filename_score
    filepath_runtime = prefix + filename_runtime
    filepath_solution = prefix + filename_solution

    if(algorithm == "BayesOpt"):

        for nth_experiment in tqdm(range(n)):
            # n = number of meta experiments
            return_list = bayesopt.bayesopt(iters, segments)
            best_solution = return_list[3]
            func_vals = return_list[4]
            runtime_per_iteration = return_list[5]
            # write the data from the list to a (csv?) file as a single line

            f = open(filepath_score, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(func_vals)

            f = open(filepath_runtime, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(runtime_per_iteration)

            f = open(filepath_solution, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(best_solution)


    elif(algorithm == "DiffEvo"):

        for nth_experiment in tqdm(range(n)):
            # n = number of meta experiments
            return_list = diffevo.diffevo(iters, segments)
            best_solution = return_list[3]
            func_vals = return_list[4]
            runtime_per_iteration = return_list[5]

            f = open(filepath_score, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(func_vals)

            f = open(filepath_runtime, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(runtime_per_iteration)

            f = open(filepath_solution, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(best_solution)


    elif(algorithm == "GenAlgo"):

        for nth_experiment in tqdm(range(n)):
            return_list = ga.ga(iters, segments)
            best_solution = return_list[3]
            func_vals = return_list[4]
            runtime_per_iteration = return_list[5]

            f = open(filepath_score, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(func_vals)

            f = open(filepath_runtime, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(runtime_per_iteration)

            f = open(filepath_solution, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(best_solution)


    elif(algorithm == "RandomSearch"):

        for nth_experiment in tqdm(range(n)):
            # n = number of meta experiments
            return_list = random_search.run_rs(iters, segments)
            best_solution = return_list[3]
            func_vals = return_list[4]
            runtime_per_iteration = return_list[5]

            f = open(filepath_score, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(func_vals)

            f = open(filepath_runtime, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(runtime_per_iteration)

            f = open(filepath_solution, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(best_solution)


    elif(algorithm == "GridSearch"):

        return_list = grid_search.grid_search(iters, segments)
        func_val = return_list[2]
        runtime = return_list[1]
        f = open(filepath, 'a', newline ='\n')

        with f:
            writer = csv.writer(f)
            writer.writerow([func_val])

        f = open(filepath_runtime, 'a', newline ='\n')

        with f:
            writer = csv.writer(f)
            writer.writerow([runtime])


def main():

    if len(sys.argv) > 8:
        print('You have specified too many arguments.')
        sys.exit()

    if len(sys.argv) < 8:
        print('Please specify the following parameters in order:')
        print("- Choose an optimization algorithm (BayesOpt/DiffEvo/GenAlgo/GridSearch/RandomSearch)")
        print("- Number of segments in the gradient profile")
        print("- Number of sub-experiments the meta-experiment should consist of")
        print("- Number of iterations. Note that if the chosen algorithm is grid search, this is the number of grid points per dimension.")
        print("- Name of the sample.")
        print("- Wet? (True/False)")
        print("- CRF name (prod_of_res/tyteca11/sum_of_res/tyteca24)")
        sys.exit()

    algorithm = sys.argv[1]
    number_of_segments = int(sys.argv[2])
    sub_experiments = int(sys.argv[3])
    iterations = int(sys.argv[4])
    sample_name = sys.argv[5]
    wet = sys.argv[6]
    crf_name = sys.argv[7]

    # Write variables to json file
    variable_dict = {
        "wet": wet,
        "crf_name": crf_name,
        "sample_name": sample_name,
        "algorithm": algorithm
    }

    json_object = json.dumps(variable_dict, indent=4)

    with open("globals.json", "w") as outfile:
        outfile.write(json_object)

    run_n_times(algorithm, number_of_segments, sub_experiments, iterations)


if __name__ == '__main__':
    main()
