import numpy as np
from scipy.stats import norm
import helper_functions as hf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# our imports for today
import numpy as np
import GPy
from scipy import stats
from scipy import optimize
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return(ei)


def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)


def bo(iterations, segments):

    bounds = hf.get_bounds(segments)
    # Sample the objective function at n initial points, (this becomes X and y)
    initial_population = hf.initialize_population_random_uniform(30, bounds)
    X = np.array(initial_population)

    # This should work instantly without in-between function
    #y = interface.interface(np.array(chromosome))
    y = np.array(hf.evaluate_population(X.tolist())

    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    # Define the surrogate function
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)

    for i in iterations:

        # Find the point xmax that maximizes the acquisition function

        #Evaluate the true objective function at xmax and append to X and y
        y = np.array(hf.evaluate_population(X.tolist())

        # Update the Gaussian process using all available data points
        model.fit(X, y)
    pass
