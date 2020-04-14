import numpy as np
import pandas as pd
import scipy.integrate

# Fixed model parameters:
INCUBATION_PERIOD = 3
DURATION_OF_INFECTION = 14

COMPARTMENTS = ['Susceptible', 'Exposed', 'Infected', 'Died or recovered']
AGE_RANGES = ['0-4', '5-9', '10-19', '20+']
ROUGH_2017_POPULATION = [32., 28., 44., 88.]  # in millions, per Wikipedia 
POPULATION_FRACTIONS = ROUGH_2017_POPULATION / np.sum(ROUGH_2017_POPULATION)

# Initial compartment populations
initial_infected = .01
pop_0 = np.round(np.array([[f - initial_infected, 0, initial_infected, 0] for f in POPULATION_FRACTIONS]), decimals=5)


def model_input(cohort_ranges):
	"""
	Function to calculuate transition matrices and to enumerate their end
	period for four cohorts.

	Accepts a list of day range tuples for each cohort's time in general
	population

	Returns a list of two lists for input into SEIRModel():
		1. A list of the transition matrices
		2. A list of epoch end times

	"""

	# The transition matrix for when ALL four cohorts are in general population
	genpop_matrix = np.array([
		[0.1, 0.1, 0.2, 0.1],
		[0.1, 0.1, 0.2, 0.1],
		[0.2, 0.2, 0.2, 0.1],
		[0.1, 0.1, 0.1, 0.1]
	])

	# Partition total range in to its unique end periods
	flat_dates = [v for sublist in cohort_ranges for v in sublist]
	flat_dates = set(flat_dates + [0, 180])

	# Unique and sorted list of epoch begin and end days.
	epoch_delims = sorted(list(dict.fromkeys(flat_dates)))

	def _tuplize(lst):
		# create a sequence of tuples from a list
		# e.g., tuplize([0, 1, 2, 3]) -> [[0, 1], [1, 2], [2, 3]]
		for i in range(0, len(lst)):
			val = lst[i:i+2]
			if len(val) == 2:
				yield val

	# Delineates the ranges for different transition matrices required.  The
	# length of this list is equal to the length of the final result.
	epoch_tuples = list(_tuplize(epoch_delims))

	# Custom check to deal with range sliders inclusive of both bounds and
	# zero-indexed python lists
	def _is_removed(cohort_range, epoch_tuple):
		cs = set(range(*cohort_range))
		es = set(range(*epoch_tuple))
		if len(cs.intersection(es)) <= 1:
			return True
		else:
			return False

	transition_matrices = []
	# iteratively alter the genpop matrix by checking if a certain population
	# should be removed for each time partition
	for j in range(len(epoch_tuples)): 

		# reset general popultation matrix for each epoch, and set transmission to
		# zero if the cohort has been removed
		res_mat = genpop_matrix.copy()

		for i in range(len(cohort_ranges)):
			if _is_removed(cohort_ranges[i], epoch_tuples[j]):
				res_mat[i, :] = 0
				res_mat[:, i] = 0


		transition_matrices.append(res_mat)

	# Remove the first delimiter, which is a start day (not an end day)
	epoch_ends = epoch_delims[1:]

	return [transition_matrices, epoch_ends]

class SEIRModel(object):
    """Class to instantiate and solve an age-structured SEIR Compartmental model."""

    def __init__(self, betas, epoch_end_times, alpha=1/INCUBATION_PERIOD, gamma=1/DURATION_OF_INFECTION, 
                 age_ranges=AGE_RANGES):
        
        if len(epoch_end_times) != len(betas):
            raise ValueError('Each beta matrix requires an epoch end time.')
        self.epoch_end_times = sorted(epoch_end_times)
        self.betas = betas
        self.alpha = alpha
        self.gamma = gamma
        self.age_ranges = age_ranges
        self.N_ages = len(age_ranges)
        
        # N.B. hard-coded values. The evolution function evolve() assumes these four compartments.
        self.compartments = COMPARTMENTS
        self.N_compartments = 4
        self.s, self.e, self.i, self.r = list(range(4))                  
                                                       
    def _beta(self, t):
        for end_time, beta in zip(self.epoch_end_times, self.betas): 
            if t <= end_time: 
                return beta      
        
    def evolve(self, t, y):
        """Compute the change in the state variable y(t)."""
        y = y.reshape(self.N_ages, self.N_compartments)
        dy = np.zeros((self.N_ages, self.N_compartments))
        beta = self._beta(t)
        for a in range(self.N_ages):
            dy[a,self.s] = -y[a,self.s] * np.sum([beta[a,b] * y[b,self.i] / np.sum(y[b,:]) for b in range(self.N_ages)])
            dy[a,self.e] = y[a,self.s] * np.sum([beta[a,b] * y[b,self.i] / np.sum(y[b,:]) for b in range(self.N_ages)]) - self.alpha * y[a, self.e]
            dy[a,self.i] = self.alpha * y[a, self.e] - self.gamma * y[a, self.i]
            dy[a,self.r] = self.gamma * y[a, self.i]
        return dy.flatten()
    
    def solve(self, y0):
        """Integrate the coupled differential equations."""
        sol = scipy.integrate.solve_ivp(self.evolve, (0, self.epoch_end_times[-1]), y0, 
                                        t_eval=np.arange(self.epoch_end_times[-1]))
        return sol.t, sol.y
    
    def solve_to_dataframe(self, y0):
        """Integrate the the differntial equations and output a tidy dataframe."""
        t, y = self.solve(y0)
        y = y.reshape(self.N_ages, self.N_compartments, len(t))
        y = np.sum(y, axis=0)
        df = pd.DataFrame(dict({'days': t}, **dict(zip(self.compartments, y))))
        df = pd.melt(df, id_vars=['days'], var_name='Group', value_name='pop')
        df['Date(s) of intervention'] = str(self.epoch_end_times[:-1])
        return df

