"""Code to build and solve an age-structured SEIR epidemiological model.

Some model assumptions, adjustable as needed:
 - Four compartment (S-E-I-R) model, stratified by age cohorts (or any other 
division of the population such that individuals do not switch cohorts). 
Age is a discrete variable. See app.py or the deployed app for analytic 
expression of the relevant differential equations. 
 - No aging: The time horizon for the model is less than one year.
 - No vital statistics: No birth or death, aside from possible deaths due to 
disease.
 - The Exposed compartment are not infectious.
 
With four age cohorts, there are 12 free parameters to the model: 
- Incubation period, denoted 1/alpha
- Duration of infection (membership in the Infected compartment), denoted 
 1/gamma
- 10 parameters describing transmissibility between cohorts, assembled into 
a symmetric matrix beta. 

An entry in the matrix beta, say beta_12, expresses the contact rate
(number of contacts per day) between individuals in cohort 1 with
individuals in cohort 2, times the probability of transmission during contact. 
If I_2 denotes the number of infected individuals in cohort
2, out of N_2 total individuals in cohort 2, the expected rate of new
infections per susceptible individual in cohort 1 is beta_12 *
I_2/N_2. Which is to say, the infected individuals in cohort 2 contribute 
a term to the change in exposed individuals of cohort 1: 

$ dE_1/dt \ni S_1 * beta_{12} * I_2/N_2 .$

Furthermore, the matrix beta varies with time. 

As currently deployed, the app allows a given cohort's entries to beta to
be a constant value or zero, e.g. beta_a2 = BETA_CONST or beta_a2 = 0 for
all a, meaning that there is or is not transmission between the 2
cohort and any others. 

In the case that the 2 cohort (indexing from 0) is not mixing but all others 
are, and BETA_CONST=.2, 

beta = np.array([[0.2, 0.2, 0.0, 0.2],
                 [0.2, 0.2, 0.0, 0.2],
                 [0.0, 0.0, 0.0, 0.0],
                 [0.2, 0.2, 0.0, 0.2]])

Of course this restricted use of the transmission matrix leaves considerable 
room for more nuanced modeling.



"""

import numpy as np
import pandas as pd
import scipy.integrate

COMPARTMENTS = ['Susceptible', 'Exposed', 'Infected', 'Died or recovered']

# Example age-cohort structure and initial populations
COHORTS = ['0-4', '5-9', '10-19', '20+']
ROUGH_2017_POPULATION = [32., 28., 44., 88.]  # in millions
POPULATION_FRACTIONS = ROUGH_2017_POPULATION / np.sum(ROUGH_2017_POPULATION)
initial_infected = .0001
pop_0 = np.round(np.array([[f - initial_infected, 0, initial_infected, 0]
                               for f in POPULATION_FRACTIONS]), decimals=5)

# Model parameters:
INCUBATION_PERIOD = 3
DURATION_OF_INFECTION = 14
BETA_CONST = .2
BETA_ALL_COHORTS_MIXING = BETA_CONST * np.ones((len(COHORTS), len(COHORTS)))

class SEIRModel(object):
    """Class to solve an age-structured SEIR Compartmental model.

    Attributes: 
        betas: List of transmission matrices, one per epoch. (An epoch here
            denotes a period in which transmission rates are constant.) 
        epoch_end_times: List of days at which transmission rates change.
        alpha: Inverse incubation period
        gamma: Inverse duration of infection
        cohorts: List of names of (age) cohorts
        N_cohorts: Number of cohorts
        compartments: List of names of compartments
        N_compartments: Number of compartments (hard-coded to 4: S-E-I-R)
        s, e, i, r: Indices of the compartments in the state vector y(t). 

    External methods: 
        f: Function giving the rate of change in the state variable y(t).
        solve: Integrate the coupled differential equations.
        solve_to_dataframe: Solve and output a tidy dataframe.
    """

    def __init__(self, betas, epoch_end_times, alpha=1/INCUBATION_PERIOD,
                 gamma=1/DURATION_OF_INFECTION, cohorts=COHORTS):
        
        if len(epoch_end_times) != len(betas):
            raise ValueError('Each beta matrix requires an epoch end time.')
        self.epoch_end_times = sorted(epoch_end_times)
        self.betas = betas
        self.alpha = alpha
        self.gamma = gamma
        self.cohorts = cohorts
        self.N_cohorts = len(cohorts)
        
        # N.B. hard-coded values. The function f() assumes these four
        # compartments.
        self.compartments = COMPARTMENTS
        self.N_compartments = 4
        self.s, self.e, self.i, self.r = list(range(4))                  
                                                       
    def _beta(self, t):
        """Fetch the beta matrix for given time t."""
        for end_time, beta in zip(self.epoch_end_times, self.betas): 
            if t <= end_time: 
                return beta      
        
    def f(self, t, y):
        """Function giving the rate of change in the state variable y(t)."""
        y = y.reshape(self.N_cohorts, self.N_compartments)
        dy = np.zeros((self.N_cohorts, self.N_compartments))
        beta = self._beta(t)
        for a in range(self.N_cohorts):
            infection_rate = np.sum([beta[a,b] * y[b,self.i] / np.sum(y[b,:])
                                         for b in range(self.N_cohorts)])
            dy[a,self.s] = -y[a,self.s] * infection_rate
            dy[a,self.e] = (y[a,self.s] * infection_rate
                                - self.alpha * y[a, self.e])
            dy[a,self.i] = self.alpha * y[a, self.e] - self.gamma * y[a, self.i]
            dy[a,self.r] = self.gamma * y[a, self.i]
        return dy.flatten()
    
    def solve(self, y0):
        """Integrate the coupled differential equations."""
        sol = scipy.integrate.solve_ivp(
            self.f, (0, self.epoch_end_times[-1]), y0, 
            t_eval=np.arange(self.epoch_end_times[-1]))
        return sol.t, sol.y
    
    def solve_to_dataframe(self, y0):
        """Solve and output a tidy dataframe."""
        t, y = self.solve(y0)
        y = y.reshape(self.N_cohorts, self.N_compartments, len(t))
        y = np.sum(y, axis=0)
        df = pd.DataFrame(dict({'days': t}, **dict(zip(self.compartments, y))))
        df = pd.melt(df, id_vars=['days'], var_name='Group', value_name='pop')
        df['Date(s) of intervention'] = str(self.epoch_end_times[:-1])
        return df


def model_input(cohort_ranges):
	"""
	Function to calculuate transmission matrices and to enumerate their end
	period.

	Argument cohort_ranges: List of day range tuples denoting the period
        during which each cohort mixes (transmits disease) with the general
        population

	Returns a list of two lists for input into SEIRModel():
		1. A list of the transition matrices
		2. A list of epoch end times

	"""
    
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
	# iteratively alter the BETA_ALL_COHORTS_MIXING matrix by checking if a
    # given population should be removed for each time partition
	for j in range(len(epoch_tuples)): 

		# reset general popultation matrix for each epoch, and set
        # transmission to zero values for cohorts that have been removed
		res_mat = BETA_ALL_COHORTS_MIXING.copy()

		for i in range(len(cohort_ranges)):
			if _is_removed(cohort_ranges[i], epoch_tuples[j]):
				res_mat[i, :] = 0
				res_mat[:, i] = 0


		transition_matrices.append(res_mat)

	# Remove the first delimiter, which is a start day (not an end day)
	epoch_ends = epoch_delims[1:]

	return [transition_matrices, epoch_ends]
