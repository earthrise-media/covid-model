import streamlit as st
import numpy as np
import pandas as pd
import scipy.integrate

COMPARTMENTS = ['Susceptible', 'Exposed', 'Infected', 'Died or recovered']
AGE_RANGES = ['0-4', '5-9', '10-19', '20+']
ROUGH_2017_POPULATION = [32., 28., 44., 88.]  # in millions, per Wikipedia (to check)
POPULATION_FRACTIONS = ROUGH_2017_POPULATION / np.sum(ROUGH_2017_POPULATION)

# Initial model parameters:
INCUBATION_PERIOD = 3
DURATION_OF_INFECTION = 14


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
		[0.1, 0.1, 0.1, 0.1],
		[0.1, 0.1, 0.1, 0.1],
		[0.1, 0.1, 0.1, 0.1],
		[0.1, 0.1, 0.1, 0.1]
	])

	# Partition total range in to it's unique end periods
	flat_dates = [v for sublist in cohort_ranges for v in sublist]

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


# Initial compartment populations
initial_infected = .0004
pop_0 = np.round(np.array([[f - initial_infected, 0, initial_infected, 0] for f in POPULATION_FRACTIONS]), decimals=5)

# Introductory text
st.title('Staged reintroduction of age cohorts')

st.write(
	"This model estimates the impact of re-introducing age-defined cohorts" \
	"back into general population. The model supports any number of cohorts, " \
	"but the interaction parameters increase exponentially. The complexity may " \
	"be valuable; and the parameters are empirically derived. This is not a " \
	"simulation. Rather, the output is the result from a set of differential equations" \
	" -- a multidimensional generalization of the standard SEIR compartmental model."
)

st.latex(r'''
	\begin{array}{lll}
		\frac{dS_a}{dt} &=& -S_a\; \sum_b \beta_{ab}(t) I_b/ N_b \\
		\\
		\frac{dE_a}{dt} &=& S_a\; \sum_b \beta_{ab}(t) I_b/ N_b - \alpha E_a\\
		\\
		\frac{dI_a}{dt} &=& \alpha E_a - \gamma I_a\\
		\\
		\frac{dR_a}{dt} &=& \gamma I_a \\

	\end{array}{}
	''')

st.write("The subscripts (a,b) index the age cohorts, while "
         "alpha, beta, and gamma are the inverse incubation period, "
         "the transmissibility between age cohorts, and the inverse duration "
         "of infection, respectively.")  

## DISPLAY INITIAL CONDITIONS
st.write(
	"Initial compartment population fractions " \
	"(a row denotes compartments for one age range)."
)

df = pd.DataFrame(
	pop_0, 
	index=AGE_RANGES
)
df.columns = COMPARTMENTS
st.write(df)

## RUN MODEL 

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

# Sidebar
show_option = st.sidebar.selectbox('Population to show', ["All"] + COMPARTMENTS)

# Create a series of sliders for the time range of each cohort
# TODO: There is probably a more elegant way to do this.
first_range = st.slider(
	'Period of mixing for [%s] cohort:' % (AGE_RANGES[0]),
	0, 180, (0, 180)
)

second_range = st.slider(
	'Period of mixing for [%s] cohort:' % (AGE_RANGES[1]),
	0, 180, (30, 180)
)

third_range = st.slider(
	'Period of mixing for [%s] cohort:' % (AGE_RANGES[2]),
	0, 180, (70, 100)
)

fourth_range = st.slider(
	'Period of mixing for [%s] cohort:' % (AGE_RANGES[3]),
	0, 180, (90, 180)
)

cohort_ranges = [
	first_range,
	second_range,
	third_range,
	fourth_range
]


# Generate the beta matrices and epoch ends:
betas, epoch_end_times = model_input(cohort_ranges)

df = SEIRModel(alpha=3, betas=betas, epoch_end_times=epoch_end_times).solve_to_dataframe(pop_0.flatten())

colors = dict(zip(COMPARTMENTS, ["#4c78a8", "#f58518", "#e45756", "#72b7b2"]))

def _vega_default_spec(color=None):
    spec={
        'mark': {'type': 'line', 'tooltip': True},
        'encoding': {
            'x': {
                'field': 'days', 
                'type': 'quantitative',
                'axis': {'title': ""},
                'scale': {'domain': [0, 180]}
            },
            'y': {
                'field': 'pop', 
                'type': 'quantitative',
                'axis': {'title': ""},
                'scale': {'domain': [0.0, 1.0]}
            },
        'color': {'field': 'Group', 'type': 'nominal'}
        }
    }
    if color:
        spec['encoding'].pop('color')
        spec['mark'].update({'color': color})
    return spec

if show_option == "All":
    st.subheader(show_option)
    st.vega_lite_chart(
        data=df,
        spec=_vega_default_spec(),
        use_container_width=True
    )
    
elif show_option in COMPARTMENTS:
	st.subheader(show_option)
	st.vega_lite_chart(
		data=df[df["Group"] == show_option],
        spec=_vega_default_spec(color=colors[show_option]),
		use_container_width=True
	)
