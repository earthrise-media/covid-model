import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import scipy.integrate
import time

COMPARTMENTS = ['Susceptible', 'Exposed', 'Infected', 'Died or recovered']
AGE_RANGES = ['0-4', '5-9', '10-19', '20+']
ROUGH_2017_POPULATION = [32., 28., 44., 88.]  # in millions, per Wikipedia (to check)
POPULATION_FRACTIONS = ROUGH_2017_POPULATION / np.sum(ROUGH_2017_POPULATION)

# Initial model parameters:
INCUBATION_PERIOD = 3
DURATION_OF_INFECTION = 14

# Transmission rates from infected to susceptible compartments, by age range, and under various assumptions.
# For example, the 1,2 entry denotes the transmission rate (contact rate * probability of transmission) from the 
# '5-9' to the '0-4' cohort. 

# Last cohort removed from general population (no transmission)
beta_last_cohort_out = np.array([
	[0.3, 0.3, 0.3, 0.0],
	[0.3, 0.3, 0.3, 0.0],
	[0.3, 0.3, 0.3, 0.0],
	[0.0, 0.0, 0.0, 0.0]
])

# All cohorts in
beta_all_in = np.array([
	[0.3, 0.3, 0.3, 0.3],
	[0.3, 0.3, 0.3, 0.3],
	[0.3, 0.3, 0.3, 0.3],
	[0.3, 0.3, 0.3, 0.3]
])

# Initial compartment populations
initial_infected = .0004
pop_0 = np.round(np.array([[f - initial_infected, 0, initial_infected, 0] for f in POPULATION_FRACTIONS]), decimals=5)

# Introductory text
st.title('Staged reintroduction of age cohorts')

st.write(
	"This model estimates the impact of re-introducing age-defined cohorts" \
	"back into general population. The model supports any number of cohorts, " \
	"but the interaction parameters increase exponentially. The complexity may" \
	"be valuable; and the parameters are empirically derived. This is not a " \
	"simulation. Rather, the output is the result from a set of differential equations" \
	" -- a multidimensional generalization of the standard SEIR compartmental model."
)

st.latex(r'''
	\begin{array}{lll}
		\frac{dS}{dt} &=& -\frac{\beta I S}{N} \\
		\\
		\frac{dI}{dt} &=& \frac{\beta I S}{N} - \gamma I\\
		\\
		\frac{dR}{dt} &=& \gamma I \\

	\end{array}{}
	''')


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
            dy[a,self.s] = -y[a,self.s] * np.sum([beta[a,b] * y[b,self.i] for b in range(self.N_ages)])
            dy[a,self.e] = y[a,self.s] * np.sum([beta[a,b] * y[b,self.i] for b in range(self.N_ages)]) - self.alpha * y[a, self.e]
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
show_option = st.sidebar.selectbox(
    'Population to show',
    [
    	"Infected", 
    	"Died or recovered"
    ]
)


values = st.slider(
	'Introduction period of [20+] cohort:',
	0, 180, (100, 145)
)

betas = [beta_last_cohort_out, beta_all_in, beta_last_cohort_out]
df = SEIRModel(alpha=3, betas=betas, epoch_end_times=(values[0], values[1], 180)).solve_to_dataframe(pop_0.flatten())
plot_group = df[df["Group"] == show_option]

if show_option == "Infected":
	st.subheader(show_option)
	st.vega_lite_chart(
		plot_group, 
		{
			'layer': [
				{
					'mark': {'type': 'line', 'tooltip': True},
					'encoding': {
						'x': {
							'field': 'days', 
							'type': 'quantitative',
							'axis': {'title': ""}
						},
						'y': {
							'field': 'pop', 
							'type': 'quantitative',
							'axis': {'title': ""}
						}
					}
				}
			]
		}, 
		use_container_width=True
	)

elif show_option == "Died or recovered":

	st.vega_lite_chart(
		plot_group, 
		{
			'layer': [
				{
					'mark': {'type': 'line', 'tooltip': True},
					'encoding': {
						'x': {
							'field': 'days', 
							'type': 'quantitative',
							'axis': {'title': ""}
						},
						'y': {
							'field': 'pop', 
							'type': 'quantitative',
							'axis': {'title': ""}
						}
					}
				}
			]
		}, 
		use_container_width=True
	)