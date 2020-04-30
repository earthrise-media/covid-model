import altair as alt
import streamlit as st
import numpy as np
import pandas as pd

import model

# Sidebar and the subsequent parameter definitions

st.sidebar.markdown(
	"""
	## Parameters and assumptions
	--------

	**Initial conditions**
	"""
)

region_option = st.sidebar.selectbox(
	'Region', 
	["North America", "Africa", "Europe"]
)

initial_infected = st.sidebar.number_input(
	label='Initial rate of infection (percent)',
	min_value=0.01,
	max_value=70.,
	step=0.01
)

# The sidebar number input is limited in significant digits.  To ensure that
# the initial conditions reflect reality, allow for lower initial infection
# rates by dividing by 100 and treating the input as percents.
initial_infected /= 100

st.sidebar.markdown(
	"""
	--------

	**Model parameters**
	"""
)

mixing_constant = st.sidebar.number_input(
	label='Transmission rate when cohort is mixing',
	value=0.4,
	min_value=0.01,
	max_value=0.9,
	step=0.01
)

seclusion_constant = st.sidebar.number_input(
	label='Transmission rate when chort is not mixing',
	value=0.03,
	min_value=0.01,
	max_value=0.9,
	step=0.01
)

duration_period = st.sidebar.number_input(
	label='Duration of infection (in days)',
	value=int(14),
	min_value=int(1),
	max_value=int(100),
	step=int()
)

incubation_period = st.sidebar.number_input(
	label='Incubation period (in days)',
	value=int(5),
	min_value=int(1),
	max_value=int(100),
	step=int()
)


st.sidebar.markdown(
	"""
	--------

	**Visualization parameters**
	"""
)

start_date = st.sidebar.number_input(
	label='Start day (for visualization purposes only)',
	value=int(),
	min_value=int(0),
	max_value=int(60),
	step=int()
)


# Derived parameters from user settings


pop_lookup = {
	"North America": [90., 77., 140., 62.],
	"Africa"       : [679., 316., 298., 47.],
	"Europe"       : [158., 136., 310., 142.]
}

cohort_ages = ['0-18', '19-34', '35-64', '65+']
population = pop_lookup[region_option]
pop_percents = 100 * np.array(population) / np.sum(population)

pop_0 = np.array([[f * (1 - 2 * initial_infected), f * initial_infected,
					   f * initial_infected, 0] for f in pop_percents])


death_rates_by_cohort = [0.0001, 0.004, 0.004, 0.06]

# Introductory text
st.title('Illustrating stacked NPIs in the browser')

st.markdown(""" 

> **If policy makers are using prevailing models and *also* talking about
lifting NPIs based on age, then they are using this model**, whether they know
it or not.

This model should be used as a heuristic to illustrate the effect of lifting
NPIs for certain sub-populations. The numbers are not projections, but rather
indications of the directional effects of stacked policies.  The authors of
this model and visualization are _not_ epidemiologists.  At best, we are
armchair epidemiologists &mdash; which is pretty bad.

We extended a standard SEIR model to incorporate the interactions between
cohorts, starting with four cohorts.  The cohorts are a partition of the 
population &mdash; a complete and non-overlapping covering of the full
population.  For example, the cohorts could be defined by age.  We use the age
specification in this example, but it's certainly not necessary to define the
cohorts by age.

Our multidimensional generalization of the standard SEIR compartmental model
is represented as follows:

""")

eqnarray = r"""
	\begin{array}{lll}
		\frac{dS_a}{dt} &=& -S_a\; \sum_b \beta_{ab}(t) I_b/ N_b \\
		\\
		\frac{dE_a}{dt} &=& S_a\; \sum_b \beta_{ab}(t) I_b/ N_b - \alpha E_a\\
		\\
		\frac{dI_a}{dt} &=& \alpha E_a - \gamma I_a\\
		\\
		\frac{dR_a}{dt} &=& \gamma I_a \\

	\end{array}{}
"""

st.latex(eqnarray)

st.write("""

The subscripts (a,b) index the (age) cohorts, while alpha, beta, and gamma are
the inverse incubation period, the transmissibility between cohorts, and
the inverse duration of infection, respectively.

Note that we use age to partition the general population. The cohorts need not
be defined by age, as long as the categories are non-overlapping and complete.
Another partition may be urban/rural or service/manufacturing/student/other.
Age is convenient and relevant way to do this, since the NPIs we examine are
generally defined by age.

""")  

st.subheader('Initial conditions and parameter assumptions')

st.write("""

The following assumptions and parameters are used as inputs to the
age-structured models. The number of people in each age cohort is part of the
initial conditions for the model. We use the following regional data on age
distributions from the United Nations:

""") 

df = pd.DataFrame(
	pop_lookup,
	index=cohort_ages
).T

df["Total (in millions)"] = df.sum(axis=1)

st.write(df)

st.write("""

Based on this information and **the parameter selections in the sidebar** the
initial conditions are as follows:

Initial conditions for **%s** (2020) as a percentage of the population:

""" % region_option)

display_pop_0 = pd.DataFrame(
	pop_0,
	index=cohort_ages,
	columns=model.COMPARTMENTS
)

st.write(display_pop_0)

st.write("""

If an age cohort is not mixing, then its transmission rate between any other
cohort is assumed to be **%s**. If the cohort is mixing when, for example, the
NPI is lifted, then the transmission rate is **%s**. These entries represent
the cell values in the model's *&beta;* matrix. 

Other parameter assumptions include the duration of infection (**%s** days)
and the incubation period (**%s** days).  These parameters can be adjusted in
the sidebar, and the graphs, tables, and text on this page will automatically
re-render according to the new set of assumptions.

There are some parameters that are not adjustable on this page.  These include
the death rates by age and the lag in death (after infection).  We have
consulted with Dr. Lee and we currently use the assumption that the lag is
**14** days, and the death rates are .


"""

% (seclusion_constant, mixing_constant, duration_period, incubation_period)

)

st.subheader('An illustration of *Flattening the Curve*')

st.write("""

The simplest version of the model is a constant transmission matrix.  This is
effectively the standard SEIR model, which has been widely used to show the
benefits of acting early in a pandemic to 'flatten the curve.' 

Note that the region will not make a difference for the infection curve for
this illustration, since the age distribution does not matter &mdash; all ages
have been collapsed into a single, total population.

""")

mixing_range = st.slider(
	'Period of shelter-in-place for whole population',
	0, 300, (10, 160)
)

cohort_ranges = np.repeat([mixing_range], 4, axis=0)

betas, epoch_end_times = model.model_input(
	cohort_ranges, 
	seclusion_scale=seclusion_constant,
	evolution_length=300,
	mixing_beta_const=mixing_constant
)

res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
df = res.solve_to_dataframe(pop_0.flatten())

infected = df[(df["Group"] == "Infected")]

chart = alt.Chart(infected[infected["days"] > start_date]).mark_line(
	color="#e45756").encode(
		x=alt.X('days', axis=alt.Axis(title='Days')),
		y=alt.Y('pop', axis=alt.Axis(title='Percent infected'), scale=alt.Scale(domain=(0,50))))

st.altair_chart(chart, use_container_width=True)

st.subheader('Stacking NPIs: An illustration.')

st.write("""

It is challenging to distinguish the effect of individual public health and
social measures (PHSMs) on the rate of COVID-19 transmission within a
population (reproductive number R0); effectiveness depends on how fully
communities adopt and adhere to PHSMs, additional interventions they are
combined with, and other variables like family size and level of
intergenerational contact within a community. Still, evidence does show that
PHSMs are more effective when implemented in combination, or “stacked,” than
when implemented individually. 

The following scenario expressly illustrates this concept, when the PHSMs are
tied to partitions of the general population (i.e., a person can only belong
to one category). This is the primary limitation of this illustration, one
that can be mitigated with more precise definitions of NPIs. There is the
added feature of chosing the initial conditions based on region.  See the
sidebar for options.

We recognize the wild parameter assumptions, as well as the instability of the
implicit projections.  This should mainly be used to illustrate the pattern of
the infection &mdash; the surge and resurgence &mdash; rather than the number
of projected deaths.  This number is tallied at the end to demonstrate the
impact of prolonged, high infection rates rather than momentary spikes.  It
is, in effect, a way to understand the area under the curve without 
additional lines on the chart.

The reason why we believe that this is so important is that this is the
model that policy makers are currently using, whether they know it or not. 
The parameter specifications may be different.  However, almost all of the
prevailing models are based on the compartmental SEIR model or some slight
variant.  We've reviewed dozens of these models in preparation of this web
app.  The model displayed here is just a cohort-based generalization.  **If
policy makers are using prevailing models and *also* talking about lifting
NPIs based on age, then they are using this model**.  This web app is just an
interactive visualization.

""")


first_npi = st.slider(
	'Schools closed.',
	0, 300, (10, 45)
)

second_npi = st.slider(
	'Restaurants, universities, and bars closed.',
	0, 300, (10, 125)
)

third_npi = st.slider(
	'Offices closed.',
	0, 300, (10, 150)
)

fourth_npi = st.slider(
	'Senior citizens remained quarantined.',
	0, 300, (10, 170)
)

npi_ranges = [
	first_npi,
	second_npi,
	third_npi,
	fourth_npi
]

betas, epoch_end_times = model.model_input(
	npi_ranges, 
	seclusion_scale=seclusion_constant,
	evolution_length=300,
	mixing_beta_const=mixing_constant
)


res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
df, y = res.solve_to_dataframe(pop_0.flatten(), detailed_output=True)
infected = df[df["Group"] == "Infected"]

chart = alt.Chart(infected[infected["days"] > start_date]).mark_line(
	color="#e45756").encode(
		x=alt.X('days', axis=alt.Axis(title='Days')),
		y=alt.Y('pop', axis=alt.Axis(title='Percent infected'), scale=alt.Scale(domain=(0,50))))


st.markdown('Infections in **%s**' % (region_option))

st.altair_chart(chart, use_container_width=True)


# Calculate the aggregate deaths by cohort.  Position matters!!!  The
# "Died or recovered" compartment is in the final position. Then,
# also, collect the final number in the evolution, hence the dual "-1"
# indices.
final_died_or_recovered = [x[-1][-1] for x in y]
number_died_or_recovered = np.multiply(final_died_or_recovered, population)/100
death_numbers = np.multiply(number_died_or_recovered, death_rates_by_cohort)*1000000

death_df = pd.DataFrame([death_numbers], columns = cohort_ages, index=["Deaths        "])
death_df["Total"] = sum(death_numbers)

display_df = np.round(death_df).astype('int32')

st.write(display_df.style.highlight_max(axis=1, color="#e39696"))

st.markdown(""" 

> **Note**: These projections are more likely incorrect than correct.  Please
be advised.  The objective is to visualize the patterns of infection and
recovery &mdash; an interactive visualization of the differential equations
that underlie the prevailing models.

""")
