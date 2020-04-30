import streamlit as st
import numpy as np
import pandas as pd
import model

# Sidebar and the subsequent parameter definitions

st.sidebar.markdown(
    """
    ## Parameters and assumptions
    --------
    """
)

region_option = st.sidebar.selectbox(
    'Region', 
    ["North America", "Africa", "Europe"]
)


pop_lookup = {
    "North America": [90., 77., 140., 62.],
    "Africa"       : [679., 316., 298., 47.],
    "Europe"       : [158., 136., 310., 142.]
}

cohort_ages = ['0-18', '19-34', '35-64', '65+']
population = pop_lookup[region_option]
pop_fractions = population / np.sum(population)

# Visualization specification (default)

def _vega_default_spec(
        color=None, 
        scale=[0.0, 1.0], 
        evolution_length=180,
        start_day=0
    ):
    spec={
        'mark': {'type': 'line', 'tooltip': True},
        'encoding': {
            'x': {
                'field': 'days', 
                'type': 'quantitative',
                'axis': {'title': ""},
                'scale': {'domain': [start_day, evolution_length]}
            },
            'y': {
                'field': 'pop', 
                'type': 'quantitative',
                'axis': {'title': ""},
                'scale': {'domain': scale}
            },
          'color': {'field': 'Group', 'type': 'nominal'}
        }
    }
    if color:
        spec['encoding'].pop('color')
        spec['mark'].update({'color': color})
    return spec


# Introductory text
st.title('Illustrating stacked NPIs in the browser')
st.subheader(
	'A simple age-structured, compartmental model.'
)

st.markdown(""" 

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

The model is run . 

""")

st.subheader('An illustration of *Flattening the Curve*')

st.write("""

The simplest version of the model is a constant transmission matrix.  This is
effectively the standard SEIR model, which has been widelyl used to show the
benefits of acting early in a pandemic to 'flatten the curve.' 

""")

mixing_range = st.slider(
    'Period of shelter-in-place for whole population',
    0, 300, (80, 140)
)

cohort_ranges = np.repeat([mixing_range], 4, axis=0)

betas, epoch_end_times = model.model_input(
    cohort_ranges, 
    seclusion_scale=0.001
)

# initial infected is 0.01% of the population
pop_0 = np.array([
    [f - 0.0001, 0, 0.0001, 0] for f in pop_fractions
])

st.write(pop_0)

# res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
# df = res.solve_to_dataframe(model.pop_0.flatten())

# st.vega_lite_chart(x`
#   data=df[df["Group"] == "Infected"],
#     spec=_vega_default_spec(
#       color="#e45756",
#       scale=[0.0, 0.6]
#     ),
#   use_container_width=True
# )



# st.sidebar.markdown("-------")

# # Create a series of sliders for the time range of each cohort
# # TODO: There is probably a more elegant way to do this.
# first_range = st.slider(
# 	'Period of NPI for [%s] cohort:' % (model.COHORTS[0]),
# 	0, 180, (115, 180)
# )

# second_range = st.slider(
# 	'Period of NPI for [%s] cohort:' % (model.COHORTS[1]),
# 	0, 180, (80, 180)
# )

# third_range = st.slider(
# 	'Period of NPI for [%s] cohort:' % (model.COHORTS[2]),
# 	0, 180, (10, 80)
# )

# fourth_range = st.slider(
# 	'Period of NPI for [%s] cohort:' % (model.COHORTS[3]),
# 	0, 180, (0, 115)
# )

# cohort_ranges = [
# 	first_range,
# 	second_range,
# 	third_range,
# 	fourth_range
# ]


# # Generate the beta matrices and epoch ends:
# betas, epoch_end_times = model.model_input(
# 	cohort_ranges, 
# 	seclusion_scale=0.001
# )

# res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
# df, death_df = res.solve_to_dataframe(model.pop_0.flatten())

# colors = dict(zip(model.COMPARTMENTS, ["#4c78a8", "#f58518", "#e45756", "#72b7b2"]))

# def _vega_default_spec(
#         color=None, 
#         scale=[0.0, 1.0], 
#         evolution_length=180,
#         start_day=0
#     ):
#     spec={
#         'mark': {'type': 'line', 'tooltip': True},
#         'encoding': {
#             'x': {
#                 'field': 'days', 
#                 'type': 'quantitative',
#                 'axis': {'title': ""},
#                 'scale': {'domain': [start_day, evolution_length]}
#             },
#             'y': {
#                 'field': 'pop', 
#                 'type': 'quantitative',
#                 'axis': {'title': ""},
#                 'scale': {'domain': scale}
#             },
#         	'color': {'field': 'Group', 'type': 'nominal'}
#         }
#     }
#     if color:
#         spec['encoding'].pop('color')
#         spec['mark'].update({'color': color})
#     return spec

# if show_option == "All":
#     st.subheader(show_option)
#     st.vega_lite_chart(
#         data=df,
#         spec=_vega_default_spec(),
#         use_container_width=True
#     )
    
# elif show_option == "Infected":
# 	st.subheader(show_option)
# 	st.vega_lite_chart(
# 		data=df[df["Group"] == show_option],
#         spec=_vega_default_spec(
#         	color=colors[show_option],
#         	scale=[0.0, 0.6]
#         ),
# 		use_container_width=True
# 	)

# elif show_option in model.COMPARTMENTS:
# 	st.subheader(show_option)
# 	st.vega_lite_chart(
# 		data=df[df["Group"] == show_option],
#         spec=_vega_default_spec(color=colors[show_option]),
# 		use_container_width=True
# 	)

# st.dataframe(death_df.style.highlight_max(axis=1, color="#e39696"))

# st.markdown(""" 

# The parameter values are **especially** wrong. We just made up some numbers. These
# should be set from empirical evidence.

# Note that there are _a lot_ of free parameters.  It is possible to create
# basically any model outcome from the choice of these interdependent
# parameters.  This web app is valuable insofar as it illustrates the broad
# trends and features of the standard compartmental models, when interacting
# cohorts are introduced.  For example, there can be two or even more waves of
# infection.  That's like, basically, that's about all we can get from this.

# """)


# # Policy scenarios through illustration of the behavior of differential
# # equations.

# st.subheader('Flattening the curve, illustrated.')

# st.write(""" 

# The concept of *flattening the curve* is now well-understood. If you delay the
# start of the NPI, and allow mixing for the early period, you can observe the
# spike in infection. (This corresponds to moving the start period of the NPI
# from Day 0 to, say, Day 20.)  This further underscores the value of immediate
# action in a pandemic. The curve remains flat if the NPI is enacted, with only
# a small bump in infections when the NPI is removed.  

# """)

# mixing_range = st.slider(
# 	'Period of intervention for whole population',
# 	0, 180, (0, 90)
# )

# # All cohorts mix at the same time.
# cohort_ranges = np.repeat([mixing_range], 4, axis=0)
# betas, epoch_end_times = model.model_input(cohort_ranges, seclusion_scale=0.05)

# res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
# df, death_df = res.solve_to_dataframe(model.pop_0.flatten())

# st.vega_lite_chart(
# 	data=df[df["Group"] == "Infected"],
#     spec=_vega_default_spec(
#     	color="#e45756",
#     	scale=[0.0, 0.6]
#     ),
# 	use_container_width=True
# )

# st.write(death_df)

# st.subheader('Different NPIs for different sub-populations.')

# st.markdown(""" 

# What happens if we allow some people to re-enter the economy before others?
# For example, suppose we allow young people to go to school before allowing
# *everyone* to mix. The infection rate won't necessarily change, but the
# hospitalizations may not spike as high, since younger people don't get so
# sick.

# Consider a made-up country with a very young population &mdash; with 75% of
# the population under a certain age, say, under the age of 35.  Assume further
# that the severity of the illness is much worse in older people, requiring 
# hospitalization at a much greater rate than for young people that contract the
# illness.  Returning to business as usual requires hospitalization to remain
# under a certain threshold.

# The following graph illustrates the concept of phased re-introduction of two
# subpopulations with different hospitalization usage.  The parameters are made
# up to clearly show the unique features of this cohort model.  Specifically,
# this model illustrates NPIs with differential treatment of sub-populations
# with different characteristics, but that interact with the evolution of the
# illness in all other sub-populations.

# Our suggestion is to play with the sliders to introduce the older population
# earlier, noting when the severe infection rate (in orange) exceeds an
# arbitrary threshold (like 0.1).

# """)

# population = [50, 20, 20, 10]  # in millions
# population_fractions = population / np.sum(population)
# initial_infected = 0.002

# # Create an initial population array, with 4 (S, E, I, and R) compartments
# pop_0 = np.round(
#     np.array([
#         [f - initial_infected, 0, initial_infected, 0] 
#         for f in population_fractions
#     ]), 
#     decimals=5
# )

# young_range = st.slider(
# 	'Period of NPI for younger cohort:',
# 	0, 100, (0, 16)
# )

# old_range = st.slider(
# 	'Period of NPI for older cohort:',
# 	0, 100, (0, 60)
# )

# cohort_ranges = [young_range, young_range, old_range, old_range]

# betas, epoch_end_times = model.model_input(
#     cohort_ranges, 
#     seclusion_scale=0.03,
#     evolution_length=130
# )

# res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
# df, death_df, y = res.solve_to_dataframe(pop_0.flatten(), detailed_output=True)

# young = pd.DataFrame(
#     {
#         "infection1": y[0][2],
#         "infection2": y[1][2]
#     }
# )

# old = pd.DataFrame(
#     {
#         "infection1": y[2][2],
#         "infection2": y[3][2]
#     }
# )

# df = pd.DataFrame(
# 	{
# 		"infection": young["infection1"] + young["infection2"] + old["infection1"] + old["infection2"],
# 		"severe": (young["infection1"] + young["infection2"]) * 0.15 + (old["infection1"] + old["infection2"]) * 0.85
# 	}
# )

# st.area_chart(df)

# ## STACKING NPIs with regional adjustment

# st.sidebar.markdown(
#     """
#     **Illustration #4**: Stacking NPIs with regional adjustment

#     """
# )

# region_option = st.sidebar.selectbox(
#     'Region', 
#     ["North America", "Africa", "Europe"]
# )

# start_date = st.sidebar.number_input(
#     label='Start day',
#     value=int(),
#     min_value=int(0),
#     max_value=int(60),
#     step=int()
# )

# st.sidebar.markdown("-------")

# st.subheader('Stacking NPIs: An illustration.')

# st.markdown(""" 

# It is challenging to distinguish the effect of individual public health and
# social measures (PHSMs) on the rate of COVID-19 transmission within a
# population (reproductive number R0); effectiveness depends on how fully
# communities adopt and adhere to PHSMs, additional interventions they are
# combined with, and other variables like family size and level of
# intergenerational contact within a community. Still, evidence does show that
# PHSMs are more effective when implemented in combination, or “stacked,” than
# when implemented individually. 

# The following scenario expressly illustrates this concept, when the PHSMs are
# tied to partitions of the general population (i.e., a person can only belong
# to one category). This is the primary limitation of this illustration, one
# that can be mitigated with more precise definitions of NPIs. There is the
# added feature of chosing the initial conditions based on region.  See the
# sidebar for options.  The default is the United States.  

# """)

# pop_lookup = {
#     "North America": [90., 77., 140., 62.],
#     "Africa"       : [679., 316., 298., 47.],
#     "Europe"       : [158., 136., 310., 142.]
# }


# cohort_ages = ['0-18', '19-34', '35-64', '65+']
# pop_2020 = pop_lookup[region_option]
# pop_fractions = pop_2020 / np.sum(pop_2020)


# st.markdown('''
#     > **%s** (2020) 
#     >> *Use sidebar to change region. Data from UN.*
# ''' % region_option)

# initial_populations = np.round(
#     np.array([
#         [f - initial_infected, 0, initial_infected, 0] for f in pop_fractions
#     ]), 
#     decimals=5
# )

# first_npi = st.slider(
#     'Schools closed.',
#     0, 180, (0, 20)
# )

# second_npi = st.slider(
#     'Restaurants, universities, and bars closed.',
#     0, 180, (0, 60)
# )

# third_npi = st.slider(
#     'Offices closed.',
#     0, 180, (0, 80)
# )

# fourth_npi = st.slider(
#     'Senior citizens remained quarantined.',
#     0, 180, (0, 120)
# )

# npi_ranges = [
#     first_npi,
#     second_npi,
#     third_npi,
#     fourth_npi
# ]

# betas, epoch_end_times = model.model_input(
#     npi_ranges, 
#     seclusion_scale=0.02
# )

# res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
# df, death_df = res.solve_to_dataframe(initial_populations.flatten())
# infected = df[df["Group"] == "Infected"]

# st.vega_lite_chart(
#     data=infected[infected["days"] > start_date],
#     spec=_vega_default_spec(
#         color="#e45756",
#         scale=[0.0, 0.6],
#         start_day=start_date
#     ),
#     use_container_width=True
# )
