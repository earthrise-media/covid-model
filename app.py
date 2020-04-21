import streamlit as st
import numpy as np
import pandas as pd
import model


# Introductory text
st.title('This model is wrong.')
st.subheader(
	'All models are wrong. Some are useful.'
)

text = """ 

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

"""

st.markdown(text)

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

text = """

The subscripts (a,b) index the (age) cohorts, while alpha, beta, and gamma are
the inverse incubation period, the transmissibility between cohorts, and
the inverse duration of infection, respectively.

"""

st.write(text)  

## DISPLAY INITIAL CONDITIONS

text = """

The table below represents the initial conditions at Day 0 for each age
cohort.  We pulled the age distribution for the United States in 2017.

"""

st.write(text)

df = pd.DataFrame(
	model.pop_0, 
	index=model.COHORTS
)
df.columns = model.COMPARTMENTS
st.write(df)

## RUN MODEL 


# Sidebar
show_option = st.sidebar.selectbox(
	'Population to show', 
	["Infected", "Died or recovered", "Exposed", "Susceptible", "All"]
)

# Create a series of sliders for the time range of each cohort
# TODO: There is probably a more elegant way to do this.
first_range = st.slider(
	'Period of mixing for [%s] cohort:' % (model.COHORTS[0]),
	0, 180, (0, 180)
)

second_range = st.slider(
	'Period of mixing for [%s] cohort:' % (model.COHORTS[1]),
	0, 180, (0, 150)
)

third_range = st.slider(
	'Period of mixing for [%s] cohort:' % (model.COHORTS[2]),
	0, 180, (72, 180)
)

fourth_range = st.slider(
	'Period of mixing for [%s] cohort:' % (model.COHORTS[3]),
	0, 180, (115, 180)
)

cohort_ranges = [
	first_range,
	second_range,
	third_range,
	fourth_range
]


# Generate the beta matrices and epoch ends:
betas, epoch_end_times = model.model_input(cohort_ranges, seclusion_scale=0.01)

res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
df, death_df = res.solve_to_dataframe(model.pop_0.flatten())

colors = dict(zip(model.COMPARTMENTS, ["#4c78a8", "#f58518", "#e45756", "#72b7b2"]))

def _vega_default_spec(
        color=None, 
        scale=[0.0, 1.0], 
        evolution_length=180
    ):
    spec={
        'mark': {'type': 'line', 'tooltip': True},
        'encoding': {
            'x': {
                'field': 'days', 
                'type': 'quantitative',
                'axis': {'title': ""},
                'scale': {'domain': [0, evolution_length]}
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

if show_option == "All":
    st.subheader(show_option)
    st.vega_lite_chart(
        data=df,
        spec=_vega_default_spec(),
        use_container_width=True
    )
    
elif show_option == "Infected":
	st.subheader(show_option)
	st.vega_lite_chart(
		data=df[df["Group"] == show_option],
        spec=_vega_default_spec(
        	color=colors[show_option],
        	scale=[0.0, 0.4]
        ),
		use_container_width=True
	)

elif show_option in model.COMPARTMENTS:
	st.subheader(show_option)
	st.vega_lite_chart(
		data=df[df["Group"] == show_option],
        spec=_vega_default_spec(color=colors[show_option]),
		use_container_width=True
	)

st.dataframe(death_df.style.highlight_max(axis=1, color="#e39696"))

text = """ 

The parameter values are **especially** wrong. We just made up some numbers. These
should be set from empirical evidence.

Note that there are _a lot_ of free parameters.  It is possible to create
basically any model outcome from the choice of these interdependent
parameters.  This web app is valuable insofar as it illustrates the broad
trends and features of the standard compartmental models, when interacting
cohorts are introduced.  For example, there can be two or even more waves of
infection.  That's like, basically, that's about all we can get from this.

"""

st.markdown(text)






# Introductory text
st.title('Toward re-opening the economy')

st.subheader('An exploration of differential equations.')

text = """ 

The concept of *flattening the curve* is now well-understood. Drag the start
period of mixing to the right to delay mixing for everyone. This reflects a
shelter-in-place order for everyone. As you drag the start period to the
right, the curve flattens.  The red period, in effect, is an open economy.

"""

st.write(text)

mixing_range = st.slider(
	'Period of mixing for everyone',
	0, 180, (0, 180)
)

cohort_ranges = np.repeat([mixing_range], 4, axis=0)
betas, epoch_end_times = model.model_input(cohort_ranges)

res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
df, death_df = res.solve_to_dataframe(model.pop_0.flatten())

st.vega_lite_chart(
	data=df[df["Group"] == "Infected"],
    spec=_vega_default_spec(
    	color="#e45756",
    	scale=[0.0, 0.6]
    ),
	use_container_width=True
)

st.write(death_df)

# st.write(epoch_end_times)


text = """ 

What happens if we allow some people to re-enter the economy before others?
For example, suppose we allow young people to go to school before allowing
*everyone* to mix. The infection rate won't necessarily change, but the
hospitalizations may not spike as high, since younger people don't get so
sick.

"""

st.markdown(text)

text = """

Consider a made-up country with a very young population &mdash; with 75% of
the population under the age of 35 and a population of 100M.  However, also
assume that the severity of the illness is much worse in older people,
requiring hospitalization; and that the severe infection rate is the most
important consideration for re-opening the economy.

"""

st.markdown(text)

COHORTS = ['0-18', '19-34', '35-64', '65+']
ROUGH_2017_POPULATION = [50, 20, 20, 10]  # in millions
POPULATION_FRACTIONS = ROUGH_2017_POPULATION / np.sum(ROUGH_2017_POPULATION)
initial_infected = .002
pop_0 = np.round(
    np.array([
        [f - initial_infected, 0, initial_infected, 0] for f in POPULATION_FRACTIONS
    ]), 
    decimals=5
)

young_range = st.slider(
	'Period of mixing for younger cohort:',
	0, 100, (5, 100)
)

old_range = st.slider(
	'Period of mixing for older cohort:',
	0, 100, (40, 100)
)

cohort_ranges = [young_range, young_range, old_range, old_range]

betas, epoch_end_times = model.model_input(
    cohort_ranges, 
    seclusion_scale=0.03,
    evolution_length=100
)


res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
df, death_df, y = res.solve_to_dataframe(pop_0.flatten(), detailed_output=True)

young = pd.DataFrame(
    {
        "infection1": y[0][2],
        "infection2": y[1][2]
    }
)

old = pd.DataFrame(
    {
        "infection1": y[2][2],
        "infection2": y[3][2]
    }
)


young["infection"] = young["infection1"] + young["infection2"]
young["severe"] = young["infection"] * 0.2

old["infection"] = old["infection1"] + old["infection2"]
old["severe"] = old["infection"] * 0.8


df1 = pd.DataFrame(
    {
        "days": range(0, 100),
        "pop" : young["infection"] + old["infection"],
        "Group": "Infection"
    }
)

df2 = pd.DataFrame(
    {
        "days": range(0, 100),
        "pop" : young["severe"] + old["severe"],
        "Group": "Severe Infection"
    }
)


df = df1.append(df2)

st.vega_lite_chart(
    data=df,
    spec=_vega_default_spec(
    	color=[],
        scale=[0.0,0.6],
        evolution_length=100
    ),
    use_container_width=True
)

