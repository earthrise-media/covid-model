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
	0, 180, (52, 180)
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
betas, epoch_end_times = model.model_input(cohort_ranges)

res = model.SEIRModel(betas=betas, epoch_end_times=epoch_end_times)
t, y = res.solve(model.pop_0.flatten())
df = res._to_dataframe(t, y)
death_df = res._estimate_deaths(t, y)

colors = dict(zip(model.COMPARTMENTS, ["#4c78a8", "#f58518", "#e45756", "#72b7b2"]))

def _vega_default_spec(color=None, scale=[0.0, 1.0]):
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
