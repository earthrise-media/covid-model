import altair as alt
import streamlit as st
import numpy as np
import pandas as pd
import shapely.geometry

import model

TOTAL_POPULATION = 1e6 # in millions

# Sidebar and the subsequent parameter definitions
st.sidebar.markdown(
	"""
	**Initial conditions**
	"""
)

region_option = st.sidebar.selectbox(
	'World Region', 
	list(model.WORLD_POP.keys())
)

initial_infected = st.sidebar.number_input(
	label='Initial rate of infection (percent or fraction of a percent)',
    value=0.1,
	min_value=0.01,
	max_value=100.,
	step=0.01
)
initial_infected *= .01

population = TOTAL_POPULATION * model.WORLD_POP[region_option]
pop_0 = np.array([[f * (1 - 2 * initial_infected), f * initial_infected,
					   f * initial_infected, 0, 0, 0] for f in population])

start_date = st.sidebar.number_input(
	label='Start day (for visualization purposes only)',
	value=int(),
	min_value=int(0),
	max_value=int(60),
	step=int()
)


# Introductory text
st.title('Illustrating stacked NPIs in the browser')

st.markdown(""" 

This model should be used as a heuristic to illustrate the effect of
lifting non-pharmaceutical interventions (NPIs) for certain
sub-populations. The numbers are not projections, but rather
indications of the directional effects of stacked policies.  The
authors of this model and visualization are _not_ epidemiologists.  At
best, we are armchair epidemiologists &mdash; which is pretty bad.

We extend a standard SEIR epidemiological model to incorporate the
interactions between different population cohorts.  The cohorts are a
partition of the population &mdash; a division into complete and
non-overlapping groups. We use age-defined cohorts in this example,
but other partitions could be used to explore differences in disease
propagation and corresponding interventions in urban/rural or
service/manufacturing/student groups. Additionally, those people
severely or *mortally* infected (M), who will end up dying (D), are
tracked in separate comparments.

Our multidimensional generalization of the SEIR compartmental model
can be represented mathematically as follows:

""")

eqnarray = r"""
	\begin{array}{lll}
		\frac{dS_a}{dt} &=& - S_a\; \sum_b \beta c_{ab}(t) (I_b + M_b)/ N_b \\
		\\
		\frac{dE_a}{dt} &=& S_a\; \sum_b \beta c_{ab}(t) (I_b + M_b)/ N_b - \alpha E_a\\
		\\
		\frac{dI_a}{dt} &=& \alpha (1-\kappa_a) E_a - \gamma I_a\\
		\\
        \frac{dM_a}{dt} &=& \alpha \kappa_a E_a - \delta M_a\\
        \\
		\frac{dR_a}{dt} &=& \gamma I_a \\
        \\
        \frac{dD_a}{dt} &=& \delta M_a\\

	\end{array}{}
"""

st.latex(eqnarray)

st.write("""

The subscripts (a,b) index the (age) cohorts, while alpha, beta,
gamma, and delta are the inverse incubation period, the probability of
transmission given a contact between two people, the inverse duration
of infection, and the inverse time to death, respectively. The matrix
c_ab is the *contact matrix*, encoding the average number of daily
contacts a person in cohort a has with people in cohort b. The vector
kappa_a encodes the infection fatality rates for each cohort.

""")  

st.subheader('Initial conditions')

st.write("""

The number of people in each age cohort is an initial
condition for the model, as defined by **the selections in the
sidebar**.  The proportions given here for **{}** reflect
world-regional demographic data from the United Nations for for a
hypothetical subset of {} million people:

""".format(region_option, int(TOTAL_POPULATION/1e6)))

df = pd.DataFrame(
	np.array(population).reshape(1,3),
	columns=model.AGE_COHORTS,
    index=["People in cohort"]
)
df["Total"] = df.sum(axis=1)

st.dataframe(df.style.highlight_max(axis=1, color="#A9BEBE").format('{:.0f}'))

st.write("""

The initial rate of infection is drawn from the sidebar at left. We assume
also that a like number of people have been exposed. The initial 
distribution of people among the various disease compartments is then:

""")

display_pop_0 = pd.DataFrame(
	pop_0,
	index=model.AGE_COHORTS,
	columns=model.COMPARTMENTS
)

st.write(display_pop_0)

st.subheader('An illustration of *Flattening the Curve*')

st.write("""

In the simplest version of the model, an intervention is applied evenly 
across age cohorts.  As in a standard SEIR model, a strong intervention 
early in the pandemic can be seen to 'flatten the curve.' The pandemic 
resurges when the intervention is lifted.

""")

START_DAY, END_DAY = 0, 300

mixing_range = st.slider(
	'Period of shelter-in-place for whole population',
	START_DAY, END_DAY, (30, 160)
)

contact_matrices, epoch_end_times = model.model_input(
    model.CONTACT_MATRICES_0[region_option],
	[mixing_range],
    ['Shelter in place'],
    END_DAY-START_DAY)

res = model.SEIRModel(contact_matrices, epoch_end_times)
df = res.solve_to_dataframe(pop_0.flatten())

infected = df[(df["Group"] == "Infected")]

chart = alt.Chart(infected[infected["days"] > start_date]).mark_line(
	color="#e45756").encode(
		x=alt.X('days', axis=alt.Axis(title='Days')),
		y=alt.Y('pop', axis=alt.Axis(title='Number infected'),
                scale=alt.Scale(domain=(0,TOTAL_POPULATION/10))))

st.altair_chart(chart, use_container_width=True)

st.subheader('Stacking NPIs: An illustration.')

st.write("""

It is challenging to distinguish the effect of individual public
health and social measures (PHSMs) on the rate of COVID-19
transmission within a population.  Effectiveness depends on how fully
communities adopt and adhere to PHSMs, additional interventions they
are combined with, and other variables like family size and level of
intergenerational contact within a community. Still, evidence does
show that PHSMs are more effective when implemented in combination, or
“stacked,” than when implemented individually. The following scenario
illustrates this concept, when the PHSMs are tied to well-defined
partitions of the general population.

The input model parameters come with a high degree of uncertainty, and
model outputs are unstable with respect to variations in these
parameters.  Outputs should therefore be understood to *illustrate* the pattern
of the infection &mdash; its surge and resurgence &mdash; rather than
to provide reliable quantitative projections. A number of expected
deaths is tallied at the end to demonstrate the impact of prolonged,
high infection rates rather than momentary spikes.  It is, in effect,
a way to understand the area under the curve without additional lines
on the chart.

""")

npi_intervals = {
    'School closure':
        st.slider('Schools closed', START_DAY, END_DAY, (30, 70)),
    'Cancel mass gatherings':
        st.slider('Cancellation of mass gatherings',
                  START_DAY, END_DAY, (30, 80)),
    'Shielding the elderly':
        st.slider('Shielding the elderly',
                  START_DAY, END_DAY, (30, 100)),
    'Quarantine and tracing':
        st.slider('Self-isolation, quarantine, and contact tracing',
                  START_DAY, END_DAY, (70, 200))
}

st.write("""Additionally, we can imagine a complete shutdown, where
the entire population is required to **shelter in place**. This
intervention supersedes those above. For the duration of the shelter in place 
order the above interventions have no additional effect.""")

shelter_interval = st.slider('Shelter in place', START_DAY, END_DAY, (20, 20))

def _trim(interval, interval_to_excise):
    l1 = shapely.geometry.LineString([[x,0] for x in interval])
    l2 = shapely.geometry.LineString([[x,0] for x in interval_to_excise])
    diff = l1.difference(l2)
    if type(diff) == shapely.geometry.linestring.LineString:
        coords = [int(x) for x,_ in diff.coords]
        coords = [coords] if coords else []
    elif type(diff) == shapely.geometry.multilinestring.MultiLineString:
        coords = [[int(x) for x,_ in segment.coords] for segment in diff.geoms]
    return coords

selected_npis, intervals = [], []
for k,v in npi_intervals.items():
    coords = _trim(v, shelter_interval)
    for c in coords:
        selected_npis.append(k)
        intervals.append(c)
            
selected_npis.append('Shelter in place')
intervals.append(shelter_interval)

contact_matrices, epoch_end_times = model.model_input(
    model.CONTACT_MATRICES_0[region_option],
    intervals,
	selected_npis,
    END_DAY-START_DAY)

res = model.SEIRModel(contact_matrices, epoch_end_times)

df, y = res.solve_to_dataframe(pop_0.flatten(), detailed_output=True)
infected = df[df["Group"] == "Infected"]

chart = alt.Chart(infected[infected["days"] > start_date]).mark_line(
	color="#e45756").encode(
		x=alt.X('days', axis=alt.Axis(title='Days')),
		y=alt.Y('pop', axis=alt.Axis(title='Number infected'),
                scale=alt.Scale(domain=(0,TOTAL_POPULATION/10))))

st.markdown('Infections in **%s** (per million population)' % (region_option))

st.altair_chart(chart, use_container_width=True)

deaths = df[df["Group"] == "Dead"]

chart = alt.Chart(deaths[deaths["days"] > start_date]).mark_line(
    color="#0080FF").encode(
		x=alt.X('days', axis=alt.Axis(title='Days')),
		y=alt.Y('pop', axis=alt.Axis(title='Number of deaths'),
                scale=alt.Scale(domain=(0,TOTAL_POPULATION/1e2))))

st.markdown('Deaths in **%s** (per million population)' % (region_option))

st.altair_chart(chart, use_container_width=True)


st.write("""Expected numbers of fatalities in the model vary
significantly with world region, because mortality rates are higher
for aging populations.""")

final_deaths = [x[res.d][-1] for x in y]
death_df = pd.DataFrame([final_deaths],
                            columns = model.AGE_COHORTS,
                            index=["Deaths"])
death_df["Total"] = death_df.sum(axis=1)

display_df = np.round(death_df).astype('int32')

st.write(display_df.style.highlight_max(axis=1, color="#e39696"))

st.markdown(""" 

> **Note**: These tallies are not to be trusted as projections of
deaths under real-world conditions. Please be advised.  The objective
is to visualize the patterns of infection and recovery &mdash; an
interactive visualization of the differential equations that underlie
the prevailing models.

""")
