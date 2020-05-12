"""Code to build and solve an age-structured SEIR epidemiological model.

Some model assumptions:
 - Four compartment (S-E-I-R) model, stratified by age cohorts (or any other 
division of the population such that individuals do not switch cohorts). 
Age is a discrete variable. See app.py or the deployed app for analytic 
expression of the relevant differential equations. 
 - No aging: The time horizon for the model is less than one year.
 - No vital statistics: No birth or death, aside from possible deaths due to 
disease.
 - The Exposed compartment are not infectious.

"""

import numpy as np
import pandas as pd
import scipy.integrate

COMPARTMENTS = ['Susceptible', 'Exposed', 'Infected', 'Died or recovered']

AGE_COHORTS = ['0-19', '20-59', '60+']

INFECTION_FATALITY = [.0001, .0032, .0328]

# population fractions by region; UN 2020 data
WORLD_POP = {
    'Africa': [.507, .438, .055],
    'Americas': [.293, .541, .166],
    'Asia': [.312, .558, .131],
    'Europe': [.211, .532, .257]
}
WORLD_POP = {region: np.array(f) for region, f in WORLD_POP.items()}

def _symmetrize(pop_fracs, contact_data):
    """Construct a contact matrix with reciprocity from empirical data."""
    f = pop_fracs.copy()
    d = contact_data.copy()
    c = np.zeros((len(f), len(f)))

    for i in range(len(f)):
        for j in range(len(f)):
            c[i,j] = (d[i,j]*f[i] + d[j,i]*f[j])/(2*f[i])
    return c

# UK data from the POLYMOD survey, basis for contact matrices
CONTACT_DATA = np.array([
    [7.86, 5.22, 0.5], [2.37, 7.69, 1.06], [1.19, 5.38, 1.92]])

CONTACT_MATRICES_0 = {
    region: _symmetrize(f, CONTACT_DATA) for region, f in WORLD_POP.items()}

# The effects of various non-pharmaceutical interventions.
# Chi denotes an overall multiplicative factor on the basic contact matrix.
# For cohort-based interventions, xi denotes a factor to be applied to
# contact matrix entries given by the corresponding indices.
NPI_IMPACTS = {
    'Cancel mass gatherings': {'chi': 0.72},
    'Quarantine': {'chi': 0.63},
    'Quarantine and tracing': {'chi': 0.48},
    'School closure': {
        'xi': 0, 'indices': [(0, 0)]
    },
    'Shelter in place': {'chi': 0.34},
    'Shielding the elderly': {
        'xi': 0.5, 'indices': [(0, -1), (1, -1), (-1, 0), (-1, 1),  (-1, -1)]
    }
}


class SEIRModel(object):
    """Class to solve an age-structured SEIR Compartmental model.

    Attributes: 
        contacts: List of effective contact matrices, one per epoch. (An epoch
            here denotes a period of set interventions.) 
        epoch_end_times: List of days at which transmission rates change.
        alpha: Inverse incubation period
        beta: Probability of transmission given a contact
        gamma: Inverse duration of infection
        N_cohorts: Number of cohorts 
        compartments: List of names of compartments
        N_compartments: Number of compartments (hard-coded to 4: S-E-I-R)
        s, e, i, r: Indices of the compartments in the state vector y(t). 

    External methods: 
        f: Function giving the rate of change in the state variable y(t).
        solve: Integrate the coupled differential equations.
        solve_to_dataframe: Solve and output a tidy dataframe.
    """
    def __init__(self, contact_matrices, epoch_end_times,
                     incubation_period=5.1, prob_of_transmission=.034,
                     duration_of_infection=6.3, N_cohorts=len(AGE_COHORTS)):
        
        if len(epoch_end_times) != len(contact_matrices):
            raise ValueError('Each contact matrix requires an epoch end time.')
        self.contacts = contact_matrices
        self.epoch_end_times = epoch_end_times
        self.alpha = 1/incubation_period
        self.beta = prob_of_transmission
        self.gamma = 1/duration_of_infection
        self.N_cohorts = N_cohorts
		
        # N.B. hard-coded values. The function f() assumes these four
        # compartments.
        self.compartments = COMPARTMENTS
        self.N_compartments = 4
        self.s, self.e, self.i, self.r = list(range(4))                  
													   
    def _fetch_contact(self, t):
        """Fetch the contact matrix for given time t."""
        for end_time, contact in zip(self.epoch_end_times, self.contacts): 
            if t <= end_time: 
                return contact      
		
    def f(self, t, y):
        """Function giving the rate of change in the state variable y(t)."""
        y = y.reshape(self.N_cohorts, self.N_compartments)
        dy = np.zeros((self.N_cohorts, self.N_compartments))
        contact = self._fetch_contact(t)
        for a in range(self.N_cohorts):
            infection_rate = np.sum([
                self.beta * contact[a,b] * y[b,self.i] / np.sum(y[b,:])
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

    def solve_to_dataframe(self, y0, detailed_output=False):
        """Solve and output a tidy dataframe."""
        t, y = self.solve(y0)
        y = y.reshape(self.N_cohorts, self.N_compartments, len(t))

        # calculate the time series for the total population
        aggregate_nums = np.sum(y, axis=0)
        df = pd.DataFrame(dict({'days': t},
                          **dict(zip(self.compartments, aggregate_nums))))
        df = pd.melt(df, id_vars=['days'], var_name='Group', value_name='pop')
        df['Date(s) of intervention'] = str(self.epoch_end_times[:-1])

        if detailed_output == True:
            return df, y
        else:
            return df

def model_input(contact_matrix, day_ranges, selected_npis, total_days,
                npi_impacts=NPI_IMPACTS):
    """
    Function to enumerate conact matrices and their epochs.

    Arguments:
        contact_matrix: Basic contact matrix, without interventions
        day_ranges: List of day range tuples denoting the period
            during which interventions are applied
        selected_npis: List of interventions, one for each date range
        total_days: Total number of days to run model
        npi_impacts: Dict of interventions of form {name: impact}, 
            cf. NPI_IMPACTS above.
        
    Returns: A list of effective contact matrices and a list of epoch end times
    """
    def _intersects(day_range, epoch_tuple):
        cs = set(range(*day_range))
        es = set(range(*epoch_tuple))
        if len(cs.intersection(es)) > 1:
            return True
        else:
            return False

    def _apply(npi, npi_impacts, contact_matrix):
        impact = npi_impacts.get(npi, {})
        contact_matrix *= impact.get('chi', 1)
        for idx_pair in impact.get('indices', []):
            contact_matrix[idx_pair] *= impact.get('xi', 1)
        return contact_matrix

    epoch_tuples = _partition(day_ranges, total_days)
    contact_matrices = []

    for epoch in epoch_tuples:
        c_eff = contact_matrix.copy()
        for day_range, npi in zip(day_ranges, selected_npis):
            if _intersects(day_range, epoch):
                c_eff = _apply(npi, npi_impacts, c_eff)
        contact_matrices.append(c_eff)

    epoch_ends = [e[1] for e in epoch_tuples]
    return [contact_matrices, epoch_ends]

def _partition(day_ranges, total_days):
    """Partition day_ranges into unique, non-overlapping epochs."""
    flat_dates = [v for sublist in day_ranges for v in sublist]
    flat_dates = set(flat_dates + [0, total_days])
    epoch_delims = sorted(flat_dates)

    def _tuplize(lst):
        # create a sequence of tuples from a list
        # e.g., tuplize([0, 1, 2, 3]) -> [[0, 1], [1, 2], [2, 3]]
        for i in range(0, len(lst)):
            val = lst[i:i+2]
            if len(val) == 2:
                yield val
                
    return list(_tuplize(epoch_delims))
