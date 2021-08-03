import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

from tqdm import tqdm

import gwpopulation
from gwpopulation.conversions import convert_to_beta_parameters
from bilby.core.result import read_in_result

np.seterr(all='ignore')

class FirstGenPop_LVC:
    """
    Generates first-generation population for seeding hierarchical merger trees
    """
    @staticmethod
    def get_samples(samples_path, Nsamps_from_post=1000):
        """
        Get hyperposterior samples and initialize model
        """
        result = read_in_result(samples_path)
        hypersamples = result.posterior.sample(Nsamps_from_post)
        hypersamples = convert_to_beta_parameters(hypersamples)[0]

        # Uses Power Law + Peak model
        model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()

        return FirstGenPop_LVC(hypersamples, model)

    def __init__(self, hypersamples, model):
        """
        Initialized FirstGenPop class
        """
        self.hypersamples = hypersamples
        self.model = model
        self._param_ranges = {"mass_1" : np.linspace(3, 100, 1000), \
            "a_1" : np.linspace(0, 1, 100) , "a_2" : np.linspace(0, 1, 100), \
            "cos_tilt_1" : np.linspace(-1, 1, 100), "cos_tilt_2" : np.linspace(-1, 1, 100)}


    def generate_population(self, Ndraws_per_post=100):
        """
        Wrapper function to generate a first-generation population
        """
        self.Ndraws_per_post = Ndraws_per_post

        m1s = np.array([])
        qs = np.array([])
        a1s = np.array([])
        a2s = np.array([])
        cost1s = np.array([])
        cost2s = np.array([])
        for ii in tqdm(range(len(self.hypersamples))):
            parameters = dict(self.hypersamples.iloc[ii])

            # draw m1 values
            m1_draws = self.draw_m1(parameters)
            m1s = np.concatenate([m1s, m1_draws])
            # draw q values (dependent on m1 draws)
            q_draws = self.draw_q(parameters, m1_draws)
            qs = np.concatenate([qs, q_draws])
            # draw spin magnitudes
            a1_draws, a2_draws = self.draw_spinmags(parameters)
            a1s = np.concatenate([a1s, a1_draws])
            a2s = np.concatenate([a2s, a2_draws])
            # draw spin tilts
            cost1_draws, cost2_draws = self.draw_costilts(parameters)
            cost1s = np.concatenate([cost1s, cost1_draws])
            cost2s = np.concatenate([cost2s, cost2_draws])

        # synthesize secondary masses
        m2s = m1s * qs
        # store population in this instance
        samples = pd.DataFrame(np.atleast_2d([m1s, m2s, qs, a1s, a2s, cost1s, cost1s]).T, \
            columns=['m1','m2','q','a1','a2','cost1','cost2'])
        samples = samples.dropna()
        self.samples = samples


    def draw_m1(self, parameters):
        """
        Draw primary masses
        """
        p_m1 = self.model.p_m1(self._param_ranges, **{key: parameters[key] for key in ["alpha", "mmin", "mmax", "lam", "mpp", "sigpp", "delta_m"]})
        cdf_m1 = cumtrapz(p_m1, initial=0)
        cdf_m1 /= cdf_m1[-1]
        m1_draws = interp1d(cdf_m1, self._param_ranges["mass_1"])(np.random.uniform(0, 1, self.Ndraws_per_post))

        return np.asarray(m1_draws)

    def draw_q(self, parameters, m1_draws):
        """
        Draw mass ratios
        """
        _qs, _m1s = np.meshgrid(np.linspace(0, 1, 110), m1_draws)
        m1q_data = dict(mass_ratio=_qs, mass_1=_m1s)

        cdf_q = cumtrapz(self.model.p_q(m1q_data, parameters["beta"], parameters["mmin"], parameters["delta_m"]), np.linspace(0, 1, 110), axis=-1, initial=0)
        cdf_q /= cdf_q[:, -1:]
        q_draws = [interp1d(_cdf, np.linspace(0, 1, 110))(np.random.uniform(0, 1)) for _cdf in cdf_q]

        return(np.asarray(q_draws))

    def draw_spinmags(self, parameters):
        p_a = gwpopulation.utils.beta_dist(self._param_ranges["a_1"], parameters["alpha_chi"], parameters["beta_chi"], scale = 1.0)
        cdf_pa = cumtrapz(p_a, initial = 0)
        cdf_pa /= cdf_pa[-1]
        cdf_pa_interp = interp1d(cdf_pa, self._param_ranges["a_1"])
        a1_draws = cdf_pa_interp(np.random.uniform(0, 1, self.Ndraws_per_post))
        a2_draws = cdf_pa_interp(np.random.uniform(0, 1, self.Ndraws_per_post))
        return np.asarray(a1_draws), np.asarray(a2_draws)

    def draw_costilts(self, parameters):
        p_cos_t = (1 - parameters["xi_spin"])/2.0 + parameters["xi_spin"] * gwpopulation.utils.truncnorm(self._param_ranges["cos_tilt_1"], 1, parameters["sigma_spin"], 1, -1)
        cdf_cos_t = cumtrapz(p_cos_t, initial = 0)
        cdf_cos_t /= cdf_cos_t[-1]
        cdf_cos_t_interp = interp1d(cdf_cos_t, self._param_ranges["cos_tilt_1"])
        cost1_draws = cdf_cos_t_interp(np.random.uniform(0, 1, self.Ndraws_per_post))
        cost2_draws = cdf_cos_t_interp(np.random.uniform(0, 1, self.Ndraws_per_post))

        return np.asarray(cost1_draws), np.asarray(cost2_draws)

    def sample(self, N):
        """
        Sample N systems from this population
        """
        samples = self.samples.sample(N)
        return samples




class FirstGenPop_fixed:
    """
    Generates first-generation population for seeding hierarchical merger trees using flat distributions
    """
    def __init__(self, Nsamps, Mmin, Mmax, amin, amax):
        """
        Initialized FirstGenPop class
        """
        self.Nsamps = Nsamps
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.amin = amin
        self.amax = amax

    def generate_population(self):
        """
        Wrapper function to generate a first-generation population
        """

        # draw mass values
        m1_draws = self.draw_masses(self.Mmin, self.Mmax, self.Nsamps)
        m2_draws = self.draw_masses(self.Mmin, self.Mmax, self.Nsamps)
        # draw spin magnitudes
        a1s = self.draw_spinmags(self.amin, self.amax, self.Nsamps)
        a2s = self.draw_spinmags(self.amin, self.amax, self.Nsamps)
        # draw spin tilts
        cost1s = self.draw_costilts(self.Nsamps)
        cost2s = self.draw_costilts(self.Nsamps)
        # flip masses and synthesize mass ratios
        m1s, m2s = np.maximum(m1_draws, m2_draws), np.minimum(m1_draws, m2_draws)
        qs = m2s/m1s
        # store population in this instance
        samples = pd.DataFrame(np.atleast_2d([m1s, m2s, qs, a1s, a2s, cost1s, cost1s]).T, \
            columns=['m1','m2','q','a1','a2','cost1','cost2'])
        samples = samples.dropna()
        self.samples = samples


    @staticmethod
    def draw_masses(Mmin, Mmax, Nsamps):
        """
        Draw masses
        """
        m_draws = np.random.uniform(Mmin, Mmax, Nsamps)
        return np.asarray(m_draws)

    @staticmethod
    def draw_spinmags(amin, amax, Nsamps):
        a_draws = np.random.uniform(amin, amax, Nsamps)
        return np.asarray(a_draws)

    @staticmethod
    def draw_costilts(Nsamps):
        cost_draws = np.random.uniform(-1, 1, Nsamps)
        return np.asarray(cost_draws)

    def sample(self, N):
        """
        Sample N systems from this population
        """
        samples = self.samples.sample(N)
        return samples
