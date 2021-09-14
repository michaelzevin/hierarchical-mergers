import numpy as np
import pandas as pd
import time

from tqdm import tqdm
import h5py
import multiprocessing
from functools import partial

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

import astropy.units as u
import astropy.constants as C
from astropy.cosmology import Planck18 as cosmo

import precession

from populations import detection_weights

class MergerTree:
    """
    Generates `Ntree` merger trees for `Nhierarch` mergers
    """
    def __init__(self, method, Ntree=100, Nhierarch=10, Ncores=1):
        """
        Initialized MergerTree class
        """
        self._method = method
        self._Ntree = Ntree
        self._Nhierarch = Nhierarch
        self._Ncores = Ncores


    # --- Growing merger trees --- #

    def grow(self, firstgen_pop, isotropic_spins=True):
        """
        Grows the merger tree without assuming a BH budget
        or escape velocities

        Can grow the tree assuming one of two methods:
        `NG1G` merges the merger product with 1G black holes
        `NGNG` merges the merger product with an NG black hole
            of the same generation
        `NGleNG` merges the merger product with a M<=N generation
            black hole
        `EqualPairing` merges the merger product with a black hole
            of the exact same mass

        Create a Pandas dataframe with index indicating each
            merger tree, and rows the properties of each merger
        Structure: ['m1', 'm2', 'a1', 'a2', 'tilt1', 'tilt2',
            'spin_phase', 'M_merge', 'a_merge', 'vkick', 'N_merge']
        Units: Msun, radians, km/s
        """
        self.isotropic_spins = isotropic_spins
        self.firstgen_pop = firstgen_pop

        # Create empty dataframe with pertinent info
        mergers = pd.DataFrame()

        # single processor (with progress bar)
        if self._Ncores==1:
            for branch_idx in tqdm(np.arange(self._Ntree)):
                branch = self.grow_branch(branch_idx)
                mergers = mergers.append(branch)
        else:
            func = partial(self.grow_branch)
            branch_idxs = np.arange(self._Ntree)
            pool = multiprocessing.Pool(self._Ncores)
            branches = pool.imap(func, branch_idxs)
            for branch in list(branches):
                mergers = mergers.append(branch)

        self.mergers = mergers

    def grow_branch(self, branch_idx):
        """
        Function to create a single merger branch of the hierarchical
        merger tree

        This is a time sink, so parallelizing is recommended
        """
        # set random seed, needed for multiprocessing
        np.random.seed(branch_idx)

        df = pd.DataFrame()
        df_cols = ['m1', 'm2', 'q', 'a1', 'a2', 'tilt1', 'tilt2', 'spin_phase', 'M_merge', 'a_merge', 'vkick', 'N_merge', 'N_bh']
        initial_binary = self.firstgen_pop.sample(1)

        m1, m2 = float(initial_binary['m1']), float(initial_binary['m2'])
        a1, a2 = float(initial_binary['a1']), float(initial_binary['a2'])
        # spin tilts
        if self.isotropic_spins==True:
            tilt1, tilt2 = np.arccos(np.random.uniform(-1,1, 1)[0]), \
                np.arccos(np.random.uniform(-1,1, 1)[0])
        else:
            tilt1, tilt2 = np.arccos(float(initial_binary['cost1'])), \
                np.arccos(float(initial_binary['cost1']))
        # phase
        spin_phase = np.random.uniform(0,2*np.pi, 1)[0]

        Mtot = m1+m2
        q = np.min([m1,m2])/np.max([m1,m2])

        # properties of initial merger remnant
        M_merge, a_merge, vkick = self.remnant_properties(Mtot, q, a1, a2, tilt1, tilt2, spin_phase)

        # track the number of mergers and number of BHs used up so far
        Nmerge = 1   # merger number
        Nbh = 2   # initial number of BHs

        df = df.append(pd.DataFrame([[m1, m2, q, a1, a2, tilt1, tilt2, spin_phase, M_merge, a_merge, vkick, Nmerge, Nbh]], columns=df_cols, index=[branch_idx]))

        # loop until we hit Nhierarch
        while Nmerge < self._Nhierarch:
            m1 = M_merge
            a1 = a_merge
            if self._method=='NG1G':
                new_binary = self.firstgen_pop.sample(1)
                m2, a2 = float(new_binary['m1']), float(new_binary['a1'])
            elif self._method=='NGNG':
                m2, a2 = self.get_NG(Nmerge)
            elif self._method=='NGleNG':
                # first, determine the number of mergers Nprime<=N for the merging BH
                Nprime = int(np.random.randint(0,Nmerge+1, size=1))
                # special treatment for Nprime=0 (1G BH)
                if Nprime==0:
                    new_binary = self.firstgen_pop.sample(1)
                    m2, a2 = float(new_binary['m1']), float(new_binary['a1'])
                else:
                    m2, a2 = self.get_NG(Nprime)
            elif self._method=='EqualPairing':
                m2, a2 = m1, a1
            else:
                raise NameError("The method you specified ({:s}) is not defined!".format(self._method))
            # spin tilt and phase
            tilt1, tilt2 = np.arccos(np.random.uniform(-1,1, 1)[0]), \
                np.arccos(np.random.uniform(-1,1, 1)[0])
            spin_phase = np.random.uniform(0,2*np.pi, 1)[0]

            Mtot = m1+m2
            q = np.min([m1,m2])/np.max([m1,m2])
            M_merge, a_merge, vkick = self.remnant_properties(Mtot, q, a1, a2, tilt1, tilt2, spin_phase)

            # update number of mergers that have occurred
            Nmerge += 1

            # get the number of BHs used up so far
            if self._method=='NG1G':
                Nbh = Nmerge+1
            elif self._method=='NGNG':
                Nbh = 2**Nmerge
            elif self._method=='NGleNG':
                Nbh = 2**(Nmerge-1) + 2**(Nprime)
            elif self._method=='EqualPairing':
                Nbh = 2**Nmerge

            df = df.append(pd.DataFrame([[m1, m2, q, a1, a2, tilt1, tilt2, spin_phase, M_merge, a_merge, vkick, Nmerge, Nbh]], columns=df_cols, index=[branch_idx]))

        return df

    def remnant_properties(self, Mtot, q, a1, a2, tilt1, tilt2, spin_phase):
        """
        Calculates mass, spin, and kick of merger product
        """
        M_red, m1_red, m2_red, a1_red, a2_red = precession.get_fixed(q,a1,a2) # Total-mass units M=1
        M_merge = precession.finalmass(tilt1, tilt2, spin_phase, q, a1_red, a2_red) * Mtot
        a_merge = precession.finalspin(tilt1, tilt2, spin_phase, q, a1_red, a2_red)
        vkick = precession.finalkick(tilt1, tilt2, spin_phase, q, a1_red, a2_red, maxkick=False) * C.c.to(u.km/u.s).value
        return M_merge, a_merge, vkick

    def get_NG(self, Nmerge):
        """
        Generates a merger product that has already proceeded through
           N_merge mergers

        NOTE: This assumes that this BH is retain throughout all of its
           prior mergers!
        """
        # We need 2^(Nmerge) BHs to generate the same generational "partner" of a NG BH
        N_initial = int(2**(Nmerge) / 2)   # divide by 2 since we generate both primaries and secondaries

        merger_ctr = 1
        while merger_ctr <= Nmerge:
            # special treatment for first round since we generate both primaries and secondaries
            if merger_ctr==1:
                new_binaries = self.firstgen_pop.sample(N_initial)
                m1s, m2s = np.asarray(new_binaries['m1']), np.asarray(new_binaries['m2'])
                a1s, a2s = np.asarray(new_binaries['a1']), np.asarray(new_binaries['a2'])
                # spin tilts
                if self.isotropic_spins==True:
                    tilt1s, tilt2s = np.arccos(np.random.uniform(-1,1, N_initial)), \
                        np.arccos(np.random.uniform(-1,1, N_initial))
                else:
                    tilt1s, tilt2s = np.arccos(np.asarray(new_binaries['cost1'])), \
                        np.arccos(np.asarray(new_binaries['cost1']))
                # phase
                spin_phases = np.random.uniform(0,2*np.pi, N_initial)
                # get merger properties
                Mtots = m1s+m2s
                qs = np.min([m1s,m2s], axis=0)/np.max([m1s,m2s], axis=0)
                M_merges, a_merges, vkicks = [], [], []
                for Mtot, q, a1, a2, tilt1, tilt2, spin_phase  in zip(Mtots, qs, a1s, a2s, tilt1s, tilt2s, spin_phases):
                    M_merge_tmp, a_merge_tmp, vkick_tmp = self.remnant_properties(Mtot, q, a1, a2, tilt1, tilt2, spin_phase)
                    M_merges.append(M_merge_tmp)
                    a_merges.append(a_merge_tmp)
                    vkicks.append(vkick_tmp)
                M_merges = np.asarray(M_merges)
                a_merges = np.asarray(a_merges)
                vkicks = np.asarray(vkicks)
            else:
                m1s = M_merges[:int(len(M_merges)/2)]
                m2s = M_merges[int(len(M_merges)/2):]
                a1s = a_merges[:int(len(a_merges)/2)]
                a2s = a_merges[int(len(a_merges)/2):]
                tilts = np.arccos(np.random.uniform(-1,1, len(M_merges)))
                tilt1s, tilt2s = tilts[:int(len(tilts)/2)], tilts[int(len(tilts)/2):]
                spin_phases = np.random.uniform(0,2*np.pi, int(len(M_merges)/2))
                # get merger properties
                Mtots = m1s+m2s
                qs = np.min([m1s,m2s], axis=0)/np.max([m1s,m2s], axis=0)
                M_merges, a_merges, vkicks = [], [], []
                for Mtot, q, a1, a2, tilt1, tilt2, spin_phase  in zip(Mtots, qs, a1s, a2s, tilt1s, tilt2s, spin_phases):
                    M_merge_tmp, a_merge_tmp, vkick_tmp = self.remnant_properties(Mtot, q, a1, a2, tilt1, tilt2, spin_phase)
                    M_merges.append(M_merge_tmp)
                    a_merges.append(a_merge_tmp)
                    vkicks.append(vkick_tmp)
                M_merges = np.asarray(M_merges)
                a_merges = np.asarray(a_merges)
                vkicks = np.asarray(vkicks)

            merger_ctr += 1

        # return properties of NG merger (should only be one left in these arrays!)
        assert len(M_merges)==1 and len(a_merges)==1 and len(vkicks)==1, "You screwed something up Zevin!"
        return float(M_merges[0]), float(a_merges[0])



    # --- Processing trees --- #

    @staticmethod
    def draw_redshifts(N, z_max, mdl='2017'):
        """
        Draws `N` redshifts according to Madau & Fragos 2017
        """
        def sfr(z, mdl=mdl):
            """
            Star formation rate as a function in redshift, in units of M_sun / Mpc^3 / yr
            mdl='2017': Default, from Madau & Fragos 2017. Tassos added more X-ray binaries at higher Z, brings rates down
            mdl='2014': From Madau & Dickenson 2014, used in Belczynski et al. 2016
                """
            if mdl=='2017':
                return 0.01*(1+z)**(2.6)  / (1+((1+z)/3.2)**6.2)
            if mdl=='2014':
                return 0.015*(1+z)**(2.7) / (1+((1+z)/2.9)**5.6)

        z_grid = np.linspace(0,z_max,5000)
        sfr_grid = sfr(z_grid)
        sfr_cdf = cumulative_trapezoid(sfr_grid, z_grid, initial=0)
        sfr_cdf /= sfr_cdf.max()
        sfr_icdf_interp = interp1d(sfr_cdf, z_grid, fill_value="extrapolate")

        rnd_variates = np.random.uniform(0,1, N)
        return sfr_icdf_interp(rnd_variates)


    def assign_redshifts(self, tdelay_min=10, tdelay_max=100, z_max=1, tlb_min=10):
        """
        Assign redshifts to the mergers in merger tree

        As of now, this follows Madau & Fragos 2017 for the first merger, and assigns redshifts
        flat-in-log on the interval [tdelay_min, tdelay_max for subsequent mergers (in Myr)
        """
        # first, get the redshifts for when the mergers started
        z_birth = self.draw_redshifts(self._Ntree, z_max=z_max)

        # convert to physical time
        tlb_birth = cosmo.lookback_time(z_birth)

        # get delay times for subsequent mergers [Ntree x Nhierarch-1]
        delay_times = 10**(np.random.uniform(np.log10(tdelay_min), np.log10(tdelay_max), \
                    size=(self._Ntree, self._Nhierarch-1)))

        # generate redshift interpolant for converting back to redshift
        z_grid = np.logspace(-3, np.log10(z_max), 1000)
        tlb_grid = cosmo.lookback_time(z_grid)
        tlb_interp = interp1d(tlb_grid.to(u.Myr).value, z_grid, fill_value="extrapolate")

        # for each tree, seed the initial redshift and apply delay times
        for idx, (tlb, tdelays) in enumerate(zip(tlb_birth, delay_times)):
            merger_tlbs = list(-1*tdelays)
            merger_tlbs.insert(0, tlb.to(u.Myr).value)
            merger_tlbs = np.cumsum(merger_tlbs)
            # apply minimum to lookback time
            merger_tlbs = np.where(merger_tlbs < tlb_min, tlb_min, merger_tlbs)
            # get redshift
            merger_redshifts = tlb_interp(merger_tlbs)
            self.mergers.loc[idx, 't_lookback'] = merger_tlbs
            self.mergers.loc[idx, 'z'] = merger_redshifts


    def apply_selection_effects(self, sensitivity, pdet_grid):
        """
        Applies selection effects to the population based on thier masses and redshifts

        Uses semi-analytic VT grid for determining detection probabilities and relative weights
        """
        VT_grid = pd.read_hdf(pdet_grid, key=sensitivity)
        self.mergers['pdets'], self.mergers['weights'] = detection_weights.selection_function(self.mergers, VT_grid)


    def prune_by_vesc(self, vesc):
        """
        Prunes trees based on escape velocity of host environment `vesc`
        """
        series_name = 'vesc_'+str(int(vesc))
        # loop over merger trees
        for midx in list(self.mergers.index.unique()):
            # track when the system is ejected, 1 means still in cluster and 0 means it was ejected by this point
            is_ejected = np.ones(len(self.mergers.loc[midx]))
            # find the first time the kick was higher than the escape velocity
            mergers_that_would_eject = np.argwhere(np.asarray(self.mergers.loc[midx]['vkick']) >= vesc)
            # make sure that at least a single encounter was ejected
            if len(mergers_that_would_eject) > 0:
                point_of_ejection = mergers_that_would_eject.min()
                is_ejected[point_of_ejection:] = 0

            # assign 0 if merger product is retained and 1 if the merger product was ejected
            self.mergers.loc[midx, series_name] = is_ejected
        self.mergers = self.mergers.astype({series_name: 'int64'})

    def prune_by_BHbudget(self, BH_budget):
        """
        Prunes trees based on a prescribed black hole budget
        """
        series_name = 'Nbh_'+str(int(BH_budget))
        # loop over merger trees
        for midx in list(self.mergers.index.unique()):
            # look for where we run out of BHs (1 means we're good, 0 means we've run out!)
            no_more_BHs = np.ones(len(self.mergers.loc[midx]))
            # find the first time we've exceeded the BH budget
            too_many_BHs = np.argwhere(np.asarray(self.mergers.loc[midx]['N_bh']) >= BH_budget)
            # set these mergers to 0
            if len(too_many_BHs) > 0:
                no_more_BHs[too_many_BHs.min():] = 0

            # assign 0 if merger product is retained and 1 if the merger product was ejected
            self.mergers.loc[midx, series_name] = no_more_BHs
        self.mergers = self.mergers.astype({series_name: 'int64'})


    # --- Writing to disk --- #

    def write(self, output_path):
        """
        Saves info from merger tree class to disk
        """
        hfile = h5py.File(output_path, "w")
        bsgrp = hfile.create_group("merger_tree")
        # add attributes
        bsgrp.attrs["method"] = self._method
        bsgrp.attrs["Ntree"] = self._Ntree
        bsgrp.attrs["Nhierarch"] = self._Nhierarch
        bsgrp.attrs["isotropic_spins"] = 'yes' if self.isotropic_spins==True else 'no'
        hfile.close()

        self.mergers.to_hdf(output_path, key='merger_tree/mergers')

