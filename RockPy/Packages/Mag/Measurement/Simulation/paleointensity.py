# coding: utf-8
import pandas as pd
import numpy as np
import scipy as sp
import RockPy

import os
import matplotlib.pylab as plt
import RockPy.Packages.Mag.Measurement.Simulation.utils as SimUtils

# from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
#     Plot, Figure, Matrix, Package
# from pylatex.utils import italic
from copy import deepcopy

import logging


class Fabian2001(object):
    presets = pd.read_csv(os.path.join(RockPy.installation_directory,
                                       'Packages', 'Mag', 'Measurement', 'Simulation',
                                       'paleointensity_presets.csv'), index_col=0)

    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.MTYPE
        return logging.getLogger('RockPy.%s' % cls.__name__)

    @staticmethod
    def get_steps(steps, tmax=680., ck_every=2, tr_every=2, ac_every=2):
        """
        Wrapper function that creates a pandas DataFrame and a list of measurement steps.

        If filepath is provided a latex document is created

        Magic Lab-treatment codes :
            'TH': 'LT-T-Z'
            'PT': 'LT-T-I'
            'CK': 'LT-PTRM-I'
            'TR': 'LT-PTRM-MD'
            'AC': 'LT-PTRM-Z'
            'NRM': 'LT-NO'
            
        Notes
        -----
            calls RockPy.Packages.Mag.Measurement.Simulation.utils.ThellierStepMaker
        """
        out = SimUtils.ThellierStepMaker(steps=steps, tmax=tmax, ck_every=ck_every, tr_every=tr_every,
                                         ac_every=ac_every)
        return out

    def __init__(self, preset=None,
                 a11=None, a12=None, a13=None, a1t=None,
                 a21=None, a22=None, a23=None, a2t=None,
                 b1=None, b2=None, b3=None, bt=None,
                 d1=0, d2=0, d3=0, dt=0,
                 grid=100, hpal=1, hlab=1, ms=1,
                 tc=560, temp_steps=11,
                 tmax=560, ck_every=2, tr_every=2, ac_every=2,
                 ):
        # todo alpha
        # todo high temperature tails

        """
        Standard parameters as used by Leonhard 2004, Fig.2c, a1t,a2t,a13,a23 are not specified in the paper

        Parameters
        ----------
            Distibution for tau_b < tau_ub
            preset: str
                A collection of models from the Fabian2001 and Leonhard2004 paper

            a11: float
                constant part of the distribution
            a12: float
                if 0: unblocking temperature distribution has same width for all tau_b
            a1t: float
                amplitude of distribution
            a13: float
                width

            Distibution for tau_b >= tau_ub
            a21: float
                constant part of the distribution
            a22: float
                if 0: unblocking temperature distribution has same width for all tau_b
            a2t: float
                amplitude of distribution
            a23: float
                width of distribution

            b1: float
            b2: float
            bt: float
            b3: float
                Characterizes the width of the remanence acquisition spectrum, small values indicate a sharp blocking

            tc: float
                Curie temperature of the simulation
                
            # Pressure demagnetization parameters
            d1: float
                linear part of the pressure demagnetization
            d2: float
                ammount of non-linear demagnetization
            d3: float
                width of the demagnetization distribution
            dt: float
                center of the distribution
            
            grid: int
                default_recipe:100
                the number of blocking and unblocking temperatures to be calculated
            hpal: float
                default_recipe: 1
                reduced paleofield
            hlab: float
                default_recipe: 1
                reduced labfield
            ms: float
                saturation magnetization of the simulated sample
            
            # measurement simulation parameters e.g. how many points
            temp_steps: int, arraylike, pd.DataFrame
                int will create an evenly spaced number of measurement temperatures up to tmax
                a list of temperatuires will create these temperatures
                - if a DataFrame is passed (mostly for fitting) no new DF will be generated
            tmax: float
                maximum temperature of the experiment
            ck_every: int
                after how many thermal demag steps a pTRM check is done
            tr_every: int
                after how many thermal demag steps a tail check is done
            ac_every: int                
                after how many thermal demag steps a additivity check is done

        """

        passed_params = {'a11': a11, 'a12': a12, 'a13': a13, 'a1t': a1t,
                          'a21': a21, 'a22': a22, 'a23': a23, 'a2t': a2t,
                          'b1': b1, 'b2': b2, 'b3': b3, 'bt': bt,
                          'd1': d1, 'd2': d2, 'd3': d3, 'dt': dt,
                          'tc': tc, 'grid': grid,
                          'hpal': hpal, 'hlab': hlab,
                          'ms': ms,
                          'temp_steps': temp_steps,
                          }

        if not isinstance(temp_steps, pd.DataFrame):
            self.steps = self.get_steps(temp_steps, tmax=tmax, ck_every=ck_every, tr_every=tr_every, ac_every=ac_every)
        else:
            self.steps = temp_steps

        if preset is None:
            preset = 'Fabian4a'

        if preset is not None and preset not in self.presets:
            print('preset << %s >> not implemented, chose from:' % preset)
            print(', '.join([i for i in self.presets]))
            print('using: Fabian Fig. 4a')
            preset = 'Fabian4a'

        self.simparams = deepcopy(passed_params)

        # update the instance dictionary  with the preset
        self.simparams.update(self.presets.to_dict()[preset])

        # update the instance dictionary  with the parameters, specified
        self.simparams.update({k: v for k, v in passed_params.items() if v is not None or k in ('d3', 'dt', 'R')})

        # create reduced blocking and unblocking temperatures
        self.tau_b = np.arange(0, grid + 1, 1) / grid
        self.tau_ub = self.tau_b

        self.initialize_distributions()



    @property
    def nrm(self):
        return self.moment(1, self.simparams['hpal'], pressure_demag=False)


    def initialize_distributions(self):
        """
        calculates the distributions of lambda1, lambda2, beta, gamma and chi
        Returns
        -------

        """

        self.log().debug('Calculating simulation using these parameters:')
        self.log().debug('='*50)
        for k, v in self.simparams.items():
            self.log().debug('\t%s: %s'%(k,v))

        # calculate lambda functions for each blocking temperature
        self.l1 = self.lambda1(self.tau_b)
        self.l2 = self.lambda2(self.tau_b)

        # calculate lambda functions for each blocking temperature
        self.betas = self.beta(self.tau_b)

        # calculate gamma function for each blocking temperature
        self.gammas = [self.gamma(tau_b) for tau_b in self.tau_b]

        # initiate dataframe for characteristic function
        self.chi = self.get_chi_grid()

        # calculate demagnetization function
        self.demag_dist = self.simparams['d1'] + self.simparams['d2'] * \
                          self.cauchy(self.tau_ub - self.simparams['dt'], self.simparams['d3'])
        self.zf_steps = 1 - self.demag_dist

    def change_simparams(self, **new_params):
        """
        Method to change the simulation parameters for the simulation.
        Recalculates the differnet distributions from new parameters.

        Parameters
        ----------
        new_params: dict
            dict with new parameters

        Returns
        -------

        """

        for k,v in new_params.items():
            self.simparams[k] = v

        self.initialize_distributions()

    @classmethod
    def cauchy(cls, x, s):
        if s == 0:
            s = 1e-16
        return 1 / (1 + (x / s) ** 2)

    def tau(self, t):
        return (t - 20) / (self.simparams['tc'] - 20)

    def beta(self, tau):
        """
        Calculates the beta function for the distribution
        Parameters
        ----------
        tau

        Returns
        -------
            np.array
        """
        return self.simparams['b1'] + self.simparams['b2'] * \
               self.cauchy(tau - self.simparams['bt'], self.simparams['b3'])

    def lambda1(self, tau, call=''):
        '''
        controls the width of the width of the distribution chi(tb, ) for values of tub > tb
        '''

        if self.simparams['a12'] > 0:
            return self.simparams['a11'] + self.simparams['a12'] * \
                   self.cauchy(tau - self.simparams['a1t'], self.simparams['a13'])
        else:
            return np.ones(tau.shape) * self.simparams['a11']

    def lambda2(self, tau, call=''):
        '''
        controls the width of the width of the distribution chi(tb, ) for values of tub < tb
        '''
        if self.simparams['a22'] > 0:
            return self.simparams['a21'] + self.simparams['a22'] * \
                   self.cauchy(tau - self.simparams['a2t'], self.simparams['a23'])
        else:
            return np.ones(tau.shape) * self.simparams['a21']

    def gamma(self, tau_b):
        """
        normalizes the integral of chi(tau_b, )
        """

        # get the index of tau_b
        idx = np.where(self.tau_ub == tau_b)[0][0]

        beta = self.betas[idx]

        lambda1 = self.l1[idx]
        lambda2 = self.l2[idx]

        int1 = np.sum(self.cauchy(self.tau_b[idx:], lambda1))
        int2 = np.sum(self.cauchy(self.tau_b[:idx + 1], lambda2))

        return beta * 1 / (int1 + int2)  # as described by Fabian2001

    def get_chi(self, tau_b):
        """
        Calculates the unblocking distribution for a given blocking temperature
        Returns
        -------
            ndarray
        """
        # get index of tb in array (tb == tub)
        # indices < idx -> tau_b < tau_ub
        # indices >= idx -> tau_b >= tau_ub
        idx = np.where(self.tau_ub == tau_b)[0][0]

        gamma = self.gammas[idx]

        # distribution for tb<tub
        c2 = self.cauchy(self.tau_ub[tau_b < self.tau_ub] - tau_b, self.l1[idx])
        # distribution for tb>tub
        c1 = self.cauchy(self.tau_ub[tau_b >= self.tau_ub] - tau_b, self.l2[idx])

        return gamma * np.concatenate([c1, c2])

    def FieldMatrix(self, tau_i, hlab, pressure_demag=False):

        data = np.ones((self.tau_ub.size, self.tau_b.size)) * 10

        # the index is where ti == tau_b and tau_ub
        if tau_i == 0:
            idx = 0
        else:
            idx = np.argmin(np.abs(self.tau_b - tau_i)) + 1

        # self.log().debug('Tau_i = %.2f, idx = %i'%(tau_i, idx))
        data[:idx, :idx] = hlab
        # self.log().debug('hlab rectangle shape: (%s, %s)'%data[:idx, :idx].shape)
        data[:idx, idx:] = 0
        # self.log().debug('demag rectangle shape: (%s, %s)'%(data[:idx, idx:].shape))
        data[idx:, :] = self.simparams['hpal']
        # self.log().debug('paleomag rectangle shape: (%s, %s)'%data[idx:, :].shape)

        if pressure_demag:
            pdem = self.zf_steps[idx:].reshape(self.zf_steps[idx:].shape[0], 1)
            data[idx:, :] = data[idx:, :] * pdem

        return data

    def H(self, tau_i, tau_b, tau_ub, hlab, pressure_demag=False):
        """
        Calculates the potential field for a given tau_i and tau_b
        Parameters
        ----------
        tau_i
        tau_b
        tau_ub
        hlab

        Returns
        -------
        """

        if tau_i >= tau_ub and tau_i >= tau_b and tau_i > 0:
            return hlab

        elif tau_ub > tau_i or tau_i == 0:
            if pressure_demag:
                idx = np.where(self.tau_ub == tau_b)[0][0]
                return self.zf_steps[idx] * self.simparams['hpal']
            else:
                return self.simparams['hpal']

        else:
            return 0

    def get_chi_grid(self):
        """
        Method that calculates a matrix of chi values.
        For each tb there is a distribution of tub, get_chi_grid calculates them for a given tb
        
        Returns
        -------
            numpy ndarray with chi values (columns = blocking temperatures, indices = unblocking temperatures)
        """

        # initialize array
        data = np.zeros((len(self.tau_ub), len(self.tau_b)))

        # cycle over tb values
        for col, tb in enumerate(self.tau_b):
            # calculate chi for given tb
            data[:, col] = self.get_chi(tb)

        return data

    def moment(self, tau_i, applied_field=1, pressure_demag=False):
        """

        Parameters
        ----------
        tau_i: float
        applied_field : float
        pressure_demag: bool
            default_recipe: False
            if True a cauchy distributed demagnetization of unblocking temperatures is calculated
        """
        h = self.FieldMatrix(tau_i=tau_i, hlab=applied_field, pressure_demag=pressure_demag)
        return (h * self.chi).sum().sum()

    def get_zerofield_data(self, steps=None, pressure_demag=False, norm=False):
        """
        function calls Fabian2001.get_data and filters for TH ('LT-T-Z') steps and NRM ('LT-NO').
        Parameters
        ----------
        steps
        pressure_demag
        norm

        Returns
        -------

        """
        data = self.get_data(steps=steps, pressure_demag=pressure_demag, norm=norm)
        data = data[np.in1d(data['LT_code'], ['LT-T-Z', 'LT-NO'])]
        return data.set_index('ti')

    def get_infield_data(self, steps=None, pressure_demag=False, norm=False):
        """
        function calls Fabian2001.get_data and filters for PT ('LT-T-I') steps and NRM ('LT-NO').

        Parameters
        ----------
        steps
        pressure_demag
        norm

        Returns
        -------

        """
        data = self.get_data(steps=steps, pressure_demag=pressure_demag, norm=norm)
        data = data[np.in1d(data['LT_code'], ['LT-T-I', 'LT-NO'])]
        return data.set_index('ti')

    def get_ptrm_data(self, steps=None, pressure_demag=False, norm=False):
        """
        function calls Fabian2001.get_data and filters for PT ('LT-T-I') steps and NRM ('LT-NO').

        Parameters
        ----------
        steps
        pressure_demag
        norm

        Returns
        -------

        """
        data = self.get_data(steps=steps, pressure_demag=pressure_demag, norm=norm)
        th = data[np.in1d(data['LT_code'], ['LT-T-Z', 'LT-NO'])].set_index('ti')
        pt = data[np.in1d(data['LT_code'], ['LT-T-I', 'LT-NO'])].set_index('ti')

        ptrm = pt.copy()
        ptrm[['x', 'y', 'z', 'm']] = pt[['x', 'y', 'z', 'm']] - th[['x', 'y', 'z', 'm']]
        return ptrm

    def get_data(self, steps=None, pressure_demag=False, norm=False):
        """

        Parameters
        ----------
        steps
        pressure_demag
        norm

        Returns
        -------

        """
        # initiate data pd.DataFrame with LT_code and K ti,tj
        data = pd.DataFrame(index=('LT_code', 'x', 'y', 'z', 'm', 'level', 'ti', 'tj'))

        # if no steps are given, use internal
        if steps is None:
            steps = self.steps
        prev = 20
        for row, d in steps.iterrows():
            for column, t in enumerate(d):

                if t == '-':
                    continue

                typ = steps.columns[column]

                if np.isnan(t):
                    continue

                tau = self.tau(t)
                # self.log().debug('Calculating temperature %i (tau_i = %.2f)'%(t, tau))

                if typ == 'LT-NO':  # todo moment in x,y,z
                    m = self.moment(tau_i=tau, applied_field=0, pressure_demag=pressure_demag)
                elif typ == 'LT-T-Z':
                    m = self.moment(tau_i=tau, applied_field=0, pressure_demag=pressure_demag)
                elif typ == 'LT-T-I':
                    m = self.moment(tau_i=tau, applied_field=self.simparams['hlab'], pressure_demag=pressure_demag)

                # elif typ == 'LT-PTRM-I': #todo add AC, TR, CK steps
                #     NRM_Tj = self.get_moment(tau_i=self.tau(prev), hlab=0, pressure_demag=pressure_demag)
                #     pTRM_Ti = self.get_moment(tau_i=tau, hlab=hlab, pressure_demag=pressure_demag)
                #     NRM_Ti = self.get_moment(tau_i=tau, hlab=0, pressure_demag=pressure_demag)
                #     m = pTRM_Ti - NRM_Ti + NRM_Tj
                # #                         print(row, column, typ, t, tau, prev, m)

                else:
                    continue

                # add NRM step
                if typ == 'LT-T-Z' and tau == 0:
                    data[0] = ['LT-NO', 0, 0, m, m, t, t, prev]

                i = data.shape[1]
                data[i] = [typ, 0, 0, m, m, t, t, prev]

                # # add extra PT step after TH(rt)
                # if typ == 'LT-T-Z' and tau == 0:
                #     data[i+1] = ['LT-T-I', 0, 0, m, m, t, t + 273, prev + 273]
                if typ == 'LT-T-Z':
                    prev = t
        data = data.T
        if norm:
            data['x'] /= self.nrm
            data['y'] /= self.nrm
            data['z'] /= self.nrm
            data['m'] /= self.nrm
        return data

    ################################################################
    # PLOTTING

    def plot_surface(self, ax=None, title=None):
        #         from mpl_toolkits.mplot3d import proj3d
        #         def orthogonal_proj(zfront, zback):
        #             a = (zfront+zback)/(zfront-zback)
        #             b = -2*(zfront*zback)/(zfront-zback)
        #             return numpy.array([[1,0,0,0],
        #                                 [0,1,0,0],
        #                                 [0,0,a,b],
        #                                 [0,0,0,zback]])
        #         proj3d.persp_transformation = orthogonal_proj

        if ax is None:
            fig = plt.figure(figsize=plt.figaspect(0.45))
            ax = fig.add_subplot(111, projection='3d')

        x, y = plt.meshgrid(self.chi.columns, self.chi.index)
        ax.plot_surface(x, y, self.chi, color='w', alpha=1, antialiased=True, linewidth=0)
        ax.plot_wireframe(x, y, self.chi.values, color='k', rcount=20, ccount=20,
                          linewidth=0.2, antialiased=False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid()

        ax.view_init(elev=34, azim=-120)

        ax.set_xlabel('$\\tau_b$')
        ax.set_ylabel('$\\tau_{ub}$')

        ax.auto_scale_xyz([0, 1], [0, 1], [self.chi.min().min(), self.chi.max().max()])
        if title:
            ax.set_title(title)

    def plot_contour(self, ax=None, colorbar=False, **kwargs):
        if ax is None:
            ax = plt.gca()

        if 'title' in kwargs.keys():
            ax.set_title(kwargs.pop('title', ''))
        im = ax.imshow(self.chi.astype(float) / self.chi.max().max(), aspect='equal', origin=(0, 0),
                       extent=(0, 1, 0, 1), **kwargs)

        ax.set_xlabel('$\\tau_b$')
        ax.set_ylabel('$\\tau_{ub}$')

        if colorbar:
            plt.colorbar(im, ax=ax)
        else:
            return im

    def plot_arai(self, steps=None, hlab=1, pressure_demag=False, norm=False, ax=None, **kwargs):

        if self.steps is None and steps is None:
            print('cant plot Arai diagram,')
            return

        data = self.get_data(steps=steps, pressure_demag=pressure_demag)

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(6, 6))

        th = data[np.in1d(data['LT_code'], ['LT-T-Z', 'LT-NO'])].set_index('ti')
        pt = data[np.in1d(data['LT_code'], ['LT-T-I', 'LT-NO'])].set_index('ti')
        ck = data[data['LT_code'] == 'LT-PTRM-I'].set_index('ti')

        if 'title' in kwargs.keys():
            ax.set_title(kwargs.pop('title', ''))

        if norm:
            ax.plot((pt['m'] - th['m']) / th['m'].max(), th['m'] / th['m'].max(), '.--', **kwargs)
        else:
            ax.plot((pt['m'] - th['m']), th['m'], '.--', **kwargs)

        #        for ti, d in ck.iterrows():
        #            ax.plot([d['m'] - th[th.index == d['tj']]['m'].values[0],
        #                     pt[pt.index == d['tj']]['m'].values[0] - th[th.index == d['tj']]['m'].values[0]],
        #                    [th[th.index == d['tj']]['m'].values[0], th[th.index == d['tj']]['m'].values[0]], 'k-', lw=1.2)
        #            ax.plot([d['m'] - th[th.index == d['tj']]['m'].values[0], d['m'] - th[th.index == d['tj']]['m'].values[0]],
        #                    [th[th.index == d['tj']]['m'].values[0], th[th.index == ti]['m'].values[0]], 'k-', lw=1.2)
        #            ax.plot(d['m'] - th[th.index == d['tj']]['m'], th[th.index == ti]['m'], marker='^', mfc='None', mec='k')

        mx = ax.lines[0].get_data()[1].max()
        ax.plot([mx, 0], [0, mx], '--', color='grey')

        ax.set_xlabel('pTRM gained')
        ax.set_ylabel('NRM remaining')

        return data

    def plot_roquet(self, steps=None, hlab=1, pressure_demag=False, norm=False, ax=None, **kwargs):

        if self.steps is None and steps is None:
            print('cant plot Roquet plot,')
            return

        data = self.get_data(steps=steps, pressure_demag=pressure_demag)

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(6, 6))

        th = data[np.in1d(data['LT_code'], ['LT-T-Z', 'LT-NO'])].set_index('level')
        pt = data[np.in1d(data['LT_code'], ['LT-T-I', 'LT-NO'])].set_index('level')

        ax.set_title(kwargs.pop('title', None))

        ls = kwargs.pop('ls', '-')
        marker = kwargs.pop('marker', '.')

        if norm:
            norm = th['m'].max()
        else:
            norm = 1

        color = kwargs.pop('color', None)
        # pTRM plot
        ax.plot(th.index, (pt['m'] - th['m'])/norm,
                ls=ls,
                marker=marker,
                color='C0' if color is None else color,
                **kwargs)
        # SUM plot
        ax.plot(pt.index, pt['m']/norm,
                ls=ls,
                marker=marker,
                color='C1' if color is None else color, **kwargs)
        # TH plot
        ax.plot(th.index, th['m']/norm,
                ls=ls,
                marker=marker,
                color='C2' if color is None else color, **kwargs)

        ax.legend(['pTRM acquisition', 'Thermal demag.', 'sum'])
        ax.set_xlabel('T [C]')
        ax.set_ylabel('M [arb. units]')


if __name__ == '__main__':
    RockPy.log.setLevel('INFO')
    s = RockPy.Sample('test')
    mnp = s.add_simulation('paleointensity', preset='Fabian4a', a11=1, a12=0, a13=1, a1t=1)
    # mp = s.add_simulation('paleointensity', preset='Fabian7a', d1=0, d2=0.1, d3=0.3, dt=0.1, pressure_demag=True)
    #
    # mp.simobj.plot_arai()
    mnp.simobj.change_simparams(a11=0)
    mnp.simobj.plot_arai()
    plt.show()
    mnp.simobj.change_simparams(a11=1, a12=1, a1t=1)
    mnp.simobj.plot_arai()
    plt.show()
