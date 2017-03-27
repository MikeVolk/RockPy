# coding: utf-8
import pandas as pd
import numpy as np
import scipy as sp

import os
import matplotlib.pylab as plt

from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Package
from pylatex.utils import italic
from copy import deepcopy
import tabulate

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s.%(funcName)s >> %(message)s', level=logging.DEBUG,
                    datefmt='%I:%M:%S')


class Fabian2001(object):
    presets = {'Fabian4a': dict(a11=0.01, a12=0, a13=0.5, a1t=0.5,
                                a21=0.01, a22=0, a23=0.5, a2t=0.5,
                                b1=0, b2=1, b3=0.5, bt=0.9),
               'Fabian4b': dict(a11=0.01, a12=0, a13=0.5, a1t=0.5,
                                a21=0.01, a22=0, a23=0.5, a2t=0.5,
                                b1=0, b2=1, b3=0.5, bt=0.1),
               'Fabian4c': dict(a11=0.1, a12=0, a13=0.5, a1t=0.5,
                                a21=0.1, a22=0, a23=0.5, a2t=0.5,
                                b1=0, b2=1, b3=0.5, bt=0.5),
               'Fabian5a': dict(a11=0.1, a12=0, a13=0.5, a1t=0.5,
                                a21=0.1, a22=0, a23=0.5, a2t=0.5,
                                b1=0, b2=1, b3=0.5, bt=0.1),
               'Fabian5b': dict(a11=0.1, a12=0, a13=0.5, a1t=0.5,
                                a21=0.1, a22=0, a23=0.5, a2t=0.5,
                                b1=0, b2=1, b3=0.5, bt=0.9),
               'Fabian5c': dict(a11=0.4, a12=0, a13=0.5, a1t=0.5,
                                a21=0.4, a22=0, a23=0.5, a2t=0.5,
                                b1=0, b2=1, b3=0.5, bt=0.9),
               'Fabian5d': dict(a11=0.4, a12=0, a13=0.5, a1t=0.5,
                                a21=0.4, a22=0, a23=0.5, a2t=0.5,
                                b1=0, b2=1, b3=0.5, bt=0.1),
               'Fabian5e': dict(a11=0., a12=0.5, a13=0.5, a1t=0.1,
                                a21=0., a22=0.5, a23=0.5, a2t=0.1,
                                b1=0, b2=1, b3=0.5, bt=0.1),
               'Fabian5f': dict(a11=0., a12=0.5, a13=0.5, a1t=0.9,
                                a21=0., a22=0.5, a23=0.5, a2t=0.9,
                                b1=0, b2=1, b3=0.5, bt=0.9),
               'Leonhard2a': dict(a11=0.001, a12=0, a13=0.5, a1t=0.5,
                                  a21=0.001, a22=0, a23=0.5, a2t=0.5,
                                  b1=0, b2=1, b3=0.2, bt=0.9),
               'Leonhard2b': dict(a11=0.05, a12=0, a13=0.5, a1t=0.5,
                                  a21=0.05, a22=0, a23=0.5, a2t=0.5,
                                  b1=0, b2=1, b3=0.5, bt=0.9),
               'Leonhard2c': dict(a11=0.4, a12=0, a13=0.5, a1t=0.5,
                                  a21=0.4, a22=0, a23=0.5, a2t=0.5,
                                  b1=0, b2=1, b3=0.5, bt=0.9)}

    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.MTYPE
        return logging.getLogger('RockPy.%s' % cls.__name__)

    @staticmethod
    def get_steps(steps, tmax=680., ck_every=2, tr_every=2, ac_every=2):
        """
        Creates a pandas DataFrame and a list of measurement steps.

        If filepath is provided a latex document is created

        Magic Lab-treatment codes :
            'TH': 'LT-T-Z'
            'PT': 'LT-T-I'
            'CK': 'LT-PTRM-I'
            'TR': 'LT-PTRM-MD'
            'AC': 'LT-PTRM-Z'
            'NRM': 'LT-NO'
        """

        if isinstance(steps, int):
            steps = np.linspace(20, tmax, steps)[1:]

        steps = sorted(list(steps))

        out = pd.DataFrame(columns=('LT-NO', 'LT-T-Z', 'LT-PTRM-I', 'LT-T-I', 'LT-PTRM-Z', 'LT-PTRM-MD'))
        out.loc[0, 'LT-NO'] = 20
        out.loc[0, 'LT-T-Z'] = 20

        pTH = 20  # previous th step

        for i, t in enumerate(steps):
            i += 1
            pTH = t

            out.loc[i, 'LT-T-Z'] = t

            if ck_every != 0 and i != 0 and not i % ck_every:
                ck_step = steps[i - ck_every]

                try:
                    if ck_step == pTH:
                        ck_step = steps[i - ck_every]
                        if i < len(steps):
                            out.loc[i + 1, 'LT-PTRM-I'] = ck_step
                    else:
                        out.loc[i, 'LT-PTRM-I'] = ck_step

                except IndexError:
                    pass

            out.loc[i, 'LT-T-I'] = t

            if ac_every != 0 and i != 0 and not i % ac_every:
                ac_step = steps[i - ac_every]

                if ac_step == pTH:
                    ac_step = steps[i - ac_every]
                    if i < len(steps):
                        out.loc[i + 1, 'LT-PTRM-Z'] = ac_step
                elif i <= len(steps):
                    out.loc[i, 'LT-PTRM-Z'] = ac_step

            if tr_every != 0 and not i % tr_every and not i == 0:
                out.loc[i, 'LT-PTRM-MD'] = t

        return out

    def __init__(self, preset=None,
                 a11=None, a12=None, a13=None, a1t=None,
                 a21=None, a22=None, a23=None, a2t=None,
                 b1=None, b2=None, b3=None, bt=None,
                 d1=0, d2=0, d3=None, dt=None, R=1, dc=1,
                 grid=100, hpal=1, ms=1, sum=True,
                 tc=560, temp_steps=11,
                 tmax=560, ck_every=2, tr_every=2, ac_every=2,
                 ):
        """
        Standard parameters as used by Leonhard 2004, Fig.2c, a1t,a2t,a13,a23 are not specified in the paper

        Parameters
        ----------
            Distibution for tau_b < tau_ub
            preset: str
                A collection of presets from the Fabian2001 and Leonhard2004 paper

            a11: float
                costant part of the distribution
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
            R
            dc
            
            grid: int
                default:100
                the number of blocking and unblocking temperatures to be calculated
            hpal: float
                default: 1
                reduced paleofield
            hlab: float
                default: 1
                reduced labfield
            ms: float
                saturation magnetization of the simulated sample
            
            # measurement simulation parameters e.g. how many points
            temp_steps: int, arraylike
                int will create an evenly spaced number of measurement temperatures up to tmax
                a list of temperatuires will create these temperatures
            tmax: float
                maximum temperature of the experiment
            ck_every: int
                after how many thermal demag steps a pTRM check is done
            tr_every: int
                after how many thermal demag steps a tail check is done
            ac_every: int                
                after how many thermal demag steps a additivity check is done

        """

        params = {'a11': a11, 'a12': a12, 'a13': a13, 'a1t': a1t,
                  'a21': a21, 'a22': a22, 'a23': a23, 'a2t': a2t,
                  'b1': b1, 'b2': b2, 'b3': b3, 'bt': bt,
                  'd3': d3, 'dt': dt, 'R': 1, 'dc': 1,
                  'tc': tc, 'grid': grid, 'hpal': hpal, 'ms': ms, 'sum': sum,
                  'temp_steps': temp_steps,
                  }


        self.steps = self.get_steps(temp_steps, tmax=tmax, ck_every=ck_every, tr_every=tr_every, ac_every=ac_every)

        if preset is None:
            preset = 'Fabian4a'

        if preset is not None and preset not in self.presets:
            print('preset << %s >> not implemented, chose from:' % preset)
            print(', '.join([i for i in self.presets]))
            print('using: Fabian Fig. 4a')
            preset = 'Fabian4a'

        # update the instance dictionary  with the preset
        self.__dict__.update(self.presets[preset])

        # update the instance dictionary  with the parameters, specified
        self.__dict__.update({k: v for k, v in params.items() if v is not None or k in ('d3', 'dt', 'R')})

        # create a dictionary with the simulation_parameters
        self.simparams = {k: self.__dict__[k] for k in params.keys()}

        # create reduced blocking and unblocking temperatures
        self.tau_b = [i / grid for i in range(0, grid + 1, 1)]
        self.tau_ub = self.tau_b

        # calculate gamma function for each blocking temperature
        self.gammas = [self.gamma(tau_b) for tau_b in self.tau_b]

        # initiate dataframe for characteristic function
        self.chi = self.get_chi_grid()

        # calculate the NRM (not pressure demagnetized)
        self.nrm = self.get_moment(1, self.hpal, pressure_demag=False)


        if temp_steps is not None:
            self.steps = self.steps.fillna('-')

        # calculate demagnetization function
        if d3 is None and dt is None:
            self.demag_dist = [R for t in self.tau_ub]
            demag = self.demag_dist
        else:
            self.demag_dist = d1 + d2*self.cauchy(np.array(self.tau_ub) - dt, d3)
            demag = np.cumsum(self.demag_dist)
            demag /= np.max(demag)

            if dc:
                demag += dc
                demag /= np.max(demag)

        self.demag = demag

    @classmethod
    def cauchy(cls, x, s):
        x = np.array(x)
        return 1 / (1 + (x / s) ** 2)

    def tau(self, t):
        return (t - 20) / (self.tc - 20)

    def beta(self, tau, call=''):
        if self.debug:
            print('\t\t:%s:calculating beta(%f) [%f,%f,%f,%f]' % (call, tau, self.b1, self.b2, self.bt, self.b3))
        return self.b1 + self.b2 * self.cauchy(tau - self.bt, self.b3)

    def lambda1(self, tau, call=''):
        '''
        controls the width of the width of the distribution chi(tb, ) for values of tub > tb
        '''
        if self.debug:
            print('\t\t:%s:calculating lambda1(%f) [%f,%f,%f,%f]' % (call, tau, self.a11, self.a12, self.a1t, self.a13))
        return self.a11 + self.a12 * self.cauchy(tau - self.a1t, self.a13)

    def lambda2(self, tau, call=''):
        '''
        controls the width of the width of the distribution chi(tb, ) for values of tub < tb
        '''
        if self.debug:
            print('\t\t:%s:calculating lambda2(%f) [%f,%f,%f,%f]' % (call, tau, self.a21, self.a22, self.a2t, self.a23))
        return self.a21 + self.a22 * self.cauchy(tau - self.a2t, self.a23)

    @classmethod
    def lam(cls, tau, a1, a2, at, a3):
        return a1 + a2 * cls.cauchy(tau - at, a3)

    def gamma(self, tau_b):
        """
        normalizes the integral of chi(tau_b, )
        """
        if self.debug:
            print('\tcalculating gamma(%f)' % tau_b)

        beta = self.beta(tau_b, call='gamma')

        lambda1 = self.lambda1(tau_b, call='gamma')
        lambda2 = self.lambda2(tau_b, call='gamma')

        # get the index of tau_b
        idx = self.tau_ub.index(tau_b)

        int1 = sum(self.cauchy(x, lambda1) for x in self.tau_b[idx:])
        int2 = sum(self.cauchy(x, lambda2) for x in self.tau_b[:idx + 1])

        return beta * 1 / (int1 + int2)  # as described by Fabian2001

    def chi(self, tau_b, tau_ub):
        if self.debug:
            print('\t\tcalculating Chi(%.2f,%.2f):' % (tau_b, tau_ub))

        gamma = self.gammas[self.tau_b.index(tau_b)]

        if tau_b < tau_ub:
            cauchy = self.cauchy(tau_ub - tau_b, self.lambda1(tau_b, call='chi'))
        else:
            cauchy = self.cauchy(tau_ub - tau_b, self.lambda2(tau_b, call='chi'))

        return gamma * cauchy

    def get_H_matrix(self, tau_i, hlab, pressure_demag=False):
        """

        Parameters
        ----------
        tau_i

        Returns
        -------

        """
        data = np.zeros((len(self.tau_ub), len(self.tau_b)))
        for row, tau_b in enumerate(self.tau_b):
            for column, tau_ub in enumerate(self.tau_ub):
                data[column, row] = self.H(tau_i, tau_b=tau_b, tau_ub=tau_ub, hlab=hlab, pressure_demag=pressure_demag)
        out = pd.DataFrame(index=self.tau_ub, columns=self.tau_b, data=data)
        return out

    def H(self, tau_i, tau_b, tau_ub, hlab, pressure_demag=False):
        """
        Calculates the potential field for a given tauu_i and tau_b
        Parameters
        ----------
        tau_i
        tau_b
        tau_ub
        hlab

        Returns
        -------
        """

        # def LabMag(t, ttail):
        #
        #     if Tub > (t + ttail) and Tb <= t:
        #         ExtFieldVec * KappaFunc[Tb, Tub]
        #     elif (Tub > t and Tb > t) or (t == 0):
        #         return ExtFieldVec * KappaFunc[Tb, Tub]
        #     elif Tub <= (t + ttail) and Tb <= t and t > 0:
        #         return LabFieldVec * (KappaFunc[Tb, Tub])
        #     else
        #         return 0, 0, 0

        if tau_ub > tau_i or tau_i == 0:
            if pressure_demag:
                idx = np.argmin(np.abs(np.array(self.tau_ub) - tau_ub))
                return self.demag[idx] * self.hpal
            else:
                return self.hpal

        elif tau_ub <= tau_i and tau_b <= tau_i and tau_i > 0:
            return hlab
        else:
            return 0

    def get_chi_grid(self):
        """
        Method that calculates a matrix of chi values.
        For each tb there is a distribution of tub, chi calculates them all
        
        Returns
        -------
            pandas DataFrame with chi values (columns = blocking temperatures, indices = unblocking temperatures)
        """
        data = np.zeros((len(self.tau_ub), len(self.tau_b)))
        for row, tb in enumerate(self.tau_b):
            gamma = self.gamma(tb)
            for column, tub in enumerate(self.tau_ub):
                data[column, row] = self.chi(tb, tub)
        return pd.DataFrame(index=self.tau_ub, columns=self.tau_b, data=data)

    def get_moment_old(self, tau_i, hlab=1, pressure_demag=False):
        integral = []
        for j, t_b in enumerate(self.tau_b):
            integral.append(sum(self.H(tau_i, t_b, t_ub, hlab, pressure_demag) * self.chi.values[i, j] for i, t_ub in
                                enumerate(self.tau_ub)))
        integral = sum(integral)
        return integral

    def get_moment(self, tau_i, hlab=1, pressure_demag=False):
        """

        Parameters
        ----------
        tau_i: float
        hlab : float
        pressure_demag: bool
            default: False
            if True a cauchy distributed demagnetization of unblocking temperatures is calculated
        """
        h = self.get_H_matrix(tau_i=tau_i, hlab=hlab, pressure_demag=pressure_demag)
        return (h * self.chi).sum().sum()

    def get_data(self, steps=None, hlab=1, pressure_demag=False):
        data = pd.DataFrame(index=('type', 'ti', 'tj', 'm'))

        if steps == None:
            steps = self.steps

        i = 0
        prev = 20
        for row, d in steps.iterrows():
            for column, t in enumerate(d):
                i += 1
                typ = steps.columns[column]
                if not t == '-':
                    tau = self.tau(t)
                    if typ == 'LT-T-Z':
                        m = self.get_moment(tau_i=tau, hlab=0, pressure_demag=pressure_demag)
                    elif typ == 'LT-T-I':
                        m = self.get_moment(tau_i=tau, hlab=hlab, pressure_demag=pressure_demag)
                    # elif typ == 'LT-PTRM-I':
                    #     NRM_Tj = self.get_moment(tau_i=self.tau(prev), hlab=0, pressure_demag=pressure_demag)
                    #     pTRM_Ti = self.get_moment(tau_i=tau, hlab=hlab, pressure_demag=pressure_demag)
                    #     NRM_Ti = self.get_moment(tau_i=tau, hlab=0, pressure_demag=pressure_demag)
                    #     m = pTRM_Ti - NRM_Ti + NRM_Tj
                    # #                         print(row, column, typ, t, tau, prev, m)

                    else:
                        continue

                    data[i] = [typ, t, prev, m]
                    if typ == 'LT-T-Z' and tau == 0:
                        i += 1
                        data[i] = ['LT-T-I', t, prev, m]
                    if typ == 'LT-T-Z':
                        prev = t

        return data.T
