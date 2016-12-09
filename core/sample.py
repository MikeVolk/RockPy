import logging
import itertools
from copy import deepcopy
import numpy as np
from collections import OrderedDict
from functools import partial

import RockPy

log = logging.getLogger(__name__)


class Sample(object):
    snum = 0

    def __lt__(self, other):

        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        return self.name < other.name

    def __iter__(self) -> object:

        return iter(self.measurements)

    @property
    def samplegroups(self):
        if self._samplegroups:
            return self._samplegroups
        else:
            return ('None',)

    def __init__(self,
                 name=None,
                 comment='',
                 mass=None, mass_unit='kg', mass_ftype='generic',
                 height=None, diameter=None,
                 x_len=None, y_len=None, z_len=None,  # for cubic samples
                 heightunit='mm', diameterunit='mm',
                 length_unit='mm', length_ftype='generic',
                 sample_shape='cylinder',
                 coord=None,
                 samplegroup=None,
                 study=None,
                 create_parameter=True,
                 ):

        """
        Parameters
        ----------

           name: str
              name of the sample.
           mass: float
              mass of the sample. If not kg then please specify the mass_unit. It is stored in kg.
           mass_unit: str
              has to be specified in order to calculate the sample mass properly.
           height: float
              sample height - stored in 'm'
           diameter: float
              sample diameter - stored in 'm'
           length_unit: str
              if not 'm' please specify
           length_machine: str
              if not 'm' please specify
           sample_shape: str
              needed for volume calculation
              cylinder: needs height, diameter
              cube: needs x_len, y_len, z_len
              sphere: diameter
           coord: str
              coordinate system
              can be 'core', 'geo' or 'bed'
           color: color
              used in plots if specified
           comment
           mass_ftype
           x_len
           y_len
           z_len
           heightunit
           diameterunit
           length_ftype
           samplegroup
           study
           create_parameter : bool
              if True: parameter measurements are created from (mass, x_len, y_len, z_len, height, diameter)
              if False: NO parameter measurements are created
        """
        # assign name to sample if no name is specified
        if not name:
            name = 'S%02i' % Sample.snum
        else:
            name = name  # unique name, only one per study

        if not study:
            study = RockPy.Study
        else:
            if not isinstance(study, RockPy.core.study.Study):
                self.log.error('STUDY not a valid RockPy3.core.Study object. Using RockPy Masterstudy')
            study = RockPy.Study

        self.comment = comment

        # assign the study
        self.study = study
        # add sample to study
        self.study.add_sample(sobj=self)

        # create a sample index number
        self.idx = self.study.n_samples

        # assign a sample id
        self.id = id(self)

        # added sample -> Sample couter +1
        Sample.snum += 1

        # initiate samplegroups
        self._samplegroups = []

        if samplegroup:
            for sg in samplegroup:
                self.add_to_samplegroup(gname=sg)

        # coordinate system
        self._coord = coord

        # logger.info('CREATING\t new sample << %s >>' % self.name) #todo logger

        self.measurements = []  # list of all measurents
        self.mean_measurements = []  # list of all mean measurements #todo implement

        # adding parameter measurements
        if create_parameter:
            # for each parameter a measurement is created and then the measurement is added to the sample by calliong
            # the add_measurement function
            pass  # todo maybe it would be better to pass a tuple (mass, massunit) to create the parameters and if only a number is passed we assume mg
            # todo implement mass
            # if mass:
            #     mass = RockPy3.implemented_measurements['mass'](sobj=self,
            #                                                     mass=mass, mass_unit=mass_unit, ftype=mass_ftype)
            #     self.add_measurement(mobj=mass)

    @property
    def series(self):
        '''
        Used to show what series are in the sample.

        Returns
        -------
        set: set of all series in the sample
        '''
        return set(s.data for m in self.measurements for s in m.series)

    @property
    def stypes(self):
        '''
        Used to see what stypes are in the series
        Returns
        -------
            set: stypes
        '''
        return set(s[0].lower() for s in self.series)

    @property
    def svals(self):
        '''

        Returns
        -------

        '''
        return set(s[1] for s in self.series if isinstance(s[1], (int, float)) if not np.isnan(s[1]))

    @property
    def sunits(self):
        '''

        Returns
        -------
            set: series units
        '''
        return set(s[2] for s in self.series)

    @property
    def mtypes(self):
        '''
        Returns a set of measurement types contained in the sample.

        Returns
        -------
            set: mtype
        '''
        return set(m.mtype for m in self.measurements)

    @property
    def mids(self):
        '''
        Returns a list of all measurement ids of the individual measurements in a sample.

        Returns
        -------
            list: measurement ids
        '''
        return [m.id for m in self.measurements]

    @property
    def mtype(self):
        """
        returns a numpy array with all mtypes.

        Notes
        -----
               idx(mtype) == idx(measurement(mtype))

        """
        return np.array([m.mtype for m in self.measurements])

    @property
    def stype(self):
        """
        returns a numpy array with all mtypes.

        Notes
        -----
               idx(stype) == idx(measurement(stype))
        """
        mx = max(len(m.series) for m in self.measurements)
        out = [[m.stypes[i] if i < len(m.stypes) else 'nan' for i in range(mx)] for m in self.measurements]
        return np.array(out)

    @property
    def sval(self):
        """
        returns a numpy array with all series values.

        Notes
        -----
               idx(sval) == idx(measurement(sval))
        """
        mx = max(len(m.series) for m in self.measurements)
        out = [[m.svals[i] if i < len(m.svals) else np.nan for i in range(mx)] for m in self.measurements]
        if mx == 1:
            return np.array(out).flatten()
        else:
            return np.array(out)

    def __repr__(self):
        return '<< RockPy3.Sample.{} >>'.format(self.name)
