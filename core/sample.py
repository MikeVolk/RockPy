import logging
from copy import deepcopy
import numpy as np
from collections import OrderedDict
from functools import partial

import RockPy
import RockPy.core.study

log = logging.getLogger(__name__)


class Sample(object):
    snum = 0

    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.MTYPE
        return logging.getLogger('RockPy.Sample')

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

    def __init__(self,
                 name=None,
                 comment='',
                 mass=None, height=None, diameter=None,
                 x_len=None, y_len=None, z_len=None,  # for cubic samples
                 sample_shape='cylinder',
                 coord=None,
                 samplegroup=None,
                 study=None,
                 create_parameter=True,
                 **kwargs):

        """
        Parameters
        ----------

           name: str
              name of the sample.
           mass: float
              mass of the sample. If not kg then please specify the mass_unit. It is stored in kg.
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
           samplegroup
           study
           create_parameter : bool
              if True: parameter measurements are created from (mass, x_len, y_len, z_len, height, diameter)
              if False: NO parameter measurements are created
        """
        # assign a sample id
        self.sid = id(self)

        # added sample -> Sample counter +1
        Sample.snum += 1

        # assign name to sample if no name is specified
        if not name:
            name = 'S%02i' % Sample.snum
        else:
            name = name  # unique name, only one per study
        # set name
        self.name = name

        self.log().debug('Creating Sample[%i] %s:%i' % (self.sid, name, self.snum))

        if not study:
            study = RockPy.core.study.Study
        else:
            if not isinstance(study, RockPy.core.study.Study):
                self.log().error('STUDY not a valid RockPy3.core.Study object. Using RockPy Masterstudy')
            study = RockPy.Study  # todo create MasterStudy on creation

        self.comment = comment

        # assign the study
        self.study = study
        # add sample to study
        # self.study.add_sample(sobj=self)

        # create a sample index number
        self.idx = self.study.n_samples

        # initiate samplegroups
        self._samplegroups = []

        # if samplegroup:#todo samplegroups
        #     for sg in samplegroup:
        #         self.add_to_samplegroup(gname=sg)

        # coordinate system
        self._coord = coord

        self.measurements = []  # list of all measurements
        self.mean_measurements = []  # list of all mean measurements #todo implement

        # adding parameter measurements
        if create_parameter:
            # for each parameter a measurement is created and then the measurement is added to the sample by calling
            # the add_measurement function
            if mass:
                self.add_measurement(mass=mass)
            if diameter:
                self.add_measurement(diameter=diameter)
            if height:
                self.add_measurement(height=height)

    def add_measurement(
            self,
            mtype=None,  # measurement type
            fpath=None, ftype=None,  # file path and file type
            idx=None,
            mdata=None,
            mobj=None,  # for special import of a measurement instance
            series=None,
            automatic_results=True,
            comment=None, additional=None,
            minfo=None,
            **kwargs):

        '''
        All measurements have to be added here

        Parameters
        ----------
        mtype: str
          the type of measurement
          default: None

        fpath: str
          the complete path to the measurement file
          default: None

        ftype: str
          the filetype from which the file is output
          default: 'generic'

        idx: index of measurement
          default: None, will be the index of the measurement in sample.measurements

        mdata: any kind of data that must fit the required structure of the data of the measurement
            will be used instead of data from file
            example:
                mdata = dict(mass=10)
                mdata = dict( x=[1,2,3,4], y = [1,2,3,4],...)

        mobj: RockPy3.Measurement object
            if provided, the object is added to self.measurements

        Returns
        -------
            RockPy3.measurement object
        '''

        # create the idx
        if idx is None:
            idx = len(self.measurements)

        ''' MINFO object generation '''
        if self.samplegroups:
            sgroups = self.samplegroups
        else:
            sgroups = None


        """ DATA import from mass, height, diameter, len ... """
        parameters = [i for i in ['mass', 'diameter', 'height', 'x_len', 'y_len', 'z_len'] if i in kwargs]
        if parameters:
            for mtype in parameters:
                mobj = RockPy.implemented_measurements[mtype](sobj=self, value=kwargs.pop(mtype),
                                                              **kwargs)

                # catch cant create error case where no data is written
                if mobj.data is None:
                    return

        # create MINFO if not provided
        if not minfo:
            minfo = RockPy.core.file_io.minfo(fpath=fpath,
                                              sgroups=sgroups,
                                              samples=self.name,
                                              mtypes=mtype, ftype=ftype,
                                              series=series,
                                              suffix=idx,
                                              comment=comment,  # unused for now
                                              read_fpath=False if mtype and ftype else True)

        """ DATA import from FILE """
        # if no mdata or measurement object are passed, create measurement file from the minfo object
        if not mdata and not mobj:
            # cycle through all samples
            for import_info in minfo.measurement_infos:
                mtype = import_info.pop('mtype')
                # check if mtype is implemented
                if not mtype in RockPy.implemented_measurements:
                    self.log().error('{} not implemented'.format(mtype))
                    continue
                # create measurement object
                mobj = RockPy.implemented_measurements[mtype].from_file(sobj=self,
                                                                        automatic_results=automatic_results,
                                                                        **import_info)



        """ DATA import from MDATA """
        if all([mdata, mtype]):
            if not mtype in RockPy.implemented_measurements:
                return
            mobj = RockPy.implemented_measurements[mtype](sobj=self, mdata=mdata, series=series, idx=idx,
                                                          automatic_results=automatic_results,
                                                          )


        """ DATA import from MOBJ """
        if mobj:
            if isinstance(mobj, tuple) or ftype == 'from_measurement':
                if not self.mtype_not_implemented_check(mtype=mtype):
                    return
                mobj = RockPy.implemented_measurements[mtype].from_measurement(sobj=self,
                                                                                mobj=mobj,
                                                                                automatic_results=automatic_results,
                                                                                **import_info)
            if not mobj:
                return

            self.log().info('ADDING\t << %s, %s >>' % (mobj.ftype, mobj.mtype()))

            self._add_mobj(mobj)

            # if minfo.sgroups:
            #     for sgroup in minfo.sgroups:
            #         self.add_to_samplegroup(sgroup, warn=False)
            return mobj

        else:
            self.log().error('COULD not create measurement << %s >>' % mtype)

    def _add_mobj(self, mobj):
        """
        Adds a measurement object to the Measurements ndarray

        Parameters
        ----------
        mobj: RockPy.Measurement object
        """

        if mobj not in self.measurements:
            self.measurements = np.append(self.measurements, mobj)

    @property
    def series(self):
        """
        Used to show what series are in the sample.

        Returns
        -------
        set: set of all series in the sample
        """
        return set(s.data for m in self.measurements for s in m.series)

    @property
    def samplegroups(self):
        if self._samplegroups:
            return self._samplegroups
        else:
            return 'None',

    @property
    def stypes(self):
        """
        Used to see what stypes are in the series
        Returns
        -------
            set: stypes
        """
        return set(s[0].lower() for s in self.series)

    @property
    def svals(self):
        """

        Returns
        -------

        """
        return set(s[1] for s in self.series if isinstance(s[1], (int, float)) if not np.isnan(s[1]))

    @property
    def sunits(self):
        """

        Returns
        -------
            set: series units
        """
        return set(s[2] for s in self.series)

    @property
    def mtypes(self):
        """
        Returns a set of measurement types contained in the sample.

        Returns
        -------
            set: mtype
        """
        return set(m.mtype for m in self.measurements)

    @property
    def mids(self):
        """
        Returns a list of all measurement ids of the individual measurements in a sample.

        Returns
        -------
            list: measurement ids
        """
        return np.array([m.id for m in self.measurements])

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
        returns a numpy array with all stypes.

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


if __name__ == '__main__':
    import RockPy

    a = Sample('Sample_test', mass=(30.9, 'mg'))
    print(a.measurements[0].data)
