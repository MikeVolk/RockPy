import logging
import numpy as np

import RockPy
import RockPy.core.study
import RockPy.core.file_io
import RockPy.core.utils
from RockPy.core.utils import to_tuple

import pandas as pd

log = logging.getLogger(__name__)


class Sample(object):
    snum = 0

    @property
    def results(self):
        self._results['sID'] = self.sid
        return self._results

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
        """
        Iterator that yields each measurement
        Returns
        -------
            RockPy.measurement
        """
        for m in self.measurements:
            yield m


    def __getitem__(self, item):

        if isinstance(item, int):
            return self.measurements[item]

        try:
            return self.get_measurement(mtype=item)
        except KeyError:
            pass

        raise KeyError('<< %s >> not valid item in sample object'%item)


    def __init__(self,
                 name=None,
                 comment='',
                 mass=None, massunit='kg',
                 height=None, diameter=None, lengthunit='m', *,

                 x_len=None, y_len=None, z_len=None,  # for cubic samples #todo implement volume
                 sample_shape='cylinder', #todo implement sample shape
                 coord=None,
                 sgroups=None,
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
           sgroups
           study
           create_parameter : bool
              if True: parameter measurements are created from (mass, x_len, y_len, z_len, height, diameter)
              if False: NO parameter measurements are created
        """
        # assign a sample id
        self.sid = id(self)
        # added sample -> Sample counter +1
        Sample.snum += 1

        # create the results DataFrame
        mcolumns = ['sID', 'mID']

        _results = pd.DataFrame(columns=mcolumns)
        self._results = _results.set_index('mID', drop=True)

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
            # study = RockPy.MasterStudy

        self.study = study

        # add sample to study
        # print(self.study.add_sample())
        # self.study.add_sample(sobj=self) #todo why does this not work

        # create a sample index number number of samples created at init
        self.idx = self.study.n_samples

        # initiate samplegroups
        self._samplegroups = []

        # if sgroups:#todo samplegroups
        #     for sg in sgroups:
        #         self.add_to_samplegroup(gname=sg)

        # coordinate system
        self._coord = coord

        self.measurements = []  # list of all measurements
        self.mean_measurements = []  # list of all mean measurements #todo implement

        # adding parameter measurements
        if create_parameter:
            # for each parameter a measurement is created and then the measurement is added to the sample by calling
            # the add_measurement function
            self.add_parameter_measurements(mass=mass, massunit=massunit,
                                            height=height, diameter=diameter, lengthunit=lengthunit)

        self.comment = comment
    
    @property
    def info(self):
        info = pd.DataFrame(columns=['mass[kg]', 'sample groups', 'mtypes', 'stypes', 'svals'])


        info.loc[self.name, 'mass[kg]'] = self.get_measurement('mass')[0].data['mass[kg]'].values[0] if self.get_measurement(
            'mass') else np.nan
        info.loc[self.name, 'sample groups'] = self._samplegroups if self._samplegroups else 'None'

        mtypes = [(mt, len(self.get_measurement(mtype=mt))) for mt in self.mtypes]

        if mtypes:
            info.loc[self.name, 'mtypes'] = mtypes if len(mtypes) > 1 else mtypes[0]
        else:
            info.loc[self.name, 'mtypes'] = None

        ''' STYPES '''
        if len(self.stypes) == 0:
            stypes = 'None'
        elif len(self.stypes) > 1:
            stypes = sorted(self.stypes)
        else:
            stypes = list(self.stypes)[0]

        info.loc[self.name, 'stypes'] = stypes

        ''' SVALS '''
        if len(self.svals) == 0:
            svals = 'None'
        elif len(self.svals) > 1:
            svals = sorted(self.svals)
        else:
            svals = list(self.stypes)[0]
        info.loc[self.name, 'svals'] = svals

        return info
    def add_simulation(self, mtype, idx=None, series= None, **sim_param):
        """
        add simulated measurements

        Parameters
        ----------
           mtype: str - the type of simulated measurement
           idx:
           sim_param: dict of parameters to specify simulation
        :return: RockPy.measurement object
        """
        mtype = mtype.lower()
        if mtype in RockPy.abbrev_to_classname:
            mtype = RockPy.abbrev_to_classname[mtype]

        if idx is None:
            idx = len(self.measurements)  # if there is no measurement index

        if mtype in RockPy.implemented_measurements:

            mobj = RockPy.implemented_measurements[mtype].from_simulation(sobj=self, idx=idx, series=series, **sim_param)

            if mobj:
                self.add_measurement(mtype=mtype, ftype='simulation', mobj=mobj, series=sim_param.get('series', None),
                                     warnings=False)
                return mobj
            else:
                self.log().info('CANT ADD simulated measurement << %s >>' % mtype)
                return None
        else:
            self.log().error(' << %s >> not implemented, yet' % mtype)

    def add_parameter_measurements(self, fpath=None, **kwargs):
        '''
        Method adds parameter measurements (mass, height, diameter...) to the sample. The parameters are passed as kwargs
        
        Parameters
        ----------
        kwargs
        '''

        # add parameter from fpath
        if fpath is not None:
            minfo = RockPy.core.file_io.ImportHelper.from_file(fpath=fpath)

            # if no parameter information in fpath
            if minfo is None:
                return

            for sinfo in minfo.gen_sample_dict:
                if sinfo['sname'] == self.name:
                    for info in ['mass', 'massunit', 'diameter', 'height', 'lengthunit']:
                        if not info in kwargs or kwargs[info] is None:
                            kwargs[info] = sinfo[info]

        parameters = [i for i in ['mass', 'diameter', 'height', 'x_len', 'y_len', 'z_len'] if i in kwargs if kwargs[i]]

        for mtype in parameters:
            if mtype == 'mass':
                value = (kwargs.pop(mtype), kwargs.pop('massunit', 'kg'))
            else:
                value = (kwargs.pop(mtype), kwargs.pop('lengthunit', 'm'))
            mobj = RockPy.implemented_measurements[mtype](sobj=self, value=value, **kwargs)
            # catch cant create error case where no data is written
            if mobj.data is None:
                return
            self.add_measurement(mtype=mtype, mobj=mobj, ftype='generic')

    def add_measurement(
            self,
            fpath=None, ftype=None, dialect=None,  # file path and file type
            mtype=None,  # measurement type
            idx=None,
            mdata=None,
            mobj=None,  # for special import of a measurement instance
            series=None,
            comment=None, additional=None,
            create_parameters=True,
            reload=False,
            **kwargs):

        """
        All measurements have to be added here

        Parameters
        ----------
        create_parameters
        additional
        mtype: str
          the type of measurement
          default_recipe: None

        fpath: str
          the complete path to the measurement file
          default_recipe: None

        ftype: str
          the filetype from which the file is output
          default_recipe: 'generic'

        dialect: str
          deals with small formatting differences in similar ftype

        idx: index of measurement
          default_recipe: None, will be the index of the measurement in sample.measurements

        mdata: any kind of data that must fit the required structure of the data of the measurement
            will be used instead of data from file
            example:
                mdata = dict(mass=10)
                mdata = dict( x=[1,2,3,4], y = [1,2,3,4],...)

        mobj: RockPy3.Measurement object
            if provided, the object is added to self.measurements

        series: lsist of tuples
            default: None
            A list of tuples consisting of (stype, svalue, sunit) 

        comment: str
            a comment 

        Returns
        -------
            RockPy3.measurement object
        """

        # create the idx
        if idx is None:
            idx = len(self.measurements)  # todo change so it counts the number of subclasses created

        ''' MINFO object generation '''
        if self.samplegroups:
            sgroups = self.samplegroups
        else:
            sgroups = None

        """ DATA import from mass, height, diameter, len ... """

        if create_parameters:
            self.add_parameter_measurements(fpath, **kwargs)

        # create the import helper
        if fpath and not any((mtype, ftype)):
            import_helper = RockPy.core.file_io.ImportHelper.from_file(fpath=fpath)
        else:
            self.log().debug('creating minfo from mtype: << %s >>'%mtype)
            import_helper = RockPy.core.file_io.ImportHelper.from_dict(fpath=fpath,
                                                                     sgroups=sgroups,
                                                                     snames=self.name,
                                                                     mtypes=mtype, ftype=ftype,
                                                                     series=series,
                                                                     suffix=idx,
                                                                     comment=comment,  # unused for now
                                                                     dialect=dialect,
                                                                     )

        """ DATA import from FILE """
        # if no mdata or measurement object are passed, create measurement file from the minfo object
        if mdata is None and mobj is None:
            # cycle through all measurements
            for import_info in import_helper.gen_measurement_dict:
                mtype = import_info['mtype']
                # check if mtype is implemented
                if not RockPy.core.utils.mtype_implemented(mtype):
                    self.log().error('{} not implemented'.format(mtype))
                    continue
                # create measurement object
                mobj = RockPy.implemented_measurements[mtype].from_file(sobj=self, reload=reload, **import_info)

        """ DATA import from MDATA """
        if all([mdata, mtype]):
            if not RockPy.core.utils.mtype_implemented(mtype):
                return
            mobj = RockPy.implemented_measurements[mtype](sobj=self, mdata=mdata, series=series, idx=idx)

        # DATA import from MOBJ
        if mobj:
            # todo from measurement

            self.log().info('ADDING\t << %s, %s >>' % (mobj.ftype, mobj.mtype))
            self._add_mobj(mobj)

            if RockPy.auto_calc_results:
                mobj.calc_results()
            return mobj
        else:
            self.log().error('COULD not create measurement << %s >>' % mtype)

    def _del_mobj(self, mobj):
        """
        removes the measurement object from the Measurements ndarray and removes the results from sample._results

        Parameters
        ----------
        mobj: RockPy.Measurement object
        """

        if mobj in self.measurements:
            idx = np.where(self.measurements == mobj)
            self.measurements = np.delete(self.measurements, idx)

        if mobj.mid in self._results.index:
            self._results = self._results.drop(mobj.mid)
            assert mobj.mid not in self.results.index

    def _add_mobj(self, mobj):
        """
        Adds a measurement object to the Measurements ndarray

        Parameters
        ----------
        mobj: RockPy.Measurement object
        """

        if mobj not in self.measurements:
            self.measurements = np.append(self.measurements, mobj)

    def calc_results(self, **kwargs):
        '''
        Method calls calc_results for each measurement
        
        Returns
        -------
            self.results
        '''
        for m in self.measurements:
            m.calc_results(**kwargs)
        return self.results.copy()

    def remove_measurement(self):  # todo write
        # needs to remove the measurement from measurement list and data from cls data
        raise NotImplementedError

    @property
    def series(self):
        """
        Used to show what series are in the sample.

        Returns
        -------
        set: set of all series in the sample
        """
        return set(s for m in self.measurements for s in m.series)

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
        return set(s for m in self.measurements for s in m.stypes if s)

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

    ### get functions

    def get_measurement(self,
                        mtype=None,
                        series=None,
                        stype=None, sval=None, sval_range=None,
                        mean=False,
                        invert=False,
                        mid=None,
                        result=None
                        ):
        """
        Returns a list of measurements of type = mtypes

        Parameters
        ----------
           mtype: list, str
              mtypes to be returned
           series: list(tuple)
              list of tuples, to search for several sepcific series. e.g. [('mtime',4),('gc',2)] will only return
              mesurements that fulfill both criteria.
           stype: list, str
              series type
           sval_range: list, str
              series range e.g. sval_range = [0,2] will give all from 0 to 2 including 0,2
              also '<2', '<=2', '>2', and '>=2' are allowed.
           sval: float
              series value to be searched for.
              caution:
                 will be overwritten when sval_range is given
           invert:
              if invert true it returns only measurements that do not meet criteria
           sval_range:
              can be used to look up measurements within a certain range. if only one value is given,
                     it is assumed to be an upper limit and the range is set to [0, sval_range]
           mean: bool
           mid: list(int)
            search for given measurement id

        Returns
        -------
            if no arguments are passed all sample.measurements
            list of RockPy.Measurements that meet search criteria or if invert is True, do not meet criteria.
            [] if none are found

        Note
        ----
            there is no connection between stype and sval. This may cause problems. I you have measurements with
               M1: [pressure, 0.0, GPa], [temperature, 100.0, C]
               M2: [pressure, 1.0, GPa], [temperature, 100.0, C]
            and you search for stypes=['pressure','temperature'], svals=[0,100]. It will return both M1 and M2 because
            both M1 and M2 have [temperature, 100.0, C].

        """

        mlist = self.measurements

        if mid is not None:
            mid = to_tuple(mid)
            mlist = (x for x in mlist if x.mid in mid)
            return list(mlist)

        if mtype:
            mtype = to_tuple(mtype)
            mtype = tuple(RockPy.abbrev_to_classname[mt] for mt in mtype)
            mlist = (m for m in mlist if m.mtype in mtype)

        if stype:
            mlist = (x for x in mlist if x.has_stype(stype=stype, method='any'))

        if sval_range is not None:
            sval_range = self._convert_sval_range(sval_range=sval_range, mean=mean)

            if not sval:
                sval = sval_range
            else:
                sval = to_tuple(())
                sval += to_tuple(sval_range)

        if sval is not None:
            mlist = (m for m in mlist if m.has_sval(sval=sval, method='any'))

        if series:
            series = RockPy.core.utils.tuple2list_of_tuples(series)
            mlist = (m for m in mlist if m.has_series(series=series, method='all'))

        if result:
            mlist = (m for m in mlist if m.has_result(result=result))

        if invert:
             return [m for m in self.measurements if m not in mlist]

        return list(mlist)

    def _convert_sval_range(self, sval_range, mean):
        """
        converts a string of svals into a list

        Parameters
        ----------
            sval_range: tuple, str
                series range e.g. sval_range = [0,2] will give all from 0 to 2 including 0,2
                also '<2', '<=2', '>2', and '>=2' are allowed.

        """
        out = []

        if mean:
            svals = set(s for m in self.mean_measurements for s in m.svals)
        else:
            svals = self.svals

        if isinstance(sval_range, tuple):
            out = [i for i in svals if sval_range[0] <= i <= sval_range[1]]

        if isinstance(sval_range, str):
            sval_range = sval_range.strip()  # remove whitespaces in case '> 4' is provided
            if '-' in sval_range:
                tup = [float(i) for i in sval_range.split('-')]
                out = [i for i in svals if min(tup) <= i <= max(tup)]

            if '<' in sval_range:
                if '=' in sval_range:
                    out = [i for i in svals if i <= float(sval_range.replace('<=', ''))]
                else:
                    out = [i for i in svals if i < float(sval_range.replace('<', ''))]
            if '>' in sval_range:
                if '=' in sval_range:
                    out = [i for i in svals if i >= float(sval_range.replace('>=', ''))]
                else:
                    out = [i for i in svals if i > float(sval_range.replace('>', ''))]

        return sorted(out)

    @property
    def mass(self, mid=None, mobj=None):
        """
        Returns the mass of that sample.

        # NOT IMPLEMENTED, YET #todo implement
        Specifying the measurement id (mid) or specifying a measurement returns the mass measurement just before
        that measurement.

        Parameters
        ----------
        mobj: RockPy.measurement
            if specified, the mass before that easurement is returned
        mid: int
            measurement id
            if specified, the mass before that easurement is returned

        Returns
        -------
            mass in kg
        """

        mass_measurements = self.get_measurement(mtype='mass')

        return [m.data['mass[kg]'].iloc[0] for m in mass_measurements]

    def __repr__(self):
        return '<< RockPy3.Sample.{} >>'.format(self.name)


if __name__ == '__main__':
    S = RockPy.Study()
    s = S.add_sample('62', mass=1)
