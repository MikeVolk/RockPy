import logging
import os
import shutil
from collections import OrderedDict
from copy import deepcopy

import RockPy
import RockPy.core
import numpy as np
import pandas as pd
import RockPy.core.utils
import RockPy.core.result
import inspect
from RockPy.core.utils import to_tuple, tuple2list_of_tuples

class Measurement(object):
    """

    Measurement Design Guide
    ========================

    How to implement a measurement
    ++++++++++++++++++++++++++++++

    1. measurement formatters
    -------------------------
        The measurement formatter uses the data from the Package.io.ftype and turns it into a RockPyData
        object fitting the Measurement.data specifications of the measurement.

    2. results
        + dependent results need dependent parameter
        + recipes
        + calculate_functions




    """

    _results = None
    _result_classes_list = []

    _clsdata = []  # raw data do not manipulate
    _sids = []
    _mids = []

    clsdata = []

    n_created = 0

    possible_plt_props = ['agg_filter', 'alpha', 'animated', 'antialiased', 'axes', 'clip_box', 'clip_on', 'clip_path',
                          'color', 'contains', 'dash_capstyle', 'dash_joinstyle', 'dashes', 'drawstyle', 'figure',
                          'fillstyle', 'gid', 'label', 'linestyle', 'linewidth', 'lod', 'marker', 'markeredgecolor',
                          'markeredgewidth', 'markerfacecolor', 'markerfacecoloralt', 'markersize', 'markevery',
                          'path_effects', 'picker', 'pickradius', 'rasterized', 'sketch_params', 'snap',
                          'solid_capstyle', 'solid_joinstyle', 'transform', 'url', 'visible', 'xdata', 'ydata',
                          'zorder']

    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.MTYPE
        return logging.getLogger('RockPy.%s' % cls.cls_mtype())

    @classmethod
    def cls_mtype(cls):
        """
        Returns the measurement type of the measurement

        Returns
        -------
            str


        """
        return cls.__name__.lower()

    @classmethod
    def subclasses(cls):
        """
        Returns a list of the implemented_visuals names
        """
        return [i.__name__.lower() for i in cls.inheritors()]

    @classmethod
    def inheritors(cls):
        """
        Method that gets all children and childrens-children ... from a class

        Returns
        -------
           list
        """
        subclasses = set()
        work = [cls]
        while work:
            parent = work.pop()
            for child in parent.__subclasses__():
                if child not in subclasses:
                    subclasses.add(child)
                    work.append(child)
        return subclasses

    ''' implemented dicts '''
    @property
    def midx(self):
        return self.__class__._mids.index(self.mid)

    @classmethod
    def implemented_ftypes(cls):  # todo move into RockPy core.core has nothing to do with measurement
        """
        Dictionary of all implemented filetypes.

        Looks for all subclasses of RockPy3.core.ftype.io
        generating a dictionary of implemented machines : {implemented out_* method : machine_class}

        Returns
        -------

        dict: classname:
        """
        implemented_ftypes = {cl.__name__.lower(): cl for cl in RockPy.core.ftype.Ftype.__subclasses__()}
        return implemented_ftypes

    @classmethod
    def ftype_formatters(cls):
        '''
        This method returns a collection of the implemented measurement formatters for a class.
        Measurement formatters are important!
        If there is no formatter for the measurement class, the measurement has not been implemented for this filetype.
        While the filetype may exist (i.e. other measurements have been implemented) this particular measurement will
        not work.

        Returns
        -------
            dict:
                Dictionary with a formatter_name : formatter_method
        '''

        measurement_formatters = {
            i.replace('format_', '').lower(): getattr(cls, i) for i in dir(cls) if i.startswith('format_')
            }
        return measurement_formatters

    @classmethod
    def correct_methods(cls) -> dict:
        """
        Dynamically searches through the class and finds all correction methods

        Returns
        -------
            dict:
                dictionary with all implemented correction methods

        """
        return {i.replace('correct_', ''): getattr(cls, i) for i in dir(cls)
                if i.startswith('correct_')
                }

    ####################################################################################################################

    @classmethod
    def get_subclass_name(cls):
        return cls.__name__

    """    measurement creation through function    """

    @classmethod
    def from_mdata(cls):
        pass

    @classmethod
    def from_file(cls, sobj,
                  fpath=None, ftype='generic',  # file path and file type
                  idx=None, sample_name=None,
                  series=None, dialect=None,
                  **options
                  ):
        '''
        This is the way a measurement class is created, if it is directly imported from a file.

        Parameters
        ----------
        sobj: RockPy.Sample
            The rockPy.Sample object the measurement belongs to
        fpath: str
            The path to the measurement file
        ftype: str
            The type of measurement file, e.g. vsm, vftb ...
        idx: int
            index of the measurement if needed
        sample_name: str
            name of the sample in case there are many samples in the data file
        series: tuple
            a series tuple with
        options: dict

        Note
        ----
        Checks if ftype is implemented -> check if ftype_formatter is implemented -> create mdata -> return measurement class

        Returns
        -------
            RockPy.measurement object
        '''

        # initialize ftype_data
        ftype_data = None

        # check if measurement is implemented
        if ftype in cls.implemented_ftypes():
            # read the ftype data from the file
            ftype_data = cls.implemented_ftypes()[ftype](fpath, sobj.name, dialect=dialect)
        else:
            cls.log().error('CANNOT IMPORT ')
            cls.log().error('ftype not in implemented ftypes: %s '%', '.join(cls.implemented_ftypes().keys()))

        # check wether the formatter for the ftype is implemented
        if ftype_data and ftype in cls.ftype_formatters():
            cls.log().debug('ftype_formatter << %s >> implemented' % ftype)
            mdata = cls.ftype_formatters()[ftype](ftype_data, sobj_name=sobj.name)
            if mdata is None:
                cls.log().debug('mdata is empty -- measurement may not be created, check formatter')
                return
        else:
            cls.log().error('UNKNOWN ftype: << %s >>' % ftype)
            cls.log().error('most likely cause is the \"format_%s\" method is missing in the measurement << %s >>' % (
                ftype, cls.__name__))
            return

        return cls(sobj=sobj, fpath=fpath, ftype=ftype,
                   mdata=mdata,
                   series=series, idx=idx, ftype_data=ftype_data, **options)

    @classmethod
    def from_simulation(cls, sobj=None, idx=None, **parameter):
        """
        pseudo abstract method that should be overridden in subclasses to return a simulated measurement
        based on given parameters
        """
        return None

    @classmethod
    def from_measurements(cls):
        """
        pseudo abstract method that should be overridden in subclasses to return a combined measurement from several measurements.

        e.g. pARM spectra -> ARM acquisition
        """
        pass

    @classmethod
    def from_result(cls, **parameter):
        """
        pseudo abstract method that should be overridden in subclasses to return a measurement created from results
        """
        return None

    def set_initial_state(self,
                          mtype=None, fpath=None, ftype=None,  # standard
                          mobj=None, series=None,
                          ):
        """
        This methods creates an initial state for the measurement. It creates a new measurement, the initial state (ISM) of base measurement (BSM). It calls the measurement constructor and assigns the created measurement (ISM) to the self.initial_state parameter of the BSM. It also sets a flag for the ISM to check if a measurement is a ISM.

        It is possible to create a ISM from a existing measurement object.

        Parameters
        ----------
           mtype: str
              measurement type
           mfile: str
              measurement data file
           machine: str
              measurement machine
            mobj: RockPy3.MEasurement object
        """
        # todo needs to be rewritten
        raise NotImplementedError

    def __init__(self, sobj,
                 fpath=None, ftype=None, mdata=None,
                 ftype_data = None,
                 series=None,
                 idx=None,
                 **options
                 ):
        """
        Constructor of the measurement class.

        Several checks have to be done:
            1. is the measurement implemented:
                this is checked by looking if the measurement is in the RockPy3.implemented_measurements
            2. if mdata is given, we can directly create the measurement #todo from_mdata?
            3. if the file format (ftype) is implemented #todo from_file
                The ftype has to be given. This is how RockPy can format data from different sources into the same
                format, so it can be analyzed the same way.

        Parameters
        ----------
            sobj: RockPy3.Sample
                the sample object the measurement belongs to. The constructor is usually called from the
                Sample.add_measurement method
            mtype: str
                MANDATORY: measurement type to be imported. Implemented measurements can be seen when calling
            fpath: str
                path to the file including filename
            ftype: str
                file type. e.g. vsm
            mdata: pd.DataFrame
                when mdata is set, this will be directly used as measurement data without formatting from file

        Note
        ----
            when creating a new measurement it automatically calculates all results using the standard prameter set
        """

        self.mtype = self.cls_mtype()
        self.mid = id(self)
        self.__idx = self.n_created

        self.sobj = sobj
        self.sid = self.sobj.sid

        self.log().debug('Creating measurement: id:{} idx:{}'.format(self.mid, self.__idx))

        # add the data to the clsdata
        self.append_to_clsdata(mdata)

        # flags for mean and the base measurements
        self.is_mean = options.get('ismean', False)  # flag for mean measurements
        self.base_measurements = options.get('base_measurements',
                                             False)  # list with all measurements used to generate the mean


        self.ftype = ftype
        self.fpath = fpath
        self.ftype_data = ftype_data

        ''' initial state '''
        self.is_initial_state = False
        self.initial_state = options.get('initial_state', False)

        ''' calibration, correction and holder'''
        self.calibration = None
        self.holder = None
        self._correction = []

        # normalization
        self.is_normalized = False  # normalized flag for visuals, so its not normalized twice
        self.norm = None  # the actual parameters
        self.norm_factor = 1

        ''' series '''
        self._series = []

        # add series if provided
        if series:
            series = RockPy.core.utils.tuple2list_of_tuples(series)
            for s in series:
                self.add_series(*s)

        self.idx = idx if idx else self.__idx  # external index e.g. 3rd hys measurement of sample 1

        ''' initialize results '''
        self._result_classes()
        self.__init_results()
        self.__class__.n_created += 1

    def _result_classes(self): #todo make iterator
        """
        Mothod that gets all result classes of the measurement
         
        Returns
        -------
            list: <class 'RockPy.Result'>
            
        """
        if not Measurement._result_classes_list:
            out = []
            for name, cls in inspect.getmembers(self.__class__):
                if not inspect.isclass(cls):
                    continue
                if isinstance(cls, RockPy.core.result.Result) or issubclass(cls, RockPy.core.result.Result):
                    out.append(cls)
        return out


    def __init_results(self, **parameters): #todo is _results needed?
        """ 
        creates a list of results for that measurement 
        
        """
        if self._results is None:
            self._results = {}
            for cls in self._result_classes():
                    res = cls.__name__.replace('result_', '')
                    instance = cls(mobj=self, **parameters)
                    # replace the method with the instance
                    self._results[res] = instance
                    self.log().debug('replacing class %s with instance %s' % (cls.__name__, instance))
                    setattr(self, cls.__name__, instance)

        return self._results

    @property
    def mass(self):
        mass = self.get_mtype_prior_to(mtype='mass')
        return mass.data['data']['mass'].v[0] if mass else None

    def get_recipes(self, res):
        """
        returns all result recipes for a given result
        :param res:
        :return:
        """
        result = self._results[res]
        recipes = [r for r in result._recipes()]
        return set(recipes)

    def has_result(self, result):
        """
        Checks if the measurement has the specified result
        
        Parameters
        ----------
        result: str

        Returns
        -------
            bool
        """

        return True if result in self._results.keys() else False



    def __lt__(self, other):
        """
        for sorting measurements. They are sorted by their index
        :param other:
        :return:
        """
        # if idx not integer
        try:
            return int(self.idx) < int(other.idx)
        except ValueError:
            pass
        # fall back, internal idx
        try:
            return int(self.__idx) < int(other._idx)
        except ValueError:
            pass

    def __repr__(self):
        if self.is_mean:
            add = 'mean_'
        else:
            add = ''
        return '<<RockPy3.{}.{}{}{} at {}>>'.format(self.sobj.name, add, self.mtype,
                                                    '['+';'.join(['{},{}({})'.format(i[0],i[1],i[2]) for i in self.get_series()])+']' if self.has_series() else '',

                                                    hex(self.mid))

    def __hash__(self):
        return hash(self.mid)

    def __eq__(self, other):
        return self.mid == other.mid

    def __add__(self, other):
        """
        Adds one measurement to another. First the measurements are interpolated to the same variables, then
        the interpolated numbers are subtracted.

        Parameters
        ----------
        other

        Returns
        -------

        """
        # deepcopy both data
        first = deepcopy(self)
        other = deepcopy(other)

        for dtype in first.data:
            # if one of them does not have the dtype skip it
            if first.data[dtype] is None or other.data[dtype] is None:
                continue
                # get the variables for both first, other of that dtype
            vars1 = set(first.data[dtype]['variable'].v)
            vars2 = set(other.data[dtype]['variable'].v)

            # get variables that are different
            diff = vars1 ^ vars2
            # if the variables are different interpolate the values
            if diff:
                vars = vars1 | vars2
                first.data[dtype] = first.data[dtype].interpolate(new_variables=vars)
                other.data[dtype] = other.data[dtype].interpolate(new_variables=vars)
            first.data[dtype] = first.data[dtype].eliminate_duplicate_variable_rows() + other.data[
                dtype].eliminate_duplicate_variable_rows()
            # remove all nan entries
            first.data[dtype] = first.data[dtype].filter(~np.any(np.isnan(first.data[dtype].v), axis=1))
            first.data[dtype] = first.data[dtype].sort()
        return self.sobj.add_measurement(mtype=first.mtype, mdata=first.data)

    def __sub__(self, other):
        first = deepcopy(self)
        other = deepcopy(other)

        for dtype in first.data:
            if first.data[dtype] is None or other.data[dtype] is None:
                continue
            vars1 = set(first.data[dtype]['variable'].v)
            vars2 = set(other.data[dtype]['variable'].v)
            diff = vars1 ^ vars2
            if diff:
                vars = vars1 | vars2
                first.data[dtype] = first.data[dtype].interpolate(new_variables=vars)
                other.data[dtype] = other.data[dtype].interpolate(new_variables=vars)
            first.data[dtype] = first.data[dtype].eliminate_duplicate_variable_rows() - other.data[
                dtype].eliminate_duplicate_variable_rows()
            first.data[dtype] = first.data[dtype].sort()
        return self.sobj.add_measurement(mtype=first.mtype, mdata=first.data)


    def append_to_clsdata(self, mdata):
        '''
        Method that adds data to the clsdata and _clsdata of the class

        Parameters
        ----------
        mdata: pandas.Dataframe

        Returns
        -------

        '''

        if mdata is None:
            self.log().error('NO mdata in instance cant assign sid, mid')
            return

        # add column with measurement ids (mID) and sample ids (sID)
        mdata['mID'] = self.mid
        mdata['sID'] = self.sid

        # append data to raw data (_clsdata)
        self.__class__._clsdata.append(mdata)
        self.__class__._sids.append(self.sid)
        self.__class__._mids.append(self.mid)

        # append data to manipulate data (clsdata)
        self.__class__.clsdata.append(mdata)

    @classmethod
    def remove_from_clsdata(cls, mid: int):
        '''
        method that removes the data of that measurement from the _clsdata and cls data as well as _sids and _mids
        lists.

        Parameters
        ----------
        mid: int
             the id of the measurement to be removed from the measurement class
        '''

        midx = cls._mids.index(mid)

        for lst in [cls._clsdata, cls.clsdata, cls._mids, cls._sids]:
            del lst[midx]

    @property
    def m_idx(self):
        return np.where(self.sobj.measurements == self)[0]  # todo change could be problematic if changing the sobj

    @property
    def fname(self):
        """
        Returns only filename from self.file

        Returns
        -------
           str: filename from full path
        """
        return os.path.basename(self.fpath) if self.fpath is not None else None

    @property
    def has_initial_state(self):
        """
        checks if there is an initial state
        """
        return True if self.initial_state else False

    @property
    def stypes(self):
        """
        list of all stypes
        """
        return [s[0] for s in self.series]

    @property
    def svals(self):
        """
        set of all svalues
        """
        return [s[1] for s in self.series]

    @property
    def data(self):
        # get the index of the sample / measurement
        try:
            midx = self.__class__._mids.index(self.mid)

            # assertion should be put, but sid may be more than one item
            # sidx = self._sids.index(self.sid)
            # assert midx == sidx, 'index of mid (%i) and sid (%i) do not match' %(sidx, midx)
            return self.clsdata[midx]
        except KeyError:
            return

    def transform_data_coord(self, final_coord):
        """
        transform columns x,y,z in data object to final_coord coordinate system
        """
        if self._actual_data_coord != final_coord:
            # check if we have x,y,z columns in data
            # TODO look for orientation measurements
            # extract coreaz, coredip, bedaz and beddip
            # transform x, y, z columns in _data['data']
            # final_xyz = RockPy3.utils.general.coord_transform(initial_xyz, self._actual_data_coord, final_coord)
            self.log.warning('data needs to be transformed from %s to %s coordinates. NOT IMPLEMENTED YET!' % (
                self._actual_data_coord, final_coord))
            # return self._data
        else:
            self.log.debug('data is already in %s coordinates.' % final_coord)
            # return self._data

    @property
    def correction(self):
        """
        If attribute _correction does not exist, it is created and returned. After that the attribute always exists.
        This is a way so any _attribute does not have to be created in __init__
        A string of the used calculation method has to be appended for any corrections.
        This way we always know what has been corrected and in what order it has been corrected
        """
        return RockPy.core.utils.set_get_attr(self, '_correction', value=list())

    def reset_data(self): #todo rewrite new data pandas
        """
        Resets all data back to the original state. Deepcopies _raw_data back to _data and resets correction
        """
        self._data = deepcopy(self._raw_data)
        # create _correction if not exists
        self.set_get_attr('_correction', value=list())
        self._correction = []

    ####################################################################################################################
    ''' DATA RELATED '''

    ''' Calculation and parameters '''

    def delete_dtype_var_val(self, dtype, var, val):
        """
        deletes datapoint with var = var and val = val

        Parameters
        ----------
           dtype: the step type to be deleted e.g. th
           var: the variable e.g. temperature
           val: the value of that step e.g. 500

        example: measurement.delete_step(dtype='th', var='temp', val=500) will delete the th step where the variable (temperature) is 500
        """
        # idx = self._get_idx_dtype_var_val(dtype=dtype, var=var, val=val)
        # self.data[dtype] = self.data[dtype].filter_idx(idx, invert=True)
        # return self
        raise NotImplementedError

    ##################################################################################################################
    ''' RESULT related '''

    @property
    def results(self):
        '''
        Property that returns the

        Returns
        -------

        '''
        try:
            return self.sobj.results.loc[self.mid, :]
        except KeyError:
            return

    def calc_all(self, **parameters):
        for resname, res in self._results.items():
            r = res(**parameters)
        return self.results


    ##################################################################################################################
    ''' SERIES related '''

    @property
    def series(self):
        """
        returns all series objects. If there are no series it will return a empty series ('none', nan, '')

        Returns
        -------
        list
            Series objects

        """
        if self._series:
            return self._series
        else:
            series = (None, np.nan, None)  # no series
            return [series]

    def has_sval(self, svals=None, method='all'):
        """
        Checks if a measurement has any, all, or none of the specified svals

        Parameters
        ----------
            sval: str, list, tuple
                stypes to test for
            method: 'all' or 'any' or 'none'
                defines the check method:
                    all: all svals need to exist
                    any: any svals needs to exist
                    none: none of the svals can exist

        Returns
        -------
            bool
            True if all stypes exist, False if not
            If stype = None : returns True if no series else True
        """
        if not self._series:
            return False

        if svals is not None:
            svals = to_tuple(svals)
            if method == 'all':
                return True if all(i in self.svals for i in svals) else False
            if method == 'any':
                return True if any(i in self.svals for i in svals) else False
            if method == 'none':
                return True if not any(i in self.svals for i in svals) else False
        else:
            return True if not self.svals else False

    def has_stype(self, stype=None, method='all'):
        """
        Checks if a measurement has all of the specified series

        Parameters
        ----------
            stype: str, list, tuple
                stypes to test for
            method: 'all' or 'any' or 'none'
                defines the check method:
                    all: all series need to exist
                    any: any series needs to exist
                    none: no series can exist

        Returns
        -------
            bool
            True if all stypes exist, False if not
            If stype = None : returns True if no series else True
        """
        if not self._series:
            return False

        if stype is not None:
            stype = to_tuple(stype)
            if method == 'all':
                return True if all(i in self.stypes for i in stype) else False
            if method == 'any':
                return True if any(i in self.stypes for i in stype) else False
            if method == 'none':
                return True if not any(i in self.stypes for i in stype) else False
        else:
            return True if not self.stypes else False

    def has_series(self, series=None, method='all'):
        '''
        Method tests for given series. 


        Parameters
        ----------
        series: list of tuples 
            each element is (stype, sval, sunit) tuple
        method: str:
            'all': returns True if measurement posesses ALL series
            'any': returns True if measurement posesses ONE or more series
            'None': returns True if measurement posesses NONE of the provided series

        Returns
        -------
            bool
            returns true if Nothing is passes
        '''

        if series is not None:
            series = tuple2list_of_tuples(series)
            if method == 'all':
                return True if all(i in self.series for i in series) else False
            if method == 'any':
                return True if any(i in self.series for i in series) else False
            if method == 'none':
                return True if not any(i in self.svals for i in series) else False
        else:
            return True if self._series else False

    def add_series(self, stype, sval, sunit=None):  # todo add (stype,sval,sunit) type calling
        #todo change to set_series with styoe, sval, suit, series
        """
        adds a series to measurement.series

        Parameters
        ----------
           stype: str
              series type to be added
           sval: float or int
              series value to be added
           unit: str
              unit to be added. can be None #todo change so it uses Pint

        Returns
        -------
           [RockPy3.Series] list of RockPy series objects

        """
        series = (stype, sval, sunit)

        # set stype as column in results
        self.sobj._results.loc[self.mid, stype+'*'] = sval

        self._series.append(series)

    def remove_series(self, stype):
        """
        Removes a series from the measurement

        Parameters
        ----------
        stype: str
            the series stype to be removes

        """
        # get the series
        stup = self.get_series(stype=stype)[0]
        # remove series from _series list
        self._series.remove(stup)

        #remove the series from the results
        self.sobj.results.loc[self.mid, stype] = np.nan

    def get_series(self, stype=None, sval=None, series=None):
        """
        searches for given stypes and svals in self.series and returns them

        Parameters
        ----------
            series: list(tuple)
                list of tuples to avoid problems with separate series and same sval
            stypes: list, str
                stype or stypes to be looked up
            svals: float
                sval or svals to be looked up

        Returns
        -------
            list
                list of series that fit the parameters
                if no parameters - > all series
                empty if none fit

        Note
        ----
            m = measurement with [<RockPy3.series> pressure, 0.00, [GPa], <RockPy3.series> temperature, 0.00, [C]]
            m.get_series(stype = 'pressure', sval = 0) -> [<RockPy3.series> pressure, 0.00, [GPa]]
            m.get_series(sval = 0) -> [<RockPy3.series> pressure, 0.00, [GPa], <RockPy3.series> temperature, 0.00, [C]]
            m.get_series(series=('temperature', 0)) -> [<RockPy3.series> pressure, 0.00, [GPa], <RockPy3.series> temperature, 0.00, [C]]
        """
        if not self._series:
            return None

        slist = self.series

        if stype is not None:
            stype = to_tuple(stype)
            slist = filter(lambda x: x[0] in stype, slist)
        if sval is not None:
            sval = to_tuple(sval)
            slist = filter(lambda x: x[1] in sval, slist)
        if series:
            series = tuple2list_of_tuples(series)
            slist = filter(lambda x: (x[0], x[1]) in series, slist)
        return list(slist)

    def equal_series(self, other, ignore_stypes=()):
        '''
        Checks if two measurement objects have the same series. 
        
        Parameters
        ----------
        other
        ignore_stypes: str, list
            list of stypes to be ignored

        Returns
        -------
            bool
        '''

        if not self.series and not other.series:
            return True

        ignore_stypes = to_tuple(ignore_stypes)
        ignore_stypes = [st.lower() for st in ignore_stypes if type(st) == str]
        selfseries = (s for s in self.series if not s[0] in ignore_stypes)
        otherseries = (s for s in other.series if not s[0] in ignore_stypes)

        if all(i in otherseries for i in selfseries):
            return True

        else:
            return False

    # todo test normalize functions
    ####################################################################################################################
    ''' Normalize functions '''

    def normalize(self,
                  reference='data', ref_dtype='mag', norm_dtypes='all', vval=None,
                  norm_method='max', norm_factor=None, result=None,
                  normalize_variable=False, dont_normalize=('temperature', 'field'),
                  norm_initial_state=True, **options):
        """
        normalizes all available data to reference value, using norm_method

        Parameter
        ---------
            reference: str
                reference state, to which to normalize to e.g. 'NRM'
                also possible to normalize to mass
            ref_dtype: str
                component of the reference, if applicable. standard - 'mag'
            norm_dtypes: list
                default_recipe: 'all'
                dtypes to be normalized, if dtype = 'all' all columns will be normalized
            vval: float
                variable value, if reference == value then it will search for the point closest to the vval
            norm_method: str
                how the norm_factor is generated, could be min
            normalize_variable: bool
                if True, variable is also normalized
                default_recipe: False
            result: str
                default_recipe: None
                normalizes the values in norm_dtypes to the result value.
                e.g. normalize the moment to ms (hysteresis measuremetns)
            dont_normalize: list
                list of dtypes that will not be normalized
                default_recipe: None
            norm_initial_state: bool
                if true, initial state values are normalized in the same manner as normal data
                default_recipe: True
        """

        if self.is_normalized and self.is_normalized['reference'] == reference:
            self.log.info('{} is already normalized with: {}'.format(self, self.is_normalized))
            return

        # dont normalize parameter measurements
        if isinstance(self, RockPy.Parameter):
            return
        # print(self.mtype, locals())
        # separate the calc from non calc parameters
        calculation_parameter, options = RockPy.core.utils.separate_calculation_parameter_from_kwargs(rpobj=self,
                                                                                                      **options)

        # getting normalization factor
        if not norm_factor:  # if norm_factor specified
            norm_factor = self._get_norm_factor(reference=reference, rtype=ref_dtype,
                                                vval=vval,
                                                norm_method=norm_method,
                                                result=result,
                                                **calculation_parameter)

        norm_dtypes = RockPy3._to_tuple(norm_dtypes)  # make sure its a list/tuple

        for dtype, dtype_data in self.data.items():  # cycling through all dtypes in data
            if dtype_data:
                if 'all' in norm_dtypes:  # if all, all non stype data will be normalized
                    norm_dtypes = [i for i in dtype_data.column_names if not 'stype' in i]

                ### DO not normalize:
                # variable
                if not normalize_variable:
                    variable = dtype_data.column_names[dtype_data.column_dict['variable'][0]]
                    norm_dtypes = [i for i in norm_dtypes if not i == variable]

                if dont_normalize:
                    dont_normalize = RockPy3._to_tuple(dont_normalize)
                    norm_dtypes = [i for i in norm_dtypes if not i in dont_normalize]

                for ntype in norm_dtypes:  # else use norm_dtypes specified
                    try:
                        dtype_data[ntype] = dtype_data[ntype].v / norm_factor
                    except KeyError:
                        self.log.warning(
                                'CAN\'T normalize << %s, %s >> to %s' % (self.sobj.name, self.mtype, ntype))

                if 'mag' in dtype_data.column_names:
                    try:
                        self.data[dtype]['mag'] = self.data[dtype].magnitude(('x', 'y', 'z'))
                    except KeyError:
                        self.log.debug('no (x,y,z) data found in {} keeping << mag >>'.format(dtype))

        self.log.debug('NORMALIZING << %s >> with << %.2e >>' % (', '.join(norm_dtypes), norm_factor))

        if self.initial_state and norm_initial_state:
            for dtype, dtype_rpd in self.initial_state.data.items():
                self.initial_state.data[dtype] = dtype_rpd / norm_factor
                if 'mag' in self.initial_state.data[dtype].column_names:
                    self.initial_state.data[dtype]['mag'] = self.initial_state.data[dtype].magnitude(('x', 'y', 'z'))

        if reference == 'mass':
            self.calc_all(force_recalc=True, **self.calculation_parameter)

        self.is_normalized = {'reference': reference, 'ref_dtype': ref_dtype,
                              'norm_dtypes': norm_dtypes, 'vval': vval,
                              'norm_method': norm_method, 'norm_factor': norm_factor, 'result': result,
                              'normalize_variable': normalize_variable, 'dont_normalize': dont_normalize,
                              'norm_initial_state': norm_initial_state}
        return self

    def _get_norm_factor(self, reference, rtype, vval, norm_method, result, **calculation_parameter):
        """
        Calculates the normalization factor from the data according to specified input

        Parameter
        ---------
           reference: str
              the type of data to be referenced. e.g. 'NRM' -> norm_factor will be calculated from self.data['NRM']
              if not given, will return 1
           rtype:
           vval:
           norm_method:

        Returns
        -------
           normalization factor: float
        """
        norm_factor = 1  # inititalize
        # print('measurement:', locals())
        if reference and not result:
            if reference == 'nrm' and reference not in self.data and 'data' in self.data:
                reference = 'data'

            if reference in self.data:
                norm_factor = self._norm_method(norm_method, vval, rtype, self.data[reference])

            if reference in ['is', 'initial', 'initial_state']:
                if self.initial_state:
                    norm_factor = self._norm_method(norm_method, vval, rtype, self.initial_state.data['data'])
                if self.is_initial_state:
                    norm_factor = self._norm_method(norm_method, vval, rtype, self.data['data'])

            if reference == 'mass':
                m = self.get_mtype_prior_to(mtype='mass')
                if not m:
                    self.log.error('CANT find mass measurement')
                    return 1
                return m.data['data']['mass'].v[0]

            if isinstance(reference, float) or isinstance(reference, int):
                norm_factor = float(reference)

        elif result:
            norm_factor = getattr(self, 'result_' + result)(**calculation_parameter)[0]
        else:
            self.log.warning('NO reference specified, do not know what to normalize to.')
        return norm_factor

    def _norm_method(self, norm_method, vval, rtype, data):
        methods = {'max': max,
                   'min': min,
                   }

        if not vval:
            if not norm_method in methods:
                raise NotImplemented('NORMALIZATION METHOD << %s >>' % norm_method)
                return
            else:
                return methods[norm_method](data[rtype].v)

        if vval:
            idx = np.argmin(abs(data['variable'].v - vval))
            out = data.filter_idx([idx])[rtype].v[0]
            return out

    def get_mtype_prior_to(self, mtype):
        """
        This method allows to search for a measurememt with a specific mtype, that was added prior to this measurement.

        Parameters
        ----------
           mtype: str
              the type of measurement that is supposed to be returned

        Returns
        -------
           RockPy3.Measurement
        """

        # measurements = self.sobj.get_measurement(mtype=mtype)
        #
        # if measurements:
        #     out = [i for i in measurements if i.__idx <= self.__idx]
        #     return out[-1]
        #
        # else:
        return None

    def set_calibration_measurement(self,
                                    fpath=None,  # file path
                                    mdata=None,
                                    mobj=None,  # for special import of a measurement instance
                                    ):
        """
        creates a new measurement that can be used as a calibration for self. The measurement has to be of the same
        mtype and has to have the same ftype

        Parameters
        ----------
        fpath: str
            the full path and filename where the file is located on the hard disk
        mdata: RockPyData
        mobj: RockPy3.Measurement
        """

        raise NotImplementedError

    ##################################################################################################################
    ''' REPORT '''

    ''' HELPER '''

    @staticmethod
    def get_values_in_both(a, b, key='level'):  # todo TEST
        '''
        Looks through pd.DataFrame(a)[key] and pd.DataFrame(b)[key] to find values in both

        Parameters
        ----------
        a: pd.DataFrame
        b: pd.DataFrame
        key: str

        Returns
        -------
            sorted(list) of items
        '''

        # get equal temperature steps for both demagnetization and acquisition measurements
        equal_vals = list(set(a[key].values) & set(b[key].values))
        return sorted(equal_vals)

if __name__ == '__main__':
    # RockPy.convertlog.setLevel(logging.WARNING)
    s = RockPy.Sample('test')
    m = s.add_simulation(mtype='paleointensity', series=[('test',1,'abc'),('test2',2,'abc')])
    print(m)
    # m.calc_all()
    # print(m.results)