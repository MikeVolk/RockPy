import logging
import os
import shutil
from collections import OrderedDict
from copy import deepcopy

import RockPy
import RockPy.core
import numpy as np
import pandas as pd


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

    mcolumns = ['sID', 'mID']

    _clsdata = pd.DataFrame(columns=mcolumns)  # raw data do not manipulate

    clsdata = pd.DataFrame()

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
        return logging.getLogger('RockPy.%s' % cls.mtype())

    @classmethod
    def mtype(cls):
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

    @classmethod
    def implemented_ftypes(cls):  # todo move into RockPy core.core has nothing to do with measurement
        """
        Dictionary of all implemented filetypes.

        Looks for all subclasses of RockPy3.io.base.ftype
        generating a dictionary of implemented machines : {implemented out_* method : machine_class}

        Returns
        -------

        dict: classname:
        """
        implemented_ftypes = {cl.__name__.lower(): cl for cl in RockPy.core.file_io.ftype.__subclasses__()}
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
    def calculate_methods(cls):
        '''
        This is a dictionary with all calculate methods of the class. The design is:
            method_name:calculate_method

        Returns
        -------
        dict
            Dictionary with all calculate methods.
        '''

        return {i.replace('calculate_', ''): getattr(cls, i) for i in dir(cls)
                if i.startswith('calculate_')
                if not i.endswith('generic')
                if not i.endswith('result')
                if not i.endswith('recipe')
                if not i.endswith('methods')
                }

    @classmethod
    def result_methods(cls):
        """
        Searches through all :code:`result_*` methods and creates a dictionary with:

            result_name : result_method

        where result_name is the name without the result_ prefix
        """
        return {i[7:]: getattr(cls, i) for i in dir(cls) if i.startswith('result_')
                if not i.endswith('generic')
                if not i.endswith('methods')
                if not i.endswith('recipe')
                if not i.endswith('category')
                }

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
                  series=None,
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

        # check if measurement is implemented
        if ftype in cls.implemented_ftypes():
            # read the ftype data from the file
            ftype_data = cls.implemented_ftypes()[ftype](fpath, sobj.name)
        else:
            log.error('CANNOT IMPORT ')

        # check wether the formatter for the ftype is implemented
        if ftype in cls.ftype_formatters():
            log.debug('ftype_formatter << %s >> implemented' % ftype)
            mdata = cls.ftype_formatters()[ftype](ftype_data, sobj_name=sobj.name)
            if not mdata:
                log.debug('mdata is empty -- measurement may not be created')
        else:
            log.error('UNKNOWN ftype: << %s >>' % ftype)
            log.error('most likely cause is the \"format_%s\" method is missing in the measurement << %s >>' % (
                ftype, cls.__name__))
            return

        return cls(sobj=sobj, fpath=fpath, ftype=ftype, mdata=mdata, series=series, idx=idx, **options)

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
            mdata: dict
                when mdata is set, this will be directly used as measurement data without formatting from file

        Note
        ----
            when creating a new measurement it automatically calculates all results using the standard prameter set
        """

        self.id = id(self)

        self.sobj = sobj
        self.log().debug('Creating measurement: id:{} idx:{}'.format(self.id, self._idx))

        # create the dictionary the data will be stored in
        if mdata is None:
            mdata = OrderedDict()  # create an ordered dict, so that several measurement 'branches' can be in one measurement (e.g. a heating and cooling run for a thermocurve)

        # the data that is used for calculations and corrections
        self._data = mdata

        # _raw_data is a backup deepcopy of _data it will be used to reset the _data if reset_data() is called
        self._raw_data = deepcopy(mdata)

        # flags for mean and the base measurements
        self.is_mean = options.get('ismean', False)  # flag for mean measurements
        self.base_measurements = options.get('base_measurements',
                                             False)  # list with all measurements used to generate the mean

        self.ftype = ftype
        self.fpath = fpath

        ''' initial state '''
        self.is_initial_state = False
        self.initial_state = options.get('initial_state', False)

        ''' calibration, correction and holder'''
        self.calibration = None
        self.holder = None
        self._correction = []

        # initialize the calculation_parameter dictionary  and results data
        self.results = pd.DataFrame(columns=self.result_methods().keys())

        self.calculation_parameter = {res: {} for res in self.result_methods()}

        # self.__initialize()

        # normalization
        self.is_normalized = False  # normalized flag for visuals, so its not normalized twice
        self.norm = None  # the actual parameters
        self.norm_factor = 1

        ''' series '''
        self._series = []

        # add series if provided
        if series:
            self.add_series(*series)

        self.idx = idx if idx else self._idx  # external index e.g. 3rd hys measurement of sample 1

        self.__class__.n_created += 1

    def reset_calculation_params(self):
        """
        resets the calculation parameters so you can force recalc
        Returns
        -------

        """
        self.calculation_parameter = {result: {} for result in self.result_methods()}

    @property
    def _idx(self):
        return 0

    @property
    def mass(self):
        mass = self.get_mtype_prior_to(mtype='mass')
        return mass.data['data']['mass'].v[0] if mass else None

    def get_RockPy_compatible_filename(self, add_series=True):

        # if add_series:
        #     series = sorted([s.get_tuple() for s in self.series if not s.get_tuple() == ('none', np.nan, '')])
        # else:
        #     series = None
        #
        # # diameter = self.get_mtype_prior_to(mtype='diameter') #todo implement
        # # height = self.get_mtype_prior_to(mtype='height')
        #
        # # convert the mass to the smallest possible exponent
        # mass, mass_unit = None, None
        # if self.mass:
        #     mass, mass_unit = RockPy3.utils.convert.get_unit_prefix(self.mass, 'kg')
        #
        # minfo_obj = RockPy3.core.file_operations.minfo(fpath=self.fpath,
        #                                                sgroups=self.sobj.samplegroups,
        #                                                samples=self.sobj.name,
        #                                                mtypes=self.mtype, ftype=self.ftype,
        #                                                mass=mass, massunit=mass_unit,
        #                                                series=series,
        #                                                suffix=self.idx,
        #                                                read_fpath=False)
        # return minfo_obj.fname
        raise NotImplementedError

    def _rename_to_RockPy_compatible_filename(self, add_series=True, create_backup=True):
        if self.fpath:
            path = os.path.dirname(self.fpath)
            backup_name = '#' + os.path.basename(self.fpath)
            fname = self.get_RockPy_compatible_filename(add_series=add_series)
            if create_backup:
                shutil.copy(self.fpath, os.path.join(path, backup_name))
            os.rename(self.fpath, os.path.join(path, fname))

    def set_recipe(self, result, recipe):
        """
        changes the recipe for a result to a new value. Changes the standard parameter dictionary to the new values

        Parameter
        ---------
            result:
            recipe:

        Note:
            if the result is indirect (e.g. hf_sus is calculated through ms) the result will overwrite the method for
            both

        """
        # break if the result does not exist
        if not result in self.result_recipe():
            self.log.error('Measurement << %s >> has no result << %s >>' % (self.mtype, result))
            return

        # for dependent results, the recipe has to be set for the method the result is dependent on
        if self.res_signature()[result].get('indirect', False):
            result = self.res_signature()[result]['signature']['dependent'][0]

        # break if the recipe is already set
        if self.selected_recipe[result].upper() == recipe.upper():
            self.log.info('RECIPE << %s, %s >> already set' % (result, recipe))
            return

        # break if recipe not implemented
        if not recipe in self.get_recipes(result):
            self.log.error('Recipe %s not found in %s, these are implemented: %s' % (
                recipe, result, self.get_recipes(result)))
            return

        # setting the recipe
        old_recipe = self.result_recipe()[result]
        self.selected_recipe[result] = recipe.upper()

        # if result is the base for other results, their methods have to be changed, too.
        if self.res_signature()[result]['subjects']:
            for dep_res in self.res_signature()[result]['subjects']:
                self.set_recipe(dep_res, recipe=recipe)
                # self.selected_recipe[dep_res] = recipe.upper()

        self.log.warning('Calculation parameter changed from:')
        self.log.warning('{}: {}'.format(old_recipe, self.calculation_parameter[result]))
        self.calculation_parameter[result] = {}

        # change the method that is called
        if recipe == 'default':
            recipe = ''

        self.cmethods[result] = getattr(self, '_'.join(['calculate', self.get_cmethod_name(result)]))

        self.calculate_result(result=result)
        self.log.warning('{}: {}'.format(self.result_recipe()[result], self.calculation_parameter[result]))

    def get_recipes(self, res):
        """
        returns all result recipes for a given result
        :param res:
        :return:
        """
        recipes = ['default'] + [r.split('_')[-1].lower() for r in self.calc_signature() if
                                 res in r and r.split('_')[-1].isupper()]
        return set(recipes)

    def get_cmethod_name(self, res):
        """
        Takes a result, looks up the defined method and returns the calculation_method_name

        Parameter
        ---------
            result: str
                The result name

        Returns
        -------
            returns the calculate_method name.
            For direct methods without a recipe this will be: resultname
            For direct methods with recipe this will be: resultname_RECIPE
            For indirect methods this will be: calculationmethod
            For indirect methods with a recipe this will be: calculationmethod

        """
        # check for indirect methods
        if self.res_signature()[res]['dependent']:
            # the result is truly indirect if there is no calculate method with its name in it
            # it needs to be specified what method is used to calculate (dependent)
            if not res in self.calculate_methods():
                res = self.res_signature()[res]['signature']['dependent'][0]

        # get the recipe
        recipe = self.selected_recipe[res]
        # add the suffix for non-default methods
        if recipe != 'DEFAULT':
            method_name = '_'.join([res, recipe.upper()])
        # default recipes do not need a suffix
        else:
            method_name = res
        return method_name

    def get_cmethod(self, res):
        cmethod_name = self.get_cmethod_name(res)
        return self.calculate_methods()[cmethod_name]

    def get_dependent_results(self, result):
        """
        gets all result names that are equally independent and the independent result they are based on

        Example
        -------
            hf_sus(hysteresis) is always dependent on the calculation of ms.
                -> returns [ms, hf_sus]
            mrs_ms(hysteresis) is always dependent on ms & mrs and dependent on each of these
                -> returns [ms, mrs, mrs_ms, hf_sus]
        :param result:
        :return:
        """

        # if not self.res_signature()[result]['indirect'] and not self.res_signature()[result]['dependent']:
        #     return [result]

        if 'calculation_method' in self.res_signature()[result]['signature']:
            cm = self.res_signature()[result]['signature']['calculation_method']
        else:
            cm = result

        indirect_methods = [res for res in self.res_signature() if self.res_signature()[res]['dependent']]
        dependent_on_cm = [res for res in indirect_methods if
                           self.res_signature()[res]['signature']['dependent'][0] == cm]

        if self.res_signature()[result]['dependent']:
            for res in self.res_signature()[result]['signature']['dependent']:
                dependent_on_cm.extend(self.get_dependent_results(res))

        return sorted(set([cm] + dependent_on_cm))

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
            return int(self._idx) < int(other._idx)
        except ValueError:
            pass

    def __repr__(self):
        if self.is_mean:
            add = 'mean_'
        else:
            add = ''
        return '<<RockPy3.{}.{}{}{} at {}>>'.format(self.sobj.name, add, self.mtype(), '',
                                                    # self.stype_sval_tuples, #todo fix
                                                    hex(self.id))

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

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

    def append_to_clsdata(self, pdDF):
        '''
        Method that adds data to the clsdata and _clsdata of the class

        Parameters
        ----------
        pdDF: pandas.Dataframe

        Returns
        -------

        '''
        ids = pd.DataFrame(data=np.ones((pdDF.size, 2), dtype=np.int64) * (self.sobj.id, self.id),
                           columns=['sID', 'mID'])
        d = pd.concat([ids, pdDF], axis=1)

        # append data to raw data (_clsdata)
        self.__class__._clsdata = pd.concat([self.__class__._clsdata, d])
        # append data to manipulate data (clsdata)
        self.__class__.clsdata = pd.concat([self.__class__.clsdata, d])

    @property
    def stype_sval_tuples(self):
        if self.get_series():
            return [(s.stype, s.value) for s in self.series]
        else:
            return []

    @property
    def m_idx(self):
        return self.sobj.measurements.index(self)  # todo change could be problematic if changing the sobj

    @property
    def fname(self):
        """
        Returns only filename from self.file

        Returns
        -------
           str: filename from full path
        """
        return os.path.basename(self.fpath)

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
        return [s.stype for s in self.series]

    @property
    def svals(self):
        """
        set of all svalues
        """
        return [s.sval for s in self.series]

    @property
    def data(self):
        try:
            return self.__class__.clsdata[self.__class__.clsdata['mID'] == self.id]
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
        return self.set_get_attr('_correction', value=list())

    def reset_data(self):
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
        idx = self._get_idx_dtype_var_val(dtype=dtype, var=var, val=val)
        self.data[dtype] = self.data[dtype].filter_idx(idx, invert=True)
        return self

    def has_result(self, result):
        """
        Checks if the measurement contains a certain result

        Parameters
        ----------
           result: str
              the result that should be found e.g. result='ms' would give True for 'hys' and 'backfield'
        Returns
        -------
           out: bool
              True if it has result, False if not
        """
        if result in self.result_methods():
            return True
        else:
            return False

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

    def add_series(self, stype, sval, sunit=None):  # todo add (stype,sval,sunit) type calling
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

        Note
        ----
            If the measurement previously had no series, the (none, 0 , none) standard series will be removed first
        """

        if all(i for i in [stype, sval, sunit]):
            series = (stype, sval, sunit)
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
        sobj = self.get_series(stype=stype)[0]
        # remove series from _series list
        self._series.remove(sobj)

    def equal_series(self, other, ignore_stypes=()):
        ignore_stypes = RockPy3._to_tuple(ignore_stypes)
        ignore_stypes = [st.lower() for st in ignore_stypes if type(st) == str]
        selfseries = [s for s in self.series if not s.stype in ignore_stypes]
        otherseries = [s for s in other.series if not s.stype in ignore_stypes]

        if all(i in otherseries for i in selfseries):
            return True
        if not self.series and not other.series:
            return True
        else:
            return False

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
                default: 'all'
                dtypes to be normalized, if dtype = 'all' all columns will be normalized
            vval: float
                variable value, if reference == value then it will search for the point closest to the vval
            norm_method: str
                how the norm_factor is generated, could be min
            normalize_variable: bool
                if True, variable is also normalized
                default: False
            result: str
                default: None
                normalizes the values in norm_dtypes to the result value.
                e.g. normalize the moment to ms (hysteresis measuremetns)
            dont_normalize: list
                list of dtypes that will not be normalized
                default: None
            norm_initial_state: bool
                if true, initial state values are normalized in the same manner as normal data
                default: True
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

        measurements = self.sobj.get_measurement(mtype=mtype)

        if measurements:
            out = [i for i in measurements if i._idx <= self._idx]
            return out[-1]

        else:
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


if __name__ == '__main__':
    # RockPy.convertlog.setLevel(logging.WARNING)
    m = Measurement(sobj='test')
