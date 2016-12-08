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
    n_created = 0

    possible_plt_props = ['agg_filter', 'alpha', 'animated', 'antialiased', 'axes', 'clip_box', 'clip_on', 'clip_path',
                          'color', 'contains', 'dash_capstyle', 'dash_joinstyle', 'dashes', 'drawstyle', 'figure',
                          'fillstyle', 'gid', 'label', 'linestyle', 'linewidth', 'lod', 'marker', 'markeredgecolor',
                          'markeredgewidth', 'markerfacecolor', 'markerfacecoloralt', 'markersize', 'markevery',
                          'path_effects', 'picker', 'pickradius', 'rasterized', 'sketch_params', 'snap',
                          'solid_capstyle', 'solid_joinstyle', 'transform', 'url', 'visible', 'xdata', 'ydata',
                          'zorder']

    _mcp = None  # mtype calculation parameter cache for all measurements implemented as a dict(measurement:method:parameter)

    _rm = None  # result methods #needed 1.3.16
    _cm = None  # calculation methods

    _sresult = None

    _scp = None  # standard_calculation parameter
    _scalculate = None

    _cmtype_params = None  # measurement parameter collection

    @property
    def log(self):
        return RockPy3.core.utils.set_get_attr(self, '_log',
                                               value=logging.getLogger(
                                                   'RockPy3.[%s].[%i]%s' % (self.sobj.name, self._idx, self.mtype)))

    @classmethod
    def implemented_ftypes(cls):
        """
        setting implemented machines
        looking for all subclasses of RockPy3.io.base.Machine
        generating a dictionary of implemented machines : {implemented out_* method : machine_class}

        Returns
        -------

        dict: classname:
        """

        # implemented_ftypes = {cl.__name__.lower(): cl for cl in RockPy.core.io.ftype.__subclasses__()}
        # return implemented_ftypes
        raise NotImplementedError

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

    @classmethod
    def measurement_formatters(cls):
        # measurement formatters are important!
        # if they are not inside the measurement class, the measurement has not been implemented for this machine.
        # the following machine formatters:
        # 1. looks through all implemented measurements
        # 2. for each measurement stores the machine and the applicable readin class in a dictionary
        measurement_formatters = {
            i.replace('format_', '').lower(): getattr(cls, i) for i in dir(cls) if i.startswith('format_')
            }
        return measurement_formatters

    ''' builtin methods '''

    @classmethod
    def result_methods(cls):
        """
        Searches through all :code:`result_*` methods and creates a dictionary with:

            result_name : result_method

        where result_name is the name without the result_ prefix
        """
        if not cls._rm:
            cls._rm = {i[7:]: getattr(cls, i) for i in dir(cls) if i.startswith('result_')
                       if not i.endswith('generic')
                       if not i.endswith('methods')
                       if not i.endswith('recipe')
                       if not i.endswith('category')
                       }
        return cls._rm

    @classmethod
    def calculate_methods(cls):
        # dynamically generating the calculation and standard parameters for each calculation method.
        # This just sets the values to non, the values have to be specified in the class itself
        calculate_methods = {i.replace('calculate_', ''): getattr(cls, i) for i in dir(cls)
                             if i.startswith('calculate_')
                             if not i.endswith('generic')
                             if not i.endswith('result')
                             if not i.endswith('recipe')
                             if not i.endswith('methods')
                             }
        return calculate_methods

    @classmethod
    def correct_methods(cls):
        """
        Dynamically searches through the class and finds all correction methods

        """
        methods = {i.replace('correct_', ''): getattr(cls, i) for i in dir(cls)
                   if i.startswith('correct_')
                   }
        # for name, method in methods.items():
        # setattr(RockPy3.Measurement.correct_methods, '__doc__', 'test')
        return methods

    ''' plotting / legend properties '''

    # todo think of a better way of creating plt_options
    @property
    def plt_props(self):
        return self._plt_props

    def set_plt_prop(self, prop, value):
        """
        sets the plt_props for the measurement.

        raises
        ------
            KeyError if the plt_prop not in the matplotlib.lines.Line2D
        """
        if value is None:
            return
        if prop not in Measurement.possible_plt_props:
            raise KeyError
        old_prop = self._plt_props[prop] if prop in self._plt_props else None

        self._plt_props[prop] = value
        self.log.debug('SETTING {:10}: {} -> {}'.format(prop, old_prop, self._plt_props[prop]))

    ####################################################################################################################

    """ measurement creation through function """

    @classmethod
    def from_mdata(cls):
        pass

    @classmethod
    def from_file(cls, sobj,
                  fpath=None, ftype='generic',  # file path and file type
                  idx=None, sample_name=None,
                  # for special import of pure data (needs to be formatted as specified in data of measurement class)
                  series=None,
                  **options
                  ):

        if ftype in cls.implemented_ftypes():
            ftype_data = cls.implemented_ftypes()[ftype](fpath, sobj.name)
        else:
            cls.clslog.error('CANNOT IMPORT ')

        if ftype in cls.measurement_formatters():
            cls.clslog.debug('ftype_formatter << %s >> implemented' % ftype)
            mdata = cls.measurement_formatters()[ftype](ftype_data, sobj_name=sobj.name)
            if not mdata:
                return
        else:
            cls.clslog.error('UNKNOWN ftype: << %s >>' % ftype)
            cls.clslog.error(
                'most likely cause is the \"format_%s\" method is missing in the measurement << %s >>' % (
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
    def from_measurements_create_mean(cls, sobj, mlist,
                                      interpolate=False, recalc_mag=False,
                                      substfunc='mean', ignore_stypes=False,
                                      color=None, marker=None, linestyle=None):
        """
        Creates a new measurement from a list of measurements

        Parameters
        ----------
        sobj
        mlist
        interpolate
        recalc_mag
        substfunc
        ignore_stypes
        color
        marker
        linestyle
        """

        # return cls(sobj=sobj, ftype='from_measurements_create_mean', mdata=mdata,
        #            initial_state=initial, series=series, ismean=True, base_measurements=mlist,
        #            color=color, marker=marker, linestyle=linestyle)

        raise NotImplementedError

    @classmethod
    def from_measurement(cls):
        """
        creates a measurement from a different type

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
        creates a new measurement (ISM) as initial state of base measurement (BSM).
        It dynamically calls the measurement _init_ function and assigns the created measurement to the
        self.initial_state value. It also sets a flag for the ISM to check if a measurement is a MIS.

        if a measurement object is passed, the initial state will be created from _measurements.
        Parameters
        ----------
           fpath
           ftype
           mobj
           series
           mtype: str
              measurement type
           mfile: str
              measurement data file
           machine: str
              measurement machine
            mobj: RockPy3.MEasurement object
        """

        with RockPy3.ignored(AttributeError):
            mtype = mtype.lower()
            ftype = ftype.lower()

        self.log.info('CREATING << %s >> initial state measurement << %s >> data' % (mtype, self.mtype))

        # can only be created if the measurement is actually implemented
        if all([mtype, ftype, fpath]) or fpath or mobj:
            self.initial_state = self.sobj.add_measurement(
                mtype=mtype, ftype=ftype, fpath=fpath, series=series, mobj=mobj)
            self.initial_state.is_initial_state = True
            return self.initial_state
        else:
            self.log.error('UNABLE to find measurement << %s >>' % mtype)

    def __init__(self,
                 sobj,
                 fpath=None, ftype=None,
                 mdata=None,
                 series=None,
                 idx=None,
                 initial_state=None,
                 ismean=False, base_measurements=None,
                 color=None, marker=None, linestyle=None,
                 automatic_results=True,
                 filename=None,
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
            initial_state:
                RockPy3.Measurement obj

        Note
        ----
            when creating a new measurement it automatically calculates all results using the standard prameter set
        """
        self.id = id(self)

        self.sobj = sobj
        self.filename = filename
        # self.log = logging.getLogger('RockPy3.MEASURMENT.' + self.get_subclass_name())

        if mdata is None:
            mdata = OrderedDict()

        # the data that is used for calculations and corrections
        self._data = mdata

        # _raw_data is a backup deepcopy of _data it will be used to reset the _data if reset_data() is called
        self._raw_data = deepcopy(mdata)

        # coordinate system that is currently used in data; _raw_data is always assumed to be in core coordiantes
        self._actual_data_coord = 'core'

        self.is_mean = ismean  # flag for mean measurements
        self.base_measurements = base_measurements  # list with all measurements used to generate the mean

        self.ftype = ftype
        self.fpath = fpath

        ''' initial state '''
        self.is_initial_state = False
        self.initial_state = initial_state

        ''' calibration, correction and holder'''
        self.calibration = None
        self.holder = None
        self._correction = []

        # initialize the calculation_parameter dictionary  and results data
        self.results = RockPy3.Data(column_names=self.result_methods().keys(),
                                    data=[np.nan] * len(self.result_methods()))
        self.calculation_parameter = {res: {} for res in self.result_methods()}

        self.__initialize()

        # normalization
        self.is_normalized = False  # normalized flag for visuals, so its not normalized twice
        self.norm = None  # the actual parameters

        ''' series '''
        self._series = []

        # add series if provided
        if series:
            self.add_series(series=series)

        self.idx = idx if idx else self._idx  # external index e.g. 3rd hys measurement of sample 1

        self.__class__.n_created += 1

        self._plt_props = {'label': ''}
        self.set_standard_plt_props(color, marker, linestyle)

        if automatic_results:
            self.calc_all(force_recalc=True)

    def set_standard_plt_props(self, color=None, marker=None, linestyle=None):
        #### automatically set the plt_props for the measurement according to the
        if color:
            self.set_plt_prop('color', color)
        else:
            self.set_plt_prop(prop='color', value=RockPy3.colorscheme[self._idx])

        if marker or marker == '':
            self.set_plt_prop('marker', marker)
        else:
            self.set_plt_prop(prop='marker', value=RockPy3.marker[self.sobj.idx])

        if linestyle:
            self.set_plt_prop('linestyle', linestyle)
        else:
            self.set_plt_prop(prop='linestyle', value='-')

    def reset_plt_prop(self):
        """
        Resets the plt_props to the standard value
        """
        self.set_standard_plt_props()
        self.plt_props['label'] = ''
        for prop in self.plt_props:
            if prop not in ('marker', 'color', 'linestyle', 'label'):
                self.plt_props.pop(prop)

    def reset_calculation_params(self):
        """
        resets the calculation parameters so you can force recalc
        Returns
        -------

        """
        self.calculation_parameter = {result: {} for result in self.result_methods()}

    @property
    def _idx(self):
        for i, v in enumerate(self.sobj.measurements):
            if v == self:
                return i
        else:
            return len(self.sobj.measurements)

    @property
    def mass(self):
        mass = self.get_mtype_prior_to(mtype='mass')
        return mass.data['data']['mass'].v[0] if mass else None

    def get_RockPy_compatible_filename(self, add_series=True):

        if add_series:
            series = sorted([s.get_tuple() for s in self.series if not s.get_tuple() == ('none', np.nan, '')])
        else:
            series = None

        # diameter = self.get_mtype_prior_to(mtype='diameter') #todo implement
        # height = self.get_mtype_prior_to(mtype='height')

        # convert the mass to the smallest possible exponent
        mass, mass_unit = None, None
        if self.mass:
            mass, mass_unit = RockPy3.utils.convert.get_unit_prefix(self.mass, 'kg')

        minfo_obj = RockPy3.core.file_operations.minfo(fpath=self.fpath,
                                                       sgroups=self.sobj.samplegroups,
                                                       samples=self.sobj.name,
                                                       mtypes=self.mtype, ftype=self.ftype,
                                                       mass=mass, massunit=mass_unit,
                                                       series=series,
                                                       suffix=self.idx,
                                                       read_fpath=False)
        return minfo_obj.fname

    def _rename_to_RockPy_compatible_filename(self, add_series=True, create_backup=True):
        if self.fpath:
            path = os.path.dirname(self.fpath)
            backup_name = '#' + os.path.basename(self.fpath)
            fname = self.get_RockPy_compatible_filename(add_series=add_series)
            if create_backup:
                import shutil
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

    @property
    def mtype(self):
        return self._mtype()

    @property
    def base_ids(self):
        """
        returns a list of ids for all base measurements
        """
        return [m.id for m in self.base_measurements]

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
        return '<<RockPy3.{}.{}{}{} at {}>>'.format(self.sobj.name, add, self.mtype,
                                                    self.stype_sval_tuples,
                                                    hex(id(self)))

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

    def __getstate__(self):
        """
        returned dict will be pickled
        :return:
        """
        pickle_me = {k: v for k, v in self.__dict__.items() if k in
                     (
                         'id', '_idx', 'idx',
                         'ftype', 'fpath',
                         # plotting related
                         '_plt_props',
                         'calculation_parameter',
                         # data related
                         '_raw_data', '_data',
                         'initial_state', 'is_initial_state', 'is_normalized',
                         'is_mean', 'base_measurements',
                         'results',
                         # sample related
                         'sobj',
                         '_series',
                         'calibration', 'holder', 'correction',
                     )
                     }
        return pickle_me

    def __setstate__(self, d):
        """
        d is unpickled data
           d:
        :return:
        """
        self.__dict__.update(d)
        self.__initialize()

    def __initialize(self):
        """
        Initialize function is called inside the __init__ function, it is also called when the object is reconstructed
        with pickle.

        :return:
        """
        self.selected_recipe = deepcopy(self.result_recipe())
        self.cmethods = {result: getattr(self, 'calculate_' + self.get_cmethod_name(result)) for result in
                         self.result_recipe()}

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
        if self._data == {}:
            self._data = deepcopy(self._raw_data)

        # transform vectorial x,y,z data to new coordinate system when needed
        # self.transform_data_coord(final_coord=self.coord)
        # TODO: fails ->
        # print self.coord

        return self._data

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

    def calculate_result(self, result, **parameter):
        """
        Helper function to dynamically call a result. Used in Visualize

        Parameters
        ----------
           result:
           parameter:
        """

        if not self.has_result(result):
            self.log.warning('%s does not have result << %s >>' % self.mtype, result)
            return
        else:
            # self.log = logging.getLogger('RockPy3.MEASURMENT.' + self.mtype + '[%s]' % self.sobj.name)
            self.log.info('CALCULATING << %s >>' % result)
            out = getattr(self, 'result_' + result)(**parameter)
        return out

    def calc_all(self, force_recalc=False, **parameter):
        """
        Calculates all result methods in the measurement

        Parameters
        ----------
        force_recalc: bool
            Forces the recalculation of all resuts: e.g. when you change the data values by normalizing

        parameter

        Returns
        -------

        """
        if force_recalc:
            self.reset_calculation_params()

        # get possible calculation parameters and put them in a dictionary
        calculation_parameter, kwargs = RockPy3.core.utils.kwargs_to_calculation_parameter(rpobj=self, **parameter)

        for result_method in sorted(self.result_methods()):
            # get calculation parameter
            calc_param = calculation_parameter.get(self.mtype, {})
            # get the calculation method
            calc_param = calc_param.get(result_method, {})
            getattr(self, 'result_' + result_method)(**calc_param)

        if kwargs:
            self.log.warning('--------------------------------------------------')
            self.log.warning('| %46s |' % 'THESE PARAMETERS COULD NOT BE USED')
            self.log.warning('--------------------------------------------------')
            for i, v in kwargs.items():
                self.log.warning('| %22s: %22s |' % (i, v))
            self.log.warning('--------------------------------------------------')
        if calculation_parameter:
            self.log.info('--------------------------------------------------')
            self.log.info('| %46s |' % 'these parameters were used')
            self.log.info('--------------------------------------------------')
            for i, v in calculation_parameter.items():
                self.log.info('| %46s |' % i)
                for method, parameter in v.items():
                    self.log.info('| %22s: %22s |' % (method, parameter))
            self.log.info('--------------------------------------------------')

        return self.results

    def delete_dtype_var_val(self, dtype, var, val):
        """
        deletes step with var = var and val = val

           dtype: the step type to be deleted e.g. th
           var: the variable e.g. temperature
           val: the value of that step e.g. 500

        example: measurement.delete_step(step='th', var='temp', val=500) will delete the th step where the temperature is 500
        """
        idx = self._get_idx_dtype_var_val(dtype=dtype, var=var, val=val)
        self.data[dtype] = self.data[dtype].filter_idx(idx, invert=True)
        return self

    def check_parameters(self, caller, parameter):
        """
        Checks if previous calculation used the same parameters, if yes returns the previous calculation
        if no calculates with new parameters

        Parameters
        ----------
           caller: str
               name of calling function ('result_generic' should be given as 'generic')
           parameter:
        Returns
        -------
           bool
              returns true is parameters are not the same
        """
        if self.calculation_parameter[caller]:
            # parameter for new calculation
            a = []
            for key in self.calculation_parameter[caller]:
                if key in parameter:
                    a.append(parameter[key])
                else:
                    a.append(self.calculation_parameter[caller][key])
                    # a = [parameter[i] for i in self.calculation_parameter[caller]]
            # get parameter values used for calculation
            b = [self.calculation_parameter[caller][i] for i in self.calculation_parameter[caller]]
            if a != b:
                return True
            else:
                return False
        else:
            return True

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

    @classmethod
    def _get_variable_list(cls, rpdata_list, var='variable'):
        """
        takes a list of rpdata objects. it checks for all steps, the size of the step and min and max values of the
        variable. It then generates a list of new variables from the max(min) -> min(max) with the mean step size
        """
        cls.clslog.debug(
            'Creating new variable list for %s measurement out of %i measurements' % (
                cls.__name__, len(rpdata_list)))
        min_vars = []
        max_vars = []
        stepsizes = []
        for rp in rpdata_list:
            stepsizes.append(np.diff(rp[var].v))
            min_vars.append(min(rp[var].v))
            max_vars.append(max(rp[var].v))
        idx, steps = max(enumerate(stepsizes), key=lambda tup: len(tup[1]))
        new_variables = np.arange(max(min_vars), min(max_vars), np.mean(np.fabs(steps)))
        return sorted(set(new_variables))

    def combine_measurements(self, others, remove_others=False):
        others = RockPy3._to_tuple(others)
        self.log.info('COMBINING << {} >> with {}'.format(self, others))

        for m in others:
            m = deepcopy(m)

            # check they are the same type of measurement
            if not m.mtype == self.mtype:
                continue

            for dtype in m.data:
                if not dtype in self.data:
                    self.data[dtype] = m.data[dtype]
                self.data[dtype] = self.data[dtype].append_rows(m.data[dtype])

            # remove other measurements
            if remove_others:
                self.log.info('REMOVING << {} >> from sample << {} >>'.format(self, self.sobj))
                self.sobj.remove_measurement(mobj=m)
        return self

    ####################################################################################################################
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
            series = RockPy3.Series(stype='none', value=np.nan, unit='')
            return [series]

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
            stype = RockPy3._to_tuple(stype)
            if method == 'all':
                return True if all(i in self.stypes for i in stype) else False
            if method == 'any':
                return True if any(i in self.stypes for i in stype) else False
            if method == 'none':
                return True if not any(i in self.stypes for i in stype) else False
        else:
            return True if not self.stypes else False

    def has_sval(self, sval=None, method='any'):
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
            True if all stypes exist, False if not
            If stype = None : returns True if no series else True
        """
        if not self._series:
            return False

        if sval is not None:
            sval = RockPy3._to_tuple(sval)
            if method == 'all':
                return all(i in self.svals for i in sval)
            if method == 'any':
                return any(i in self.svals for i in sval)
            if method == 'none':
                return not any(i in self.svals for i in sval)
        else:
            return True if not self.svals else False

    def has_series(self, series=None, method='all'):

        if series is not None:
            series = RockPy3.core.utils.tuple2list_of_tuples(series)
            if method == 'all':
                return True if all(i in self.stype_sval_tuples for i in series) else False
            if method == 'any':
                return True if any(i in self.stype_sval_tuples for i in series) else False
            if method == 'none':
                return True if not any(i in self.svals for i in series) else False
        else:
            return True if not self.svals else False

    def get_series(self, stype=None, sval=None, series=None):
        """
        searches for given stypes and svals in self.series and returns them

        Parameters
        ----------
            series: list(tuple)
                list of tuples to avoid problems wit separate series and same sval
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
            stype = RockPy3.core.utils.to_list(stype)
            slist = filter(lambda x: x.stype in stype, slist)
        if sval is not None:
            sval = RockPy3.core.utils.to_list(sval)
            slist = filter(lambda x: x.sval in sval, slist)
        if series:
            series = RockPy3.core.utils.tuple2list_of_tuples(series)
            slist = filter(lambda x: (x.stype, x.sval) in series, slist)
        return list(slist)

    def get_sval(self, stype):
        """
        Searches for stype and returns sval
        """
        s = self.get_series(stype=stype)
        return s[0].value if s else None

    def add_series(self, stype=None, sval=0, unit='', series_obj=None, series=None):
        """
        adds a series to measurement.series, then adds is to the data and results datastructure

        Parameters
        ----------
           stype: str
              series type to be added
           sval: float or int
              series value to be added
           unit: str
              unit to be added. can be None #todo change so it uses Pint
            series_obj: RockPy3.series
                if a previously created object needs to be passed
            series: list(tuples)
                default: None
                Series object gets created for a list of specified series

        Returns
        -------
           [RockPy3.Series] list of RockPy series objects

        Note
        ----
            If the measurement previously had no series, the (none, 0 , none) standard series will be removed first
        """
        # if a series object is provided other wise create series object
        if not any(i for i in [stype, sval, unit, series_obj, series]):
            return

        elif series_obj:
            series = series_obj

        # if series provided of type ('stype', value, 'unit')
        elif series:
            series = RockPy3.core.utils.tuple2list_of_tuples(series)
            sobjs = [RockPy3.Series.from_tuple(series=stup) for stup in series]
        else:
            sobjs = [RockPy3.Series(stype=stype, value=sval, unit=unit)]

        sobjs = (sobj for sobj in sobjs if sobj not in self.series)

        for sinst in sobjs:
            if not any(sinst == s for s in self._series):
                self._series.append(sinst)
                self._add_sval_to_data(sinst)
                self._add_sval_to_results(sinst)

        return series

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
        self._remove_series_from_data(sobj)
        self._remove_series_from_results(sobj)

    def _add_sval_to_data(self, sobj):
        """
        Adds stype as a column and adds svals to data. Only if stype != none.

        Parameter
        ---------
           sobj: series instance
        """
        if sobj.stype != 'none':
            for dtype in self._raw_data:
                if self._raw_data[dtype]:
                    data = np.ones(len(self.data[dtype]['variable'].v)) * sobj.value
                    if not 'stype ' + sobj.stype in self.data[dtype].column_names:
                        self.data[dtype] = self.data[dtype].append_columns(column_names='stype ' + sobj.stype,
                                                                           data=data)  # , unit=sobj.unit) #todo add units

    def _add_sval_to_results(self, sobj):
        """
        Adds the stype as a column and the value as value to the results. Only if stype != none.

        Parameter
        ---------
           sobj: series instance
        """

        if sobj.stype != 'none':
            if not 'stype ' + sobj.stype in self.results.column_names:
                self.results = self.results.append_columns(
                    column_names='stype ' + sobj.stype, data=sobj.value)  # , unit=sobj.unit) #todo add units

    def _remove_series_from_data(self, sobj):
        # remove series from data
        for dtype in self.data:
            if self.data[dtype]:
                self.data[dtype].delete_columns(keys='stype ' + sobj.stype)

    def _remove_series_from_results(self, sobj):
        # remove series from results
        self.results.delete_columns(keys='stype ' + sobj.stype)

    def _get_idx_dtype_var_val(self, dtype, var, val):
        """
        returns the index of the closest value with the variable(var) and the step(step) to the value(val)

        """
        out = [np.argmin(abs(self.data[dtype][var].v - val))]
        return out

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
        if isinstance(self, RockPy3.Parameter):
            return
        # print(self.mtype, locals())
        # separate the calc from non calc parameters
        calculation_parameter, options = RockPy3.core.utils.separate_calculation_parameter_from_kwargs(rpobj=self,
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
                   # 'val': self.get_val_from_data,
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
        search for last mtype prior to self

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

    def _add_stype_to_results(self):
        """
        adds a column with stype stype.name to the results for each stype in measurement.series
        :return:
        """
        if self._series:
            for t in self.series:
                if t.stype:
                    if t.stype not in self.results.column_names:
                        self.results.append_columns(column_names='stype ' + t.stype,
                                                    data=t.value,
                                                    # unit = t.unit      # todo add units
                                                    )

    ####################################################################################################################
    '''  CORRECTIONS '''

    def correct_dtype(self, dtype='th', var='variable', val='last', initial_state=True):
        """
        corrects the remaining moment from the last th_step

           dtype:
           var:
           val:
           initial_state: also corrects the initial state if one exists
        """

        try:
            calc_data = self.data[dtype]
        except KeyError:
            self.log.error('REFERENCE << %s >> can not be found ' % (dtype))

        if val == 'last':
            val = calc_data[var].v[-1]
        if val == 'first':
            val = calc_data[var].v[0]

        idx = self._get_idx_dtype_var_val(step=dtype, var=var, val=val)

        correction = self.data[dtype].filter_idx(idx)  # correction step

        for dtype in self.data:
            # calculate correction
            self._data[dtype]['m'] = self._data[dtype]['m'].v - correction['m'].v
            # recalc mag for safety
            self.data[dtype]['mag'] = self.data[dtype].magnitude(('x', 'y', 'z'))
        self.reset__data()

        if self.initial_state and initial_state:
            for dtype in self.initial_state.data:
                self.initial_state.data[dtype]['m'] = self.initial_state.data[dtype]['m'].v - correction['m'].v
                self.initial_state.data[dtype]['mag'] = self.initial_state.data[dtype].magnitude(('x', 'y', 'z'))
        return self

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

        cal = self.sobj.add_measurement(mtype=self.mtype, ftype=self.ftype, fpath=fpath,
                                        mobj=mobj, mdata=mdata,
                                        create_only=True)
        self.calibration = cal

    ####################################################################################################################
    '''' PLOTTING '''''

    def get_series_labels(self, stype=None, add_stype=True, add_sval=True, add_unit=True):
        """
        takes a list of stypes or stypes = True and returns a string with stype_sval_sunit; for each stype

        Parameters
        ----------
            stype: list / bool
                default: True
                if True all series will be returned
                if a list of strings is provided the ones where a matching stype can be found are appended
            add_stype: bool
                default: True
                if True: stype is first part of label
                if False: stype not in label
            add_unit: bool
                default: True
                if True: unit is last part of label
                if False: unit not in label
        """
        out = []

        if stype is None:
            stype = list(self.stypes)

        stype = RockPy3._to_tuple(stype)
        for st in stype:
            if not st or st == 'none':
                continue
            sobj = self.get_series(stype=st)
            if sobj:
                sobj = sobj[0]
                aux = []
                if add_stype:
                    aux.append(sobj.stype)
                if add_sval:
                    aux.append(str(np.round(sobj.value, 2)))
                if add_unit:
                    aux.append(sobj.unit)
                stype_label = ' '.join(aux)
                if not stype_label in out:
                    out.append(stype_label)
            else:
                self.log.warning('CANT find series << %s >>' % st)
                self.log.warning('\tonly one of these are possible:\t%s' % self.stypes)
        return '; '.join(out)

    def label_add_sname(self):
        """
        adds the corresponding sample_name to the measurement label
        """
        if isinstance(self.sobj, RockPy3.MeanSample):
            mean = 'mean '
        else:
            mean = ''
        text = ' '.join([mean, self.sobj.name])
        self.label_add_text(text)

    def label_add_stype(self, stype=None, add_stype=True, add_sval=True, add_unit=True):
        """
        adds the corresponding sample_name to the measurement label
        """
        text = self.get_series_labels(stype=stype, add_stype=add_stype, add_sval=add_sval, add_unit=add_unit)
        self.label_add_text(text)

    def label_add_text(self, text):
        if not text in self.plt_props['label']:
            self.plt_props['label'] = ' '.join([self.plt_props['label'], text])

    def set_get_attr(self, attr, value=None):
        """
        checks if attribute exists, if not, creates attribute with value None
           attr:
        :return:
        """
        if not hasattr(self, attr):
            setattr(self, attr, value)
        return getattr(self, attr)

    def series_to_color(self, stype: str, reverse: bool = False) -> None:
        """
        Sets the color of the measurement to the corresponding value of the series with specified stype

        Parameters
        ----------
        stype: str
            the stype to be set
        reverse: bool
            default: False
            if True the min(sval) will be set to the max(color)
            else: min(sval) -> min(color)
        """
        # get all possible svals in the hierarchy

        svals = sorted(self.sobj.svals)

        # create colormap from svals
        color_map = RockPy3.core.utils.create_heat_color_map(value_list=svals,
                                                             reverse=reverse)  # todo implement matplotlib.cmap

        # get the index and set the color
        sval = self.get_series(stype=stype)[0].value
        sval_index = svals.index(sval)
        self.set_plt_prop('color', color_map[sval_index])

    def plot(self, save_path=None, return_figure=False, **plt_props):
        """
        Quick plot depending on the mtype._visual specified in the measurement class
        Parameters
        ----------
        plt_props: dict
            additional plot properties
        """
        if self._visuals:
            fig = RockPy3.Figure(title='{}'.format(self.sobj.name))
            self.add_visuals(fig, **plt_props)
            f = fig.show(save_path=save_path, return_figure=return_figure)
            return f

    def add_visuals(self, fig, **plt_props):
        for v in self._visuals:
            visual = v[0]
            vprops = v[1]
            vprops.update(plt_props)
            v = fig.add_visual(data=self, visual=visual, **vprops)
            if self.is_mean:
                feature = [f for f in vprops['features'] if 'data' in f]
                for f in feature:
                    fprops = {k: v for k, v in vprops.items() if not k == 'features'}
                    fprops['alpha'] = 0.5
                    v.add_feature(feature=f, data=self.base_measurements, **fprops)
        return fig

    ####################################################################################################################
    ''' REPORT '''

    # todo report sheet

    ####################################################################################################################
    ''' XML io'''

    @property
    def etree(self):
        """
        Returns the content of the measurement as an xml.etree.ElementTree object which can be used to construct xml
        representation

        Returns
        -------
             etree: xml.etree.ElementTree
        """

        measurement_node = etree.Element(type(self).MEASUREMENT, attrib={'id': str(self.id), 'mtype': str(self.mtype),
                                                                         'is_mean': str(self.is_mean)})

        # store _data dictionary
        for name, data in self._data.items():
            de = etree.SubElement(measurement_node, type(self).DATA, attrib={type(self).NAME: name})
            if data is not None:
                de.append(data.etree)

        # store _raw_data dictionary
        for name, data in self._raw_data.items():
            de = etree.SubElement(measurement_node, type(self).RAW_DATA, attrib={type(self).NAME: name})
            if data is not None:
                det = data.etree
                de.append(det)

        if self.is_mean:
            # store ids of base measurements
            base_measurements_node = etree.SubElement(measurement_node, type(self).BASE_MEASUREMENTS, attrib={})
            for bid in self.base_ids:
                etree.SubElement(base_measurements_node, type(self).BID, attrib={}).text = str(bid)

        return measurement_node

    @classmethod
    def from_etree(cls, et_element, sobj):
        """

        :param et_element: ElementTree.Element containing the xml data
        :param sobj: sample object to which this measurement will belong
        :return:
        """
        if et_element.tag != cls.MEASUREMENT:
            cls.clslog.error('XML Import: Need {} node to construct object.'.format(cls.MEASUREMENT))
            return None

        # readin data
        mdata = {}
        for data in et_element.findall(cls.DATA):
            mdata[data.attrib[cls.NAME]] = None

        # readin raw data
        raw_data = {}
        for data in et_element.findall(cls.RAW_DATA):
            raw_data[data.attrib[cls.NAME]] = None

        is_mean = (et_element.attrib['is_mean'].upper == 'TRUE')
        if is_mean:
            # TODO: readin base measurements
            pass
        mobj = cls(sobj=sobj, id=et_element.attrib['id'], mtype=et_element.attrib['mtype'], ismean=is_mean)
        return mobj
