import numpy as np
import scipy as sp
import pandas as pd
import logging
import inspect

class Result():
    dependencies = None
    default = None
    indirect = False

    __subj = []
    __deps = []

    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.MTYPE
        return logging.getLogger('RockPy.%s' % cls.__name__)

    def set_result(self, result, result_name=None):

        if result_name is None:
            result_name = self.name

        self.mobj.sobj.results.loc[self.mobj.mid, result_name] = result

    def get_result(self, result_name=None):

        if result_name is None:
            result_name = self.name

        return self.mobj.sobj.results.loc[self.mobj.mid, result_name]

    def __call__(self, recipe=None, **parameters):
        """
        calling method for the result. iterates over the current worker of the mobj.

        Returns
        -------
            Pandas.Series
                the result that is actually calculated by the recipe. can be more than just the value, not sure if that is good
        """
        # if no recipe is specified use the last one calculated
        if recipe is None:
            recipe = self.recipe

        # initialize signature
        signature = None

        # list of results that were calculated
        # when initially called by the user, the list is created.
        # only the first element is returned

        calculation_stack = parameters.pop('calculation_stack', list())

        # check if result was calculated keeps results from being recalculated all the time
        if self.name in calculation_stack:
            self.log().debug('%s is already calculated in current run' % self.name)
            return
        else:
            calculation_stack.append(self.name)

        # set default parameters
        if not self.indirect:
            # todo so only parameters are passed on to a function that the recipe function likes
            try:
                signature = inspect.signature(self._recipes()[recipe]).parameters
                for p in signature:
                    if p == 'check':
                        continue
                    if p == 'self':
                        continue
                    if p in parameters:
                        continue
                    if p == 'unused_params':
                        continue
                    else:
                        parameters[p] = signature[p].default
            except KeyError:
                pass

        self.log().debug('called: %s' % self)
        self.log().debug('with parameters:')
        
        for k, v in parameters.items():
            # remove unused parameters
            if signature and k not in signature:
                self.log().debug('\tunused %s: %s' % (k, v))
            else:
                self.log().debug('\t%s: %s' % (k, v))

        calculate = False

        if not self.indirect:
            if not self._is_calculated:
                calculate = True
            if self._parameters_changed(**parameters):
                calculate = True
            if self._recipe_changed(recipe):
                calculate = True

            # if not calculate:
            #     return self.mobj.results.loc[self.name]
        
        # calculate all dependencies
        if self.dependencies:
            for dep_res in self._dependencies:
                self.log().debug('calling dependent << %s >> for result %s' % (dep_res.name, self.name))
                dep_res(recipe=recipe, calculation_stack=calculation_stack, **parameters)

        if calculate:
            # update parameters with new parameters
            self.params.update(**parameters)

            if recipe not in self._recipes():
                self.log().error('result %s recipe has no recipe << %s >>:' % (self.name, recipe))
                for r in self._recipes():
                    self.log().error('%s' % r)
            else:
                self.log().debug('calling result %s recipe %s' % (self.name, recipe))
                self.log().debug('\t%s' % self._recipes()[recipe])
                res = self._recipes()[recipe](self, **parameters)
        
        # calculate the subjects of the result
        if self._subjects:
            for subj_res in self._subjects:
                self.log().debug('calling subject << %s >> of result %s' % (subj_res.name, self.name))
                subj_res(recipe=recipe, calculation_stack=calculation_stack, **parameters)

        # only the first iun the stack is the one to be called and calculated 
        if self.name == calculation_stack[0]:
            return self.get_result()

    @property
    def _is_calculated(self):
        """
        Checks if the result has been calculated e.g. checks in Measurement.results
        Returns
        -------

        """
        self.log().debug('checking if %s calculated' % self.name)
        if self.mobj.results is None or self.name not in self.mobj.results:
            self.log().debug('%s NOT calculated' % self.name)
            return False
        else:
            self.log().debug('%s IS calculated' % self.name)
            return True


    def _parameters_changed(self, **params):
        self.log().debug('checking if parameters changed')

        # force recalculation for checking results and or forced recalculation with 'recalc'
        if 'check' in params or 'reclac' in params:
            self.log().debug('  FORCED RECALCULATION')
            return True

        if all(params[p] == self.params[p] for p in params if p in self.params):
            self.log().debug('  NO parameters changed')
            return False
        else:
            self.log().debug('parameters changed')
            for p in params:
                if p in self.params and not params[p] == self.params[p]:
                    try:
                        self.log().debug('%s %f --> %f' % (p, self.params[p], params[p]))
                    except TypeError:
                        self.log().debug('%s %s --> %s' % (p, self.params[p], params[p]))
                else:
                    self.log().debug('parameter %s not used' % p)
            return True

    def _recipe_changed(self, recipe):
        self.log().debug('checking recipe:')
        if self.recipe == recipe:
            self.log().debug('recipe NOT changed')
            return False
        else:
            self.log().debug('recipe changed %s -> %s' % (self.recipe, recipe))
            return True

    @property
    def name(self):
        return self.__class__.__name__.replace('result_', '')

    @classmethod
    def _recipes(cls):
        ''' creates a list of recipes for this result'''
        return {i.replace('recipe_', ''): getattr(cls, i) for i in dir(cls)
                if i.startswith('recipe') if not i.endswith('recipes')}

    def __init__(self, mobj, **kwargs):
        self.log().debug('initializing instance %s' % self.name)
        self.mobj = mobj
        self.recipe = None
        self.params = {}

    @property
    def _dependencies(self):
        '''
        returns a list of the instances that need to be calculated before result can be calculated
        '''

        if not self.__deps:
            self.log().debug('Checking dependencies for << %s >>' % self.name)
            if self.dependencies is not None:
                self.log().debug('  dependent on: %s' % ', '.join(self.dependencies))
                dependencies = [instance for res, instance in self.mobj._results.items() if res in self.dependencies]
            else:
                self.log().debug('  no dependencies')
                dependencies = []
            self.__deps = dependencies
        return self.__deps

    @property
    def _subjects(self):
        if not self.__subj:
            subjects = []
            self.log().debug('Checking subjects for << %s >>' % self.name)
            for res, instance in self.mobj._results.items():
                if instance.dependencies is None:
                    continue
                if self.name in instance.dependencies:  # and not instance.indirect:
                    subjects.append(instance)
            if subjects:
                self.log().debug('  subjects: %s' % ', '.join([s.name for s in subjects]))
            else:
                self.log().debug('  no subjects')

            self.__subj = subjects
        return self.__subj