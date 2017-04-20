import numpy as np
import scipy as sp
import pandas as pd
import logging
import inspect

class Result():
    dependencies = None
    default_recipe = None

    __calculates__ = []
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
        """
        Helper function for easy retrieval of results from a measurement
        
        Parameters
        ----------
        result_name

        Returns
        -------
            Fasle if the mID is not in the result, yet
            Otherwise it returns the result
        """
        if result_name is None:
            result_name = self.name

        return self.mobj.sobj.results.loc[self.mobj.mid, result_name]

    def get_stack(self, stack=None):

        if stack is None:
            stack = []

        if self.dependencies:
            for dep_res in self._dependencies:
                stack = dep_res.get_stack(stack)

        if not self in stack:
            stack.append(self)
        else:
            return stack

        if self._subjects:
            for subj_res in self._subjects:
                stack = subj_res.get_stack(stack)

        return stack

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

        # if this is the first run use the default_recipe
        if recipe is None:
            recipe = self.default_recipe
        # initialize signature
        signature = None

        if self._needs_to_be_calculated(self, recipe, **parameters):
            # calculation stack with all dependents and subjects of the result, that need to be calculated
            stack = self.get_stack()

            # cycle throught the stack
            for result in stack:

                # set default_recipe parameters
                signature = inspect.signature(result._recipes()[recipe]).parameters
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

                result.log().info('called: %s' % result)
                result.log().debug('with parameters:')

                for k, v in parameters.items():
                    # remove unused parameters
                    if signature and k not in signature:
                        result.log().debug('\tunused %s: %s' % (k, v))
                    else:
                        result.log().debug('\t%s: %s' % (k, v))

                if self._needs_to_be_calculated(result, recipe, **parameters):
                    # update parameters with new parameters
                    result.params.update(**parameters)

                    if recipe not in result._recipes():
                        result.log().error('result %s recipe has no recipe << %s >>:' % (result.name, recipe))
                        for r in result._recipes():
                            result.log().error('%s' % r)
                    else:
                        result.log().debug('calling result %s recipe %s' % (result.name, recipe))
                        result.log().debug('\t%s' % result._recipes()[recipe])
                        result._recipes()[recipe](result, **parameters)

                    # set the calculation recipe
                    result.recipe = recipe
        return self.get_result()

    @staticmethod
    def _needs_to_be_calculated(result, recipe, **parameters):
        if result.mobj.mid not in result.mobj.sobj.results:
            return True

        if not result._is_calculated:
            return True
        if result._parameters_changed(**parameters):
            return True
        if result._recipe_changed(recipe):
            return True
        else:
            return False

    @property
    def _is_calculated(self):
        """
        Checks if the result has been calculated e.g. checks in Measurement.results
        Returns
        -------

        """
        if self.mobj.results is not None:
            if self.name in self.mobj.results:
                if not np.isnan(self.get_result()):
                    self.log().debug('%s IS calculated' % self.name)
                    return True

        self.log().debug('%s NOT calculated' % self.name)
        return False

    def _parameters_changed(self, **params):

        # force recalculation for checking results and or forced recalculation with 'recalc'
        if 'check' in params or 'reclac' in params:
            self.log().debug('FORCED RECALCULATION')
            return True

        if all(params[p] == self.params[p] for p in params if p in self.params):
            self.log().debug('NO parameters changed')
            return False
        else:
            self.log().debug('YES parameters changed')
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
        if self.recipe == recipe:
            self.log().debug('NO recipe changed')
            return False
        else:
            self.log().debug('YES recipe changed %s -> %s' % (self.recipe, recipe))
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
        self.mobj = mobj
        self.log().debug('initializing instance %s' % self.name)
        self.recipe = None
        self.params = {}
        self.set_default_recipe()
        #
        # # create the classes for each of the indirect results from cls.__calculates__
        # self.__calculates__ = kwargs.pop('calculates', self.__class__.__calculates__)
        # for method in self.__calculates__:
        #     if not method in dir(self.mobj):
        #         self.log().debug('METACLASS creation << %s >> for result << %s >>' %(method, self))
        #         SubClass = type(method, (self.__class__, ), {'mobj': None})
        #         setattr(self.mobj, method, SubClass(mobj, calculates=[]))

    def set_default_recipe(self):
        """
        Sets the default_recipe recipe if only one recipe exists.
        """
        if self.default_recipe is None:
            if not len(self._recipes()) == 1:
                self.log().error('Result << %s >> has more than one recipe, but no default_recipe recipe ' % (self.name))
                raise KeyError
            self.default_recipe = list(self._recipes().keys())[0]
            self.log().debug('default_recipe recipe for %s not specified setting to only available << %s >>' % (
                self.name, self.default_recipe))
            self.log().debug('setting default_recipe recipe << %s >> for %s' % (self.default_recipe, self.name))

    @property
    def _dependencies(self):
        '''
        returns a list of the instances that need to be calculated before result can be calculated
        '''

        if not self.__deps:
            if self.dependencies is not None:
                self.log().debug('YES dependencies: %s' % ', '.join(self.dependencies))
                dependencies = [instance for res, instance in self.mobj._results.items() if res in self.dependencies]
            else:
                self.log().debug('NO dependencies')
                dependencies = []
            self.__deps = dependencies
        return self.__deps

    @property
    def _subjects(self):
        if not self.__subj:
            subjects = []
            for res, instance in self.mobj._results.items():
                if instance.dependencies is None:
                    continue
                if self.name in instance.dependencies:  # and not instance.indirect:
                    subjects.append(instance)
            if subjects:
                self.log().debug('YES subjects: %s' % ', '.join([s.name for s in subjects]))
            else:
                self.log().debug('NO subjects')

            self.__subj = subjects
        return self.__subj