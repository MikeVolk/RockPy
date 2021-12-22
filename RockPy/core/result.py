import numpy as np
import scipy as sp
import pandas as pd
import logging
import inspect
import RockPy

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

        """
        Args:
            result:
            result_name:
        """
        if result_name is None:
            result_name = self.name

        self.mobj.sobj._results.loc[self.mobj.mid, result_name] = result
        self.mobj.sobj._results.loc[self.mobj.mid, 'sID'] = self.mobj.sobj.sid


    def get_result(self, result_name=None):
        """Helper function for easy retrieval of results from a measurement

        Args:
            result_name:

        Returns:
            * *False if the mID is not in the result, yet*
            * *Otherwise it returns the result*
        """
        if result_name is None:
            result_name = self.name

        return self.mobj.sobj._results.loc[self.mobj.mid, result_name]

    def get_stack(self, stack=None):
        """Method retrieves the all results that have to be calculated in order
        to maintain consistency

        i.e. if parameters are changed and resultB is calculated using
        resultA, both have to be recalculated.

        Returns a list of result instances

        1. results dependent on result 1b results dependent on dependency 2.
        Result (self) 3. subjects to self

        Args:
            stack (passed to next result, so that results are not calculated multiple times):

        Returns:
            list:
        """
        if stack is None:
            stack = []

        if self.dependencies:
            for dep_res in self._dependencies:
                stack = dep_res.get_stack(stack)

        if self not in stack:
            stack.append(self)
        else:
            return stack

        if self._subjects:
            for subj_res in self._subjects:
                stack = subj_res.get_stack(stack)

        return stack

    def get_recipe(self, recipe):
        """Sets the given recipe for the result. If recipe is None it uses the
        previously used recipe. If the result was not calculated, yet,
        self.recipe is also None and it falls back to the default_recipe

        Args:
            recipe (str): name of the recipe with or without 'recipe_'

        Returns:
            str:

        Raises:
            * KeyError if recipe is not in self._recipes().keys()
        """

        if recipe == 'default':
            self.log().info('Setting %s to default recipe << %s >>'%(self.name, self.default_recipe))
            recipe = self.default_recipe

        if recipe is not None and recipe.replace('recipe_', '') not in self._recipes().keys():
            raise KeyError('%s is not a valid recipe for the measurement %s: try one of these << %s >>'%(recipe, self.mobj.mtype, list(self._recipes().keys())))

        # if no recipe is specified use the last one calculated
        if recipe is None:
            recipe = self.recipe

        # if this is the first -> recipe is still None -> run use the default_recipe
        if recipe is None:
            recipe = self.default_recipe
        return recipe

    def __call__(self, recipe=None, **parameters):
        """calling method for the result. iterates over the current worker of
        the mobj.

        Args:
            recipe:
            **parameters:

        Returns:
            Pandas.Series: the result that is actually calculated by the recipe.
            can be more than just the value, not sure if that is good
        """
        recipe = self.get_recipe(recipe=recipe)

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
                    if (
                        p == 'check'
                        or p == 'self'
                        or p in parameters
                        or p == 'unused_params'
                    ):
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
                        parameters.pop('recalc', None)
                        result._recipes()[recipe](result, **parameters)

                    # set the calculation recipe
                    result.recipe = recipe
        return self.get_result()

    @staticmethod
    def _needs_to_be_calculated(result, recipe, **parameters):

        """
        Args:
            result:
            recipe:
            **parameters:
        """
        if result.mobj.mid not in result.mobj.sobj.results.index:
            return True
        if parameters.pop('recalc', False):
            result.log().debug('FORCED recalculation of %s using keyword ''recalc''' % result.name)
            return True
        if not result._is_calculated:
            return True
        if result._parameters_changed(**parameters):
            return True
        return bool(result._recipe_changed(recipe))

    @property
    def _is_calculated(self):
        """Checks if the result has been calculated e.g. checks in
        Measurement.results
        """
        if (
            self.mobj.results is not None
            and self.name in self.mobj.results
            and not np.isnan(self.get_result())
        ):
            self.log().debug('%s IS calculated' % self.name)
            return True

        self.log().debug('%s NOT calculated' % self.name)
        return False

    def _parameters_changed(self, **params):

        # force recalculation for checking results and or forced recalculation with 'recalc'
        """
        Args:
            **params:
        """
        if 'check' in params or 'reclac' in params:
            self.log().debug('FORCED RECALCULATION')
            return True

        if all(params[p] == self.params[p] for p in params if p in self.params):
            self.log().debug('NO parameters changed')
            return False
        else:
            self.log().debug('YES parameters changed')
            for p in params:
                if p in self.params and params[p] != self.params[p]:
                    try:
                        self.log().debug('%s %f --> %f' % (p, self.params[p], params[p]))
                    except TypeError:
                        self.log().debug('%s %s --> %s' % (p, self.params[p], params[p]))
                else:
                    self.log().debug('parameter %s not used' % p)
            return True

    def _recipe_changed(self, recipe):
        """
        Args:
            recipe:
        """
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
        """creates a list of recipes for this result"""
        return {i.replace('recipe_', ''): getattr(cls, i) for i in dir(cls)
                if i.startswith('recipe') if not i.endswith('recipes')}

    @property
    def implemented_recipes(self):
        return list(self._recipes().keys())

    def __init__(self, mobj, **kwargs):
        """
        Args:
            mobj:
            **kwargs:
        """
        self.mobj = mobj
        self.log().debug('initializing instance %s' % self.name)
        self.recipe = None
        self.params = {}
        self.set_default_recipe()

    def set_default_recipe(self):
        """Sets the default_recipe recipe if only one recipe exists."""
        if self.default_recipe is None:
            if len(self._recipes()) != 1:
                self.log().error('Result << %s >> has more than one recipe, but no default_recipe recipe ' % (self.name))
                raise KeyError
            self.default_recipe = list(self._recipes().keys())[0]
            self.log().debug('default_recipe recipe for %s not specified setting to only available << %s >>' % (
                self.name, self.default_recipe))
            self.log().debug('setting default_recipe recipe << %s >> for %s' % (self.default_recipe, self.name))

    @property
    def _dependencies(self):
        """returns a list of the instances that need to be calculated before
        result can be calculated
        """

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