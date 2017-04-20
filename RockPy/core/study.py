import logging
import time

import RockPy
from IPython.display import display
from ipywidgets import HBox, VBox
from ipywidgets import widgets

log = logging.getLogger(__name__)


class Study(object):
    """
    comprises data of a whole study
    i.e. container for samplegroups
    """

    def __init__(self, name=None, folder=None):
        # type: (str, str) -> RockPy.Study
        """
        general Container for Samplegroups and Samples.

        Parameters
        ----------
            name: str
                default_recipe: 'study'
                name of the Study

        Returns
        -------
            object
        """

        # give id to the study object
        self.studID = id(self)

        # use time if no name is specified for the study
        if not name:
            self.name = time.strftime("%Y%m%d:%H%M")
        else:
            self.name = name

        # create empty dictionary for storing samples
        self._samples = dict()  # {'sname':'sobj'}

        # create variable for all imported files to be stored. If file has been imported the fpath is stored here.
        self.imported_files = []

        # if folder:
        #     self.import_folder(folder)

    def __repr__(self):
        # if self == RockPy.Study:
        #     return '<< RockPy3.MasterStudy >>'.format(self.name)
        return '<< RockPy.Study.{} -- {} >>'.format(self.name, self.studID)

    ''' SAMPLES '''

    @property
    def n_samples(self):
        return len(self._samples)

    @property
    def samples(self):
        """
        Returns a list of all samples

        Returns
        -------
            list of all samples
        """

        return sorted([v for k, v in self._samples.items()])

    @property
    def samplenames(self):
        """
        Returns a list of all samplenames

        Returns
        -------
            list of all samplenames
        """
        return sorted([k for k, v in self._samples.items()])

    ''' SAMPLE GROUPS '''

    @property
    def n_groups(self):
        return len(self.groupnames)

    @property
    def groupnames(self):
        return sorted(set(i for j in self.samples for i in j._samplegroups))

    @property
    def samplegroups(self):
        return self.groupnames

    ''' ADD functions '''

    def add_sample(self,
                   name=None,
                   comment='',
                   mass=None,
                   height=None, diameter=None,
                   samplegroup=None,
                   sobj=None,
                   **options
                   ):
        """

        Parameters
        ----------
            name : str
                the name of the sample
            comment
            mass
            mass_unit
            height
            diameter
            x_len
            y_len
            z_len
            length_unit
            sample_shape
            coord
            samplegroup
            sobj
            options
        """

        if name in self.samplenames:
            log.warning('CANT create << %s >> already in Study. Please use unique sample names. '
                        'Returning sample' % name)
            return self._samples[name]

        if not sobj:
            sobj = RockPy.core.sample.Sample(
                    name=str(name),
                    comment=comment,
                    mass=mass,
                    height=height, diameter=diameter,
                    samplegroup=samplegroup,
            )

        self._samples.setdefault(sobj.name, sobj)
        return sobj

    def add_samplegroup(self,
                        gname=None,
                        sname=None,
                        mtype=None,
                        series=None,
                        stype=None, sval=None, sval_range=None,
                        mean=False,
                        invert=False,
                        slist=None,
                        ):
        """
        adds selected samples to a samplegroup

        Parameter
        ---------
            name: str
            default_recipe: None
            if None, name is 'SampleGroup #samplegroups'
            slist: list
                list of samples to be added to the sample_group

        Returns
        -------
            list
                list of samples in samplegroup
        """
        raise NotImplementedError

    ''' REMOVE functions '''

    def remove_sample(self,
                      gname=None,
                      sname=None,
                      mtype=None,
                      series=None,
                      stype=None, sval=None, sval_range=None,
                      mean=False,
                      invert=False,
                      ):
        """

        Parameters
        ----------
        gname
        sname
        mtype
        series
        stype
        sval
        sval_range
        mean
        invert
        """
        raise NotImplementedError

    def remove_samplegroup(self,
                           gname=None,
                           sname=None,
                           mtype=None,
                           series=None,
                           stype=None, sval=None, sval_range=None,
                           mean=False,
                           invert=False,
                           slist=None,
                           ):
        """
        removes selected samples from a samplegroup

        Parameter
        ---------
            gname: str
                the name of the samplegroup that is supposed to be removed
            slist: list
                list of samples to be removed the sample_group

        Returns
        -------
            list
                list of samples in samplegroup
        """
        raise NotImplementedError

    ''' GET functions '''

    def get_measurement(self,
                        gname=None,
                        sname=None,
                        mtype=None,
                        series=None,
                        stype=None, sval=None, sval_range=None,
                        mean=False, groupmean=False,
                        invert=False,
                        mid=None,
                        ):
        """

        Args:
            gname:
            sname:
            mtype:
            series:
            stype:
            sval:
            sval_range:
            mean:
            groupmean:
            invert:
            mid:
        """
        raise NotImplementedError

    def get_sample(self,
                   gname=None,
                   sname=None,
                   mtype=None,
                   series=None,
                   stype=None, sval=None, sval_range=None,
                   mean=False,
                   invert=False,
                   ):
        # type: (str, str, str, str, str, float, str, bool, bool) -> list

        """
        Parameters
        ----------
        gname
        sname
        mtype
        series
        stype
        sval
        sval_range
        mean
        invert
        """
        raise NotImplementedError

    def get_samplegroup(self, gname=None):
        # type: (str) -> list
        """
        wrapper for simply getting all samples of one samplegroup

        Parameters
        ----------
        gname : str
            name of the samplegroup

        Returns
        -------
        list

        Examples
        --------

        """
        return self.get_sample(gname=gname)

    ''' IMPORT functions '''

    def import_folder(self,
                      folder,
                      filter,
                      ):
        """

        Args:
            folder:
            filter:
        """
        raise NotImplementedError

    def import_file(self):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError
