import logging
import time

import RockPy
from RockPy.core.file_io import ImportHelper

log = logging.getLogger(__name__)


class Study(object):
    """
    comprises data of a whole study
    i.e. container for samplegroups
    """

    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.MTYPE
        return logging.getLogger('RockPy.Study')

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
                   mass=None, massunit=None,
                   height=None, diameter=None, lengthunit=None,
                   sgroups=None,
                   sobj=None,
                   **options
                   ):
        """

        Parameters
        ----------
            massunit
            lengthunit
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
            sgroups
            sobj
            options
        """

        if name in self.samplenames:
            log.warning('CANT create << %s >> already in Study. Please use unique sample names. '
                        'Returning sample' % name)
            return self._samples[name]

        if not sobj:
            sobj = RockPy.Sample(
                    name=str(name),
                    comment=comment,
                    mass=mass, massunit=massunit,
                    height=height, diameter=diameter, lengthunit=lengthunit,
                    sgroups=sgroups,
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
                      filter=None,
                      ):
        """

        Args:
            folder:
            filter:
        """
        filter = RockPy.to_tuple(filter)

        iHelper = ImportHelper.from_folder(folder)

        # create all samples
        for f in iHelper.sample_infos:
            if any(f[v] in filter for v in ('name',)):
                self.log().debug('filtering out file: %s'%f['fpath'])
                continue
            self.add_sample(**f)

            for ih in iHelper.getImportHelper(snames=f['name']):
                pass
        # raise NotImplementedError

    def import_file(self, fpath):
        iHelper = ImportHelper.from_file(fpath)

        imported_samples = []
        for sample in iHelper.sample_infos:
            if sample['name'] not in self._samples:
                imported_samples.append(self.add_sample(**sample))
        for sample in imported_samples:
            for importinfos in iHelper.getImportHelper(snames=sample.name):
                sample.add_measurement(importinfos=importinfos, create_parameters=False)

    def info(self):
        raise NotImplementedError

if __name__ == '__main__':
    S = RockPy.Study()
    # S.import_folder('/Users/mike/github/2016-FeNiX.2/data/(HYS,DCD)')

    S.import_file('/Users/mike/github/2016-FeNiX.2/data/(HYS,DCD)/FeNiX_FeNi00-Fa36-G01_(IRM,DCD)_VSM#36.5mg#(ni,0,perc)_(gc,1,No).002')
    print(S.samples)