import logging
import time

import RockPy
from RockPy.core.file_io import ImportHelper
import pandas as pd

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

    def __iter__(self):
        for s in sorted(self._samples.values()):
            yield s


    ''' SAMPLES '''

    @property
    def n_samples(self):
        return len(self._samples)

    @property
    def samples(self):
        """
        Iterator that returns each sample in Study

        Returns
        -------
            RockPy.sample
        """

        for s in sorted(self._samples.values()):
            yield s

    @property
    def samplenames(self):
        """
        Iterator yields all samplenames in Study

        Returns
        -------
            str: RockPy.Sample.name
        """

        for sname in sorted(self._samples.keys()):
            yield sname

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

    def sample_exists(self, sname=None, sobj=None):
        '''
        Method checks if a sample is in the _samples dictionary
         
        Parameters
        ----------
        sname: str (optional)
        sobj: RockPy.Sample (optional)

        Returns
        -------
            False if sample does not exist
            RockPy.Sample if sample exists
        
        Notes
        -----
            either 
            
        '''
        if sname is None and sobj is None:
            raise TypeError('sample_exists() missing 1 required positional argument: ''sname'' or ''sobj''')

        if sname is not None and sname in list(self.samplenames):
            return self._samples[sname]

        if sobj is not None and sobj in list(self.samples):
            return sobj

    def add_sample(self,
                   sname=None,
                   comment='',
                   sgroups=None,
                   sobj=None,
                   **kwargs
                   ):
        """

        Parameters
        ----------
        sname: str
            sname of sample
        comment: str
            comment - no use yet
        sgroups: str, iterable(str)
            samplegroup or groups the sample should belong to
        sobj: RockPy.Sample (optional)
            RockPy.Sample instace, no sample is created instance is just added to Study
            
        kwargs are passed on to RockPy.Sample

        """

        if self.sample_exists(sname, sobj):

            log.warning('CANT create << %s >> already in Study. Please use unique sample names. '
                        'Returning sample' % (sname if sname is not None else sobj.name))

        if sobj is None:
            sobj = RockPy.Sample(
                    name=str(sname),
                    comment=comment,
                    sgroups=sgroups, **kwargs)

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
        Method takes folder as input, cycles through all files and imports them. Does not import subfolders.
        Can be filtered to only import certain files.
        
        Parameters
        ----------
        folder: str
        filter: str
         
        Notes
        -----
            for now only samplenames can be filtered
        """
        filter = RockPy.to_tuple(filter)

        iHelper = ImportHelper.from_folder(folder)

        # create all samples
        for file_info_dict in iHelper.gen_sample_dict:
            if any(file_info_dict[v] in filter for v in ('sname',)):
                self.log().debug('filtering out file: %s'%file_info_dict['fpath'])
                continue

            self.add_sample(**file_info_dict)

            for ih in iHelper.getImportHelper(snames=file_info_dict['sname']):
                pass

    def import_file(self, fpath):
        iHelper = ImportHelper.from_file(fpath)

        for sample in iHelper.gen_sample_dict:
            s = self.add_sample(**sample)
            for importinfos in iHelper.gen_measurement_dict:
                s.add_measurement(create_parameters=False, **importinfos)

    def info(self):
        info = pd.DataFrame(columns=['mass', 'sample groups', 'mtypes', 'stypes', 'svals'])

        for s in self.samples:
            info.loc[s.name, 'mass'] = s.get_measurement('mass')[0].data

if __name__ == '__main__':
    S = RockPy.Study()
    S.import_folder('/Users/mike/github/2016-FeNiX.2/data/(HYS,DCD)')
    print(S.info())
    # S.import_file('/Users/mike/github/2016-FeNiX.2/data/(HYS,DCD)/FeNiX_FeNi00-Fa36-G01_HYS_VSM#36.5mg#(ni,0,perc)_(gc,1,No).002')

    # print(S.samples)