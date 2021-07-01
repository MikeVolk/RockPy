import logging
import time

import RockPy
from RockPy.core.file_io import ImportHelper
from RockPy.core.utils import to_tuple

import pandas as pd
import numpy as np

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
        """
        general Container for Samples.

        Parameters
        ----------
            name: str
                name of the Study

        Returns
        -------
            RockPy.Study object
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

        self.log().debug(f'Creating Study[{self.studID}] << {name} >>')
        # if folder:
        #     self.import_folder(folder)

    def __repr__(self):
        # if self == RockPy.Study:
        #     return '<< RockPy3.MasterStudy >>'.format(self.name)
        return '<< RockPy.Study.{} -- {} >>'.format(self.name, self.studID)

    def __iter__(self):
        for s in sorted(self._samples.values()):
            yield s

    def __getitem__(self, item):
        if isinstance(item, int):
            item = list(self._samples.keys())[item]
        item = str(item)
        if item in self._samples:
            return self._samples[item]
        else:
            self.log().error('Sample not in Study')

    ''' SAMPLES '''

    @property
    def n_samples(self):
        """
        Property that returns the number of samples

        Returns
        -------
            int
        """
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

    @property
    def sample_list(self):
        """
        Iterator that returns each sample in Study

        Returns
        -------
            RockPy.sample
        """

        return list(self.samples)


    ''' measurements '''
    @property
    def measurements(self):
        '''
        Iterator that returns each measurement in the Study

        Returns
        -------
            iterator of measurements
        '''

        for s in self.samples:
            for m in s.measurements:
                yield m

    @property
    def measurement_list(self):
        """
        List of all measurements in the Study

        Returns
        -------
            list
        """
        return list(self.measurements)

    @property
    def mtypes(self):
        '''
        returns a sorted list of unique mtypes
        '''
        return sorted(set(m.mtype for m in self.measurements))

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
                study=self,
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

    ####################################################################################################################

    def get_samplegroup(self, gname=None):
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

    def get_sample(self,
                   sid=None,
                   gname=None,
                   sname=None,
                   mtype=None,
                   series=None,
                   stype=None, sval=None, sval_range=None,
                   mean=False,
                   invert=False,
                   ):

        slist = list(self.samples)

        if not any(i for i in [gname, sname, mtype, series, stype, sval, sval_range, mean, invert]):
            return slist

        # samplegroup filtering
        if gname:
            gname = to_tuple(gname)
            slist = [s for s in slist if any(sg in gname for sg in s._samplegroups)]

        # sample filtering
        if sname:
            sname = to_tuple(sname)
            slist = [s for s in slist if s.name in sname]

        if any(i for i in [mtype, series, stype, sval, sval_range, mean, invert]):
            slist = [s for s in slist if s.get_measurement(mtype=mtype,
                                                           stype=stype, sval=sval, sval_range=sval_range,
                                                           series=series,
                                                           mean=mean,
                                                           invert=invert)]
        return slist

    def get_measurement(self,
                        gname=None,
                        sname=None,
                        mtype=None,
                        series=None,
                        stype=None, sval=None, sval_range=None,
                        invert=False,
                        mid=None,
                        sid=None,
                        ):

        if mid:
            return [m for s in self.samples for m in s.get_measurement(mid=mid, invert=invert)]

        else:
            samples = self.get_sample(gname=gname, sname=sname, mtype=mtype, series=series,
                                      stype=stype, sval=sval, sval_range=sval_range, invert=invert,
                                      sid=sid)

            mlist = (m for s in samples for m in s.get_measurement(mtype=mtype, series=series,
                                                                   stype=stype, sval=sval, sval_range=sval_range,
                                                                   invert=invert))
        return list(mlist)

    ''' IMPORT functions '''

    def import_folder(self,
                      folder,
                      arg_filter=None,
                      **kwargs
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
        start = time.time()
        arg_filter = RockPy.to_tuple(arg_filter)

        iHelper = ImportHelper.from_folder(folder, **kwargs)

        mlist = []
        slist = []
        # create all samples
        for sample_info_dict in iHelper.gen_sample_dict:
            if any(sample_info_dict[v] in arg_filter for v in ('sname',)):
                self.log().debug('filtering out file: %s' % sample_info_dict['fpath'])
                continue

            s = self.add_sample(**sample_info_dict)
            slist.append(s)

            # create all measurements
            for i, measurement_dict in enumerate(iHelper.gen_measurement_dict):
                if s.name != measurement_dict['sname']:
                    continue
                m = s.add_measurement(create_parameters=False, **measurement_dict)
                if m is not None:
                    mlist.append(m)

        self.log().info(
            '%i / %i files imported in %.2f seconds' % (
                len(mlist),
                iHelper.nfiles,
                time.time() - start))

    def import_file(self, fpath):
        iHelper = ImportHelper.from_file(fpath)

        for sample in iHelper.gen_sample_dict:
            s = self.add_sample(**sample)
            for importinfos in iHelper.gen_measurement_dict:
                s.add_measurement(create_parameters=False, **importinfos)

    @property
    def info(self):
        info = pd.DataFrame()

        for s in self.samples:
            info = pd.concat([info, s.info])

        return info

    @property
    def results(self):
        results = pd.concat([s.results for s in self.samples], sort=True)
        results['sname'] = [s.name for s in self.samples for m in range(s.results.shape[0])]
        results['mtype'] = [self.get_measurement(mid=mid)[0].mtype for mid in results.index]
        return results



