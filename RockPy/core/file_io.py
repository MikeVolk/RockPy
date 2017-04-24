import RockPy
from RockPy.core.utils import extract_tuple, to_tuple, split_num_alph, tuple2str
import os, re
from copy import deepcopy
import warnings
import logging

def read_abbreviations():
    """
    Reads the abbreviations.txt file into a dictionary

    Returns
    -------
        get_mtype_ftype: dict (abbrev:mtype/ftype)
            a dictionary used to get to the true mtype/ftype name from an abbreviation
        get_abbreviations: dict(list) (mtype/ftype:[abbreviation])
            a dictionary with a list of all possible abbreviations.txt
    """
    RockPy.log.debug('READING FTYPE/MTYPE abbreviations.txt')

    # open the file
    with open(os.path.join(RockPy.installation_directory, 'abbreviations.txt')) as f:
        get_abbreviations = f.readlines()

    # create the mtype:abbreviation dict
    get_abbreviations = [tuple(i.rstrip().split(':')) for i in get_abbreviations if i.rstrip() if not i.startswith('#')]
    get_abbreviations = dict((i[0], [j.lstrip() for j in i[1].split(',')]) for i in get_abbreviations)

    # create inverse abbrev:mtype/ftype
    get_mtype_ftype = {i: k for k in get_abbreviations for i in get_abbreviations[k]}
    get_mtype_ftype.update({k: k for k in get_abbreviations})
    return get_mtype_ftype, get_abbreviations


class minfo():
    """
    Class for creating and reading RockPy filenames.

    Sample name considerations
    --------------------------

        Sample names have to be unique
        Sample names can not contain any of the following characters: ','


    Examples
    --------
        The filenames have to be structured in this way to be read in properly.

        MEASUREMENT_BLOCK # SAMPLE_BLOCK # SERIES_BLOCK # ADDITIONALS_BLOCK # COMMENT .idx

        (sgroups)_(samples)_(mtypes)_ftype#mass&unit_height&unit_diameter&unit#(STYPE,SVAL,SUNIT)#(add1:addval,add2:addval2)#this is a comment.index
    """

    def extract_series(self, s):
        s = extract_tuple(s)
        s = tuple([s[0], float(s[1]), s[2]])
        return s

    def measurement_block(self, block):
        """
        The measurement block entails the samplegroups the sample belongs to, the samples included in the measurement
        file if more than one, the measurement types included in the measurement file (if more than one) and the
        filetype the file is in

        Parameters
        ----------
            block:

        Returns
        -------

        """
        sgroups, samples, mtypes, ftype = block.split('_')

        # names with , need to be replaced
        if not '(' in samples and ',' in samples:
            samples = samples.replace(',', '.')
            RockPy.log.warning('sample name %s contains \',\' will be replaced with \'.\'' % samples)

        self.sgroups = extract_tuple(sgroups)
        self.samples = extract_tuple(samples)
        self.mtypes = extract_tuple(mtypes)
        self.ftype = ftype
        self.mtypes = tuple(RockPy.abbrev_to_classname[mtype.lower()] for mtype in to_tuple(self.mtypes))
        self.ftype = RockPy.abbrev_to_classname[ftype.lower()]

    def sample_block(self, block):
        """
        Sample block holds the sample information e.g. mass, height, diameter

        Parameters
        ----------
        block

        Returns
        -------

        """
        out = [[None, None], [None, None], [None, None]]
        units = []

        if '_' in block:
            # old style infos
            block = block.replace('[', '').replace(']', '')
            block = block.replace(',', '.')
            parts = block.split('_')
        else:
            parts = block.split(',')

        for i in range(3):
            try:
                p = parts[i]
                val = float(re.findall(r"[-+]?\d*\.\d+|\d+", p)[0])
                unit = ''.join([i for i in p if not i.isdigit()]).strip('.')
            except IndexError:
                val = None
                unit = None
            out[i][0] = val
            out[i][1] = unit
        [self.mass, self.massunit], [self.height, self.heightunit], [self.diameter, self.diameterunit] = out

    def series_block(self, block):
        """
        Series block in the minfo

        Parameters
        ----------
        block

        Returns
        -------

        """
        # old style series block: e.g. mtime(h)_0,0_h;mtime(m)_0,0_min;speed_1100,0_rpm
        if not any(s in block for s in ('(', ')')) or ';' in block:
            block = block.translate(str.maketrans(",_;", ".,_", ""))

        # split block parts
        series = block.split('_')

        if not series:
            self.series = None
        self.series = [self.extract_series(s) for s in series if s]

    def add_block(self, block):
        """
        creates the add block. additional information is copmma seperated

        Parameters
        ----------
        block

        Returns
        -------

        """
        if block:
            self.additional = block.split(',')
        else:
            self.additional = []

    def comment_block(self, block):
        self.comment = block

    def get_measurement_block(self):
        """
        Function that joins the parts in the measurement block.

            (Samplegroups)_(Samples)_(MTYPES)_ftype

        Returns
        -------
            str

        Examples
        --------

            (SG1,SG2)_(S1,S2)_(HYS,DCD)_VSM
        """
        block = deepcopy(self.storage[0])
        block[2] = [RockPy.classname_to_abbrev[mtype][0].upper() for mtype in to_tuple(block[2]) if mtype]
        block[3] = RockPy.classname_to_abbrev[block[3]][0].upper()
        if not all(block[1:]):
            raise ImportError('sname, mtype, ftype needed for minfo to be generated')
        return '_'.join((tuple2str(b) for b in block))

    def get_sample_block(self):
        out = ''
        block = self.storage[1]

        if not any((all(b) for b in block)):
            return None

        for i, b in enumerate(block):
            if not all(b):
                if i == 0:
                    aux = 'XXmg'
                else:
                    aux = 'XXmm'
            else:
                aux = ''.join(map(str, b))
            if not out:
                out = aux
            else:
                out = ','.join([out, aux])

            # stop if no more entries follow
            if not any(all(i) for i in block[i + 1:]):
                break
        return out

    def get_series_block(self):
        block = self.storage[2]
        if block:
            if type(block[0]) != tuple:
                block = (block,)
            out = [tuple2str(b) for b in block]
            return '_'.join(out)

    def get_add_block(self):
        self.additional = to_tuple(self.additional)
        if self.additional:
            out = tuple(''.join(map(str, a)) for a in self.additional)
            return tuple2str(out)

    def is_readable(self):
        if not os.path.isfile(self.fpath):
            return False
        if all(self.storage[0][1:]):
            return True
        else:
            return False

    @staticmethod
    def read_input(item, unit=None):
        """
        takes an input for mass, height ... and reads it. If no unit is given assumes SI

        Parameters
        ----------
        item: str, float, tuple
            The item to be read. Can be '30mg', (30,'mg') or 30
        unit: str
            standard unit. e.g. 'kg' if no unit can be found

        Returns
        -------
            tuple
                value, unit
        """

        # no input is given
        if not item:
            return None, None

        # if string input then extract float
        if isinstance(item, str):
            value, unit = split_num_alph(item)

        elif isinstance(item, (tuple, list)):
            value, unit = item
        else:
            value = item
        return value, unit

    def __init__(self, fpath=None,
                 sgroups=None, samples=None,
                 mtypes=None, ftype=None, dialect=None,
                 mass=None, height=None, diameter=None,
                 series=None, comment=None, folder=None, suffix=None,
                 read_fpath=True, **kwargs):

        """

        Parameters
        ----------
        fpath
        sgroups
        samples
        mtypes
        ftype
        mass
        height
        diameter
        series
        comment
        folder
        suffix
        read_fpath: bool
            if true the path will be read for info
        kwargs
        """
        # copy any misspelling ( mtype instead of types )
        if 'mtype' in kwargs and not mtypes:
            mtypes = kwargs.pop('mtype')
        if 'sgroup' in kwargs and not sgroups:
            sgroups = kwargs.pop('sgroup')
        if 'sample' in kwargs and not samples:
            samples = kwargs.pop('sample')

        self.dialect = dialect

        # create the blocks
        blocks = (self.measurement_block, self.sample_block, self.series_block, self.add_block, self.comment_block)
        self.additional = []

        sgroups = to_tuple(sgroups)
        sgroups = tuple([sg if sg != 'None' else None for sg in sgroups])

        if mtypes:
            mtypes = tuple(RockPy.abbrev_to_classname[mtype] for mtype in to_tuple(mtypes))
        if ftype:
            ftype = RockPy.abbrev_to_classname[ftype]

        # initialize the class variables
        self.__dict__.update({i: None for i in ('sgroups', 'samples', 'mtypes', 'ftype',
                                                'mass', 'height', 'diameter',
                                                'massunit', 'lengthunit', 'heightunit', 'diameterunit',
                                                'series', 'comment', 'folder', 'suffix')
                              })
        self.fpath = fpath

        if read_fpath and fpath:  # todo add check for if path is readable
            # get the directory
            self.folder = os.path.dirname(fpath)
            # get the file name and the suffix
            f, self.suffix = os.path.splitext(os.path.basename(fpath))
            # remove . from suffix
            self.suffix = self.suffix.strip('.')
            splits = f.split('#')

            # check if RockPy compatible e.g. first part must be len(4)
            if not len(splits[0]) == 4:
                pass
            for i, block in enumerate(blocks[:len(splits)]):
                if splits[i]:
                    try:
                        block(splits[i])
                    except (ValueError,):
                        pass

        # read the mass, height, diameter inputs
        # read input only if it wasn't already read from the filepath
        if not self.mass and not self.massunit:
            self.mass, self.massunit = self.read_input(mass, 'kg')
        if not self.height and not self.heightunit:
            self.height, self.heightunit = self.read_input(height, 'm')
        if not self.diameter and not self.diameterunit:
            self.diameter, self.diameterunit = self.read_input(diameter, 'm')

        # set the attributes
        for i in ('sgroups', 'samples', 'mtypes', 'ftype',
                  'series', 'comment', 'folder'):

            if locals()[i]:
                if isinstance(locals()[i], (tuple, list, set)):
                    if not all(locals()[i]):
                        continue
                setattr(self, i, locals()[i])
        if kwargs:
            for k, v in kwargs.items():
                self.additional.append('{}:{}'.format(k, v))

        if dialect is not None:
            self.additional.append('d:{}'.format(dialect))

        if suffix:
            self.suffix = suffix

        if type(self.suffix) == int:
            self.suffix = u'{0:03d}'.format(self.suffix)

        if not self.suffix:
            self.suffix = '000'

        if not self.sgroups: self.sgroups = None

        self.storage = [[self.sgroups, self.samples, self.mtypes, self.ftype],
                        [[self.mass, self.massunit], [self.height, self.heightunit],
                         [self.diameter, self.diameterunit], ],
                        self.series,
                        (self.additional,),
                        self.comment]

    @property
    def fname(self):
        """
        filename after new RockPy convention

        Examples
        --------
            (a,b)_S1_(HYS,DCD)_VSM#30.0mg,30.0mm,30.0mm#(test,2.0,unit).000
        """

        out = [self.get_measurement_block(), self.get_sample_block(),
               self.get_series_block(), self.get_add_block(), self.comment]

        for i, block in enumerate(out[::-1]):
            if not block:
                out.pop()
            else:
                break
        fname = '#'.join(map(str, out)) + '.' + self.suffix
        fname = fname.replace('None', '')
        return fname

    @property
    def measurement_infos(self):
        """
        Generator object that cycles through all samples and returns a measurement_info dictionary for each of the samples.

        The dictionary has this structure

            {'fpath': self.fpath,
             'ftype': self.ftype,
             'idx': self.suffix,
             'series': self.series}

        Returns
        -------

        """
        idict = {'fpath': self.fpath,
                 'ftype': self.ftype,
                 'idx': self.suffix,
                 'series': self.series,
                 'dialect':self.dialect}

        samples = to_tuple(self.samples)
        for i in samples:
            for j in self.mtypes:
                mtype = RockPy.abbrev_to_classname[j]
                idict.update({'mtype': mtype, 'sample': i})
                yield idict

    @property
    def sample_infos(self):
        """
        Generator object that cycles through all samples and returns a sample_info dictionary for each of the samples.

        The dictionary has this structure

            dict(
                mass=self.mass,
                diameter=self.diameter,
                height=self.height,
                mass_unit=self.massunit,
                height_unit=self.heightunit,
                diameter_unit=self.diameterunit,
                samplegroup=self.sgroups
                )

        Returns
        -------

        """
        sdict = dict(mass=str(self.mass) + self.massunit if self.mass else None,
                     diameter=str(self.diameter) + self.diameterunit if self.diameter else None,
                     height=str(self.height) + self.heightunit if self.height else None,
                     samplegroup=self.sgroups, dialect=self.dialect)

        samples = to_tuple(self.samples)
        for i in samples:
            sdict.update({'name': i})
            yield sdict


class ImportHelper(object):
    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.ImportHelper
        return logging.getLogger('RockPy.ImportHelper')

    @staticmethod
    def extract_measurement_block(block):
        """
        The measurement block entails the samplegroups the sample belongs to, the samples included in the measurement
        file if more than one, the measurement types included in the measurement file (if more than one) and the
        filetype the file is in

        Parameters
        ----------
            block

        Returns
        -------

        """
        sgroups, snames, mtypes, ftype = block.split('_')

        # names with , need to be replaced
        if '(' not in snames and ',' in snames:
            snames = snames.replace(',', '.')
            RockPy.log().warning('sample name %s contains \',\' will be replaced with \'.\'' % snames)

        sgroups = RockPy.core.utils.extract_tuple(sgroups)
        snames = RockPy.core.utils.extract_tuple(snames)
        mtypes = RockPy.core.utils.extract_tuple(mtypes)
        ftype = RockPy.abbrev_to_classname[ftype.lower()]

        mtypes = tuple(RockPy.abbrev_to_classname[mtype.lower()] for mtype in RockPy.to_tuple(mtypes))

        return sgroups, snames, mtypes, ftype

    @staticmethod
    def extract_sample_block(block):
        """
        Sample block holds the sample information e.g. mass, height, diameter

        Parameters
        ----------
        block

        Returns
        -------

        """

        # has to be initialized
        out = [[None, None], [None, None], [None, None]]

        if '_' in block:
            warnings.warn('old style naming convention << 1,0[mg]_1,0[mm]_3,0[mm] >> use 1.0mg,2.0mm,3.0mm')
            # old style infos
            block = block.replace('[', '').replace(']', '')
            block = block.replace(',', '.')
            parts = block.split('_')
        else:
            parts = block.split(',')

        # extract (mass, massunit), (height, heightunit), (diameter,diameterunit)) from the block
        for i in range(3):
            try:
                p = parts[i]
                val = float(re.findall(r"[-+]?\d*\.\d+|\d+", p)[0])
                unit = ''.join([i for i in p if not i.isdigit()]).strip('.')
            except IndexError:
                val = None
                unit = None
            out[i][0] = val
            out[i][1] = unit
        return out

    @staticmethod
    def extract_series_block(block):
        """
        Series block in the minfo

        Parameters
        ----------
        block

        Returns
        -------

        """

        def extract_series(s):
            """
            extracts a series tuple from a str
            """
            s = RockPy.core.utils.extract_tuple(s)
            s = tuple([s[0], float(s[1]), s[2]])
            return s

        # translate oldstyle files
        # old style series block: e.g. mtime(h)_0,0_h;mtime(m)_0,0_min;speed_1100,0_rpm
        if not any(s in block for s in ('(', ')')) or ';' in block:
            warnings.warn(
                'old style series block: ''mtime(h)_0,0_h;height_10,0_m'' use: ''(mtime,0.0,h)_(height,10.0,m)'' ')
            block = block.translate(str.maketrans(",_;", ".,_", ""))

        # split block parts
        series = block.split('_')

        if not series:
            self.series = None

        try:
            return [extract_series(s) for s in series if s]
        except IndexError:
            return block

    @classmethod
    def extract_add_dialect_block(block):
        """
        creates the add block. additional information is comma seperated. 

        if dialect=XXX: this will then be passed

        Parameters
        ----------
        block

        Returns
        -------
            list of additionals
            str, dialect

        """
        parts = block.split(',')
        dialect = [p for p in parts if 'dialect' in p][0].replace('dialect=', '')
        parts = [p for p in parts if not 'dialect' in p]
        return additional, dialect

    @classmethod
    def from_folder(cls, folder, filter=dict()):
        dfiles = [os.path.join(folder, i) for i in os.listdir(folder) if not i.startswith('#')]

        minfo = None
        for f in dfiles:
            a = cls.from_file(f)

            if a is None:
                cls.log().warning('cant read file: %s' % os.path.basename(f))
                continue

            if minfo is None:
                minfo = a
            else:
                minfo = minfo + a

        return minfo

    @classmethod
    def from_file(cls, fpath):

        # get the directory
        folder = os.path.dirname(fpath)
        # get the file name and the suffix
        filename, suffix = os.path.splitext(os.path.basename(fpath))
        # remove . from suffix
        suffix = suffix.strip('.')

        splits = filename.split('#')

        # check if RockPy compatible e.g. first part must be len(4)
        if not len(splits[0].split('_')) == 4:
            return

        # extract mesurement information
        sgroups, snames, mtypes, ftype = cls.extract_measurement_block(splits[0])

        # extract sample informations
        if len(splits) > 1:
            (mass, massunit), (height, heightunit), (diameter, diameterunit) = cls.extract_sample_block(splits[1])

        # extract series information
        try:
            series = cls.extract_series_block(splits[2])

            if series == splits[2]:
                cls.log().warning("cant read series: %s" % os.path.basename(filename))
                series = None

        except IndexError:
            series = None

        # additional and dialect
        if len(splits) > 3:
            additional, dialect = cls.extract_add_block(splits[3])
        else:
            additional, dialect = (None, None)

        # comment
        if len(splits) > 4:
            comment = splits[4]
        else:
            comment = None

        return cls(snames, mtypes, ftype, fpath, sgroups,
                   dialect,
                   mass, massunit, height, heightunit, diameter, diameterunit,
                   series, '', additional, suffix)

    @classmethod
    def from_dict(cls, **kwargs):
        return cls(**{param: kwargs.get(param, None) for param in ('snames', 'mtypes', 'ftype', 'fpath', 'sgroups',
                      'dialect',
                      'mass', 'massunit', 'height', 'heightunit', 'diameter', 'diameterunit',
                      'series', 'comment', 'additional', 'suffix')})

    def __add__(self, other):

        for param in ('snames', 'mtypes', 'ftype', 'fpath', 'sgroups',
                      'dialect',
                      'mass', 'massunit', 'height', 'heightunit', 'diameter', 'diameterunit',
                      'series', 'comment', 'additional', 'suffix'):
            getattr(self, param).extend(getattr(other, param))
        return self

    def __init__(
            self,
            snames,
            mtypes=None,
            ftype=None, fpath=None,
            sgroups=None,
            dialect=None,
            mass=None, massunit=None,
            height=None, heightunit=None,
            diameter=None, diameterunit=None,
            series=None,
            comment=None,
            additional=None,
            suffix=None, ):
        """
        constructor
        """
        self.snames = RockPy.core.utils.tuple2list_of_tuples(RockPy.to_tuple(snames))
        self.sgroups = RockPy.core.utils.tuple2list_of_tuples(RockPy.to_tuple(sgroups))

        self.mtypes = RockPy.core.utils.tuple2list_of_tuples(RockPy.to_tuple(mtypes))

        self.ftype = RockPy.core.utils.to_list(RockPy.to_tuple(ftype))
        self.fpath = RockPy.core.utils.to_list(fpath)
        self.dialect = RockPy.core.utils.to_list(dialect)

        self.mass = RockPy.core.utils.to_list(mass)
        self.massunit = RockPy.core.utils.to_list(massunit)

        self.height = RockPy.core.utils.to_list(height)
        self.heightunit = RockPy.core.utils.to_list(heightunit)
        self.diameter = RockPy.core.utils.to_list(diameter)
        self.diameterunit = RockPy.core.utils.to_list(diameterunit)

        self.series = RockPy.core.utils.to_list([series])
        self.comment = RockPy.core.utils.to_list(comment)
        self.additional = RockPy.core.utils.to_list(additional)
        self.suffix = RockPy.core.utils.to_list(suffix)

    @classmethod
    def get_measurement_block(cls, sgroups, snames, mtypes, ftype):
        """
        Method that joins the parts in the measurement block.

            (Samplegroups)_(snames)_(MTYPES)_ftype

        Returns
        -------
            str

        Examples
        --------

            (SG1,SG2)_(S1,S2)_(HYS,DCD)_VSM
        """
        mtypes = (RockPy.classname_to_abbrev[i][0] for i in mtypes)
        block = [sgroups, snames, mtypes, ftype]
        if not all(i for i in block):
            raise ImportError('sname, mtype, ftype needed for minfo to be generated')
        return '_'.join((RockPy.core.utils.tuple2str(b) for b in block))

    @classmethod
    def get_sample_block(cls,
                         mass, massunit,
                         height, heightunit,
                         diameter, diameterunit):
        """
        Method that joins the parts of the sample info block.

            (XXmg)_(XXmm)_(XXMM)

        In the case there is a mass, diameter and height associated

        Returns
        -------
            str

        Examples
        --------

            (0.1mg)_(S1,S2)_(HYS,DCD)_VSM
        """
        out = []
        block = [[mass, massunit], [height, heightunit], [diameter, diameterunit]]

        if not any((all(b) for b in block)):
            return ''

        # get the point for wich there is still info
        # 1.0mg -> n=1
        # 1.2mg,2.0mm ->n=3
        # XXmg,XXmm,1.2mm -> n=3

        for n, v in enumerate(block):
            if v[0] == v[1]:
                break

        for i in range(n):
            b = block[i]
            if not all(j for j in b):
                if i == 0:
                    aux = 'XXmg'
                else:
                    aux = 'XXmm'
            else:
                out.append(''.join(map(str, b)))
        return ','.join(out)

    @classmethod
    def get_series_block(cls, series):
        """
        Method that joins the series parts of the sample.

            (stype1,svalue1,sunit1)_(stype2, sval2, sunit2)


        Returns
        -------
            str

        """
        if series:
            out = [RockPy.core.utils.tuple2str(b) for b in series]
            return '_'.join(out)
        else:
            return ''

    @classmethod
    def get_add_block(cls, additional):

        if additional:
            out = tuple(''.join(map(str, a)) for a in additional)
            return RockPy.core.utils.tuple2str(out)

    @property
    def new_filenames(self):
        for f in range(self.nfiles):
            mblock = self.get_measurement_block(self.sgroups[f], self.snames[f], sorted(self.mtypes[f]), self.ftype[f])
            sblock = self.get_sample_block(self.mass[f], self.massunit[f],
                                           self.diameter[f], self.diameterunit[f],
                                           self.height[f], self.heightunit[f])
            seriesblock = self.get_series_block(self.series[f])
            addblock = self.get_add_block(self.additional[f])

            out = [mblock, sblock, seriesblock]

            # work through blocks backwards throw out blocks that are empty,
            # if ones is not empty, stop.
            for i, block in enumerate(out[::-1]):
                if not block:
                    out.pop()
                else:
                    break

            fname = '#'.join(map(str, out)) + '.' + str(self.suffix[f])
            fname = fname.replace('None', '')
            yield fname

    @property
    def nfiles(self):
        return len(self.snames)

    @property
    def file_infos(self):
        for f in range(self.nfiles):
            for sname in self.snames[f]:
                for mtype in self.mtypes[f]:
                    yield dict(name=sname, mtype=mtype, sgroups=self.sgroups[f],
                               mass=self.mass[f], diameter=self.diameter[f], height=self.height[f],
                               # file path and file type
                               fpath=self.fpath[f], ftype=self.ftype[f], dialect=self.dialect[f],
                               idx=self.suffix[f],
                               series=self.series[f],
                               comment=self.comment[f], )

    @property
    def sample_infos(self):
        for f in range(self.nfiles):
            for s in self.snames[f]:
                yield dict(name=s, sgroups=self.sgroups[f],
                           mass=self.mass[f], massunit=self.massunit[f],
                           diameter=self.diameter[f], height=self.height[f])
if __name__ == '__main__':
    a = minfo('testpath', sgroup='a', samples=('S1', 'S2'), mtypes=('hys', 'dcd'), ftype='vsm', mass='30mg',
              diameter=(30, 'mm'), series=('test', 2, 'A'), comment='post heating',
              std=13, mad=666, d='tdt')
    print(a.get_sample_block())
    print(list(a.sample_infos)[0])
    # for i in a.sample_infos:
    #     print(i)
    # b = minfo('FeNi20H_FeNi20-Ha36e060-G01_COE_VSM#11,925[mg]_[]_[]##STD:13,mad:666')
    # print(b.fname)
    # print(list(a.sample_infos)[0])
