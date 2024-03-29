import RockPy
from RockPy.core.utils import str2tuple, to_tuple, tuple2list_of_tuples, series_to_dict, tuple2str, list_or_item
import os, re
import warnings
import logging


def read_abbreviations():
    """Reads the abbreviations.txt file into a dictionary

    Returns:
        * **get_mtype_ftype** (*dict (abbrev:mtype/ftype)*) -- a dictionary used
          to get to the true mtype/ftype sname from an abbreviation
        * **get_abbreviations** (*dict(list) (mtype/ftype:[abbreviation])*) -- a
          dictionary with a list of all possible abbreviations.txt
    """
    RockPy.log.debug("READING FTYPE/MTYPE abbreviations.txt")

    # open the file
    with open(os.path.join(RockPy.installation_directory, "abbreviations.txt")) as f:
        get_abbreviations = f.readlines()

    # create the mtype:abbreviation dict
    get_abbreviations = [tuple(i.rstrip().split(":")) for i in get_abbreviations if i.rstrip() if not i.startswith("#")]
    get_abbreviations = dict((i[0], [j.lstrip() for j in i[1].split(",")]) for i in get_abbreviations)

    # create inverse abbrev:mtype/ftype
    get_mtype_ftype = {i: k for k in get_abbreviations for i in get_abbreviations[k]}
    get_mtype_ftype.update({k: k for k in get_abbreviations})
    return get_mtype_ftype, get_abbreviations


class ImportHelper(object):
    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.ImportHelper
        return logging.getLogger("RockPy.ImportHelper")

    @staticmethod
    def extract_measurement_block(block):
        """The measurement block entails the samplegroups the sample belongs to,
        the samples included in the measurement file if more than one, the
        measurement types included in the measurement file (if more than one)
        and the filetype the file is in

        Args:
            block:
        """
        sgroups, snames, mtypes, ftype = block.split("_")

        # names with , need to be replaced
        if "(" not in snames and "," in snames:
            snames = snames.replace(",", ".")
            RockPy.log().debug("sample sname %s contains \",\" will be replaced with \".\"" % snames)

        sgroups = RockPy.core.utils.str2tuple(sgroups)
        snames = RockPy.core.utils.str2tuple(snames)
        mtypes = RockPy.core.utils.str2tuple(mtypes)
        ftype = ftype.upper()#RockPy.abbrev_to_classname[ftype.lower()]

        mtypes = tuple(RockPy.abbrev_to_classname[mtype.lower()] for mtype in RockPy.to_tuple(mtypes))
        return sgroups, snames, mtypes, ftype

    @staticmethod
    def extract_sample_block(block):
        """Sample block holds the sample information e.g. mass, height, diameter

        Args:
            block:
        """

        # has to be initialized
        out = [[None, None], [None, None], [None, None]]

        if "_" in block:
            warnings.warn("old style naming convention << 1,0[mg]_1,0[mm]_3,0[mm] >> use 1.0mg,2.0mm,3.0mm")
            # old style infos
            block = block.replace("[", "").replace("]", "")
            block = block.replace(",", ".")
            parts = block.split("_")
        else:
            parts = block.split(",")

        # extract (mass, massunit), (height, heightunit), (diameter,diameterunit)) from the block
        for i in range(3):
            try:
                p = parts[i]
                val = float(re.findall(r"[-+]?\d*\.\d+|\d+", p)[0])
                unit = "".join([i for i in p if not i.isdigit()]).strip(".")
            except IndexError:
                val = None
                unit = None
            out[i][0] = val
            out[i][1] = unit
        return out

    @staticmethod
    def extract_series_block(block):
        """Series block in the minfo

        Args:
            block:
        """

        def extract_series(s):
            """
            extracts a series tuple from a str
            """
            s = RockPy.core.utils.str2tuple(s)
            s = tuple([s[0], float(s[1]), s[2]])
            return s

        # translate oldstyle files
        # old style series block: e.g. mtime(h)_0,0_h;mtime(m)_0,0_min;speed_1100,0_rpm
        if not any(s in block for s in ("(", ")")) or ";" in block:
            warnings.warn(
                "old style series block: ""mtime(h)_0,0_h;height_10,0_m"" use: ""(mtime,0.0,h)_(height,10.0,m)"" ")
            block = block.translate(str.maketrans(",_;", ".,_", ""))

        # split block parts
        series = block.split("_")

        if not series:
            series = None

        try:
            return [extract_series(s) for s in series if s]
        except IndexError:
            return block

    @classmethod
    def extract_add_dialect_block(cls, block):
        """creates the add block. additional information is comma seperated.

        if dialect=XXX: this will then be passed

        Args:
            block:

        Returns:
            * *list of additionals*
            * *str, dialect*
        """
        if not 'dialect' in block:
            return block, ''
        parts = block.split(",")
        dialect = [p for p in parts if "dialect" in p][0].replace("dialect=", "")
        parts = [p for p in parts if "dialect" not in p]
        return parts, dialect

    @classmethod
    def from_folder(cls, folder, filter=None):
        """Creates a minfo (measurement info) instance for all files in a given
        directory.

        Args:
            folder (str): location of files
            filter: unused

        Returns:
            RockPy.minfo:
        """

        if filter == None:
            filter = dict()

        dfiles = [os.path.join(folder, i) for i in os.listdir(folder) if not i.startswith("#")]

        minfo = None
        for f in dfiles:
            # create minfo instance
            finfo = cls.from_file(f)

            if finfo is None:
                cls.log().debug("cant read file: %s" % os.path.basename(f))
                continue

            if minfo is None:
                minfo = finfo
            else:
                # append subsequent minfos
                minfo = minfo + finfo

        return minfo

    @classmethod
    def from_file(cls, fpath):
        """Reads a path into RockPy readable minfo structure :param fpath:

        Args:
            fpath:
        """
        cls.log().info("reading file infos: ... %s" % fpath[-30:])

        # # get the directory
        # folder = os.path.dirname(fpath)

        # get the file sname and the suffix
        filename, suffix = os.path.splitext(os.path.basename(fpath))

        # remove . from suffix
        suffix = suffix.strip(".")

        splits = filename.split("#")

        # check if RockPy compatible e.g. first part must be len(4)
        if not len(splits[0].split("_")) == 4:
            cls.log().debug('filename << %s >> does not conform to the RockPy file naming scheme. At least 3 elements (sname, mtype, ftype) '
                              'have to be given, separated by \'_\' '%filename)
            return

        # extract mesurement information
        sgroups, snames, mtypes, ftype = cls.extract_measurement_block(splits[0])

        # extract sample informations
        (mass, massunit), (height, heightunit), (diameter, diameterunit) = (None, None), (None, None), (None, None)
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
        if len(splits) > 3 and splits[3]:
            additional, dialect = cls.extract_add_dialect_block(splits[3])
        else:
            additional, dialect = (None, None)

        # comment
        if len(splits) > 4:
            comment = splits[4]
        else:
            comment = None

        return cls(snames=snames, mtypes=mtypes, ftype=ftype, fpath=fpath, sgroups=sgroups,
                   dialect=dialect,
                   mass=mass, massunit=massunit if massunit else "kg",
                   height=height, heightunit=heightunit if heightunit else "m",
                   diameter=diameter, diameterunit=diameterunit if diameterunit else "m",
                   series=series, comment=comment, additional=additional, suffix=suffix)

    @classmethod
    def from_dict(cls, **kwargs):
        """
        Args:
            **kwargs:
        """
        return cls(**{param: kwargs.get(param, None) for param in ("snames", "mtypes", "ftype", "fpath", "sgroups",
                                                                   "dialect",
                                                                   "mass", "massunit", "height", "heightunit",
                                                                   "diameter", "diameterunit", "lengthunit",
                                                                   "series", "comment", "additional", "suffix")})

    def __add__(self, other):
        """adds the importinfos of one class to the other.

        Args:
            other (ImportHelper istance):
        """

        for param in ("snames", "mtypes", "ftype", "fpath", "sgroups",
                      "dialect",
                      "mass", "massunit", "height", "heightunit", "diameter", "diameterunit", "lengthunit",
                      "series", "comment", "additional", "suffix"):
            getattr(self, param).extend(getattr(other, param))
        return self

    def __init__(
            self,
            snames,
            mtypes,
            ftype, fpath=None,
            sgroups=None,
            dialect=None,
            mass=None, massunit=None,
            height=None, heightunit=None,
            diameter=None, diameterunit=None,
            lengthunit=None,
            series=None,
            comment=None,
            additional=None,
            suffix=None,
            ):
        """constructor

        Args:
            snames:
            mtypes:
            ftype:
            fpath:
            sgroups:
            dialect:
            mass:
            massunit:
            height:
            heightunit:
            diameter:
            diameterunit:
            lengthunit:
            series:
            comment:
            additional:
            suffix:
        """
        self.snames = RockPy.core.utils.tuple2list_of_tuples(RockPy.to_tuple(snames))
        self.sgroups = RockPy.core.utils.tuple2list_of_tuples(RockPy.to_tuple(sgroups))

        self.mtypes = RockPy.core.utils.tuple2list_of_tuples(RockPy.to_tuple(mtypes))
        self.mtypes = [[RockPy.abbrev_to_classname[mt.lower()] for mt in mtypes] for mtypes in
                       self.mtypes]  # translate from abbrev to classname

        self.ftype = RockPy.core.utils.to_list(RockPy.to_tuple(ftype.lower()))

        self.fpath = RockPy.core.utils.to_list(fpath)
        self.dialect = RockPy.core.utils.to_list(dialect)

        #################################################################################
        """ PARAMETER """
        self.mass = RockPy.core.utils.to_list(mass)
        self.massunit = RockPy.core.utils.to_list(massunit)

        self.height = RockPy.core.utils.to_list(height)
        self.heightunit = RockPy.core.utils.to_list(heightunit)
        self.diameter = RockPy.core.utils.to_list(diameter)
        self.diameterunit = RockPy.core.utils.to_list(diameterunit)

        # in case of different units convert them to the same
        if self.heightunit != self.diameterunit:
            self.height *= RockPy.core.utils.convert(self.height, self.heightunit, self.diameterunit)
            self.heightunit = self.diameterunit

        if lengthunit is None:
            self.lengthunit = RockPy.core.utils.to_list(self.diameterunit)
        else:
            self.lengthunit = RockPy.core.utils.to_list(lengthunit)

        if series is not None:
            # if only one series and not three
            if len(series) == 3 and not len(series[0]) == 3:
                series = tuple2list_of_tuples(to_tuple(series))

        self.series = [series]

        self.comment = RockPy.core.utils.to_list(comment)
        self.additional = [to_tuple(additional)]
        self.suffix = RockPy.core.utils.to_list(suffix)

    @classmethod
    def get_measurement_block(cls, sgroups, snames, mtypes, ftype):
        """Method that joins the parts in the measurement block.

            (Samplegroups)_(snames)_(MTYPES)_ftype

        Examples:
            (SG1,SG2)_(S1,S2)_(HYS,DCD)_VSM

        Args:
            sgroups:
            snames:
            mtypes:
            ftype:

        Returns:
            str:
        """
        mtypes = (RockPy.classname_to_abbrev[i][0] for i in mtypes)
        block = [sgroups, snames, mtypes, ftype]
        if not all(i for i in block):
            raise ImportError("sname, mtype, ftype needed for minfo to be generated")
        return "_".join((RockPy.core.utils.tuple2str(b) for b in block))

    @classmethod
    def get_sample_block(cls,
                         mass, massunit,
                         height, heightunit,
                         diameter, diameterunit):
        """Method that joins the parts of the sample info block.

            (XXmg)_(XXmm)_(XXMM)

        In the case there is a mass, diameter and height associated

        Examples:
            (0.1mg)_(S1,S2)_(HYS,DCD)_VSM

        Args:
            mass:
            massunit:
            height:
            heightunit:
            diameter:
            diameterunit:

        Returns:
            str:
        """
        out = []
        block = [[mass, massunit], [height, heightunit], [diameter, diameterunit]]

        if not any((all(b) for b in block)):
            return ""

        # get the point for which there is still info
        # 1.0mg -> n=1
        # 1.2mg,2.0mm ->n=3
        # XXmg,XXmm,1.2mm -> n=3
        n = None

        for n, v in enumerate(block):
            if v[0] == v[1]:
                break

        for i in range(n):
            b = block[i]
            if not all(j for j in b):
                if i == 0:
                    aux = "XXmg"
                else:
                    aux = "XXmm"
            else:
                out.append("".join(map(str, b)))
        return ",".join(out)

    @classmethod
    def get_series_block(cls, series):
        """Method that joins the series parts of the sample.

            (stype1,svalue1,sunit1)_(stype2, sval2, sunit2)

        Args:
            series:

        Returns:
            str:
        """
        import numpy as np
        if series:
            if np.array(series).shape == (3,):
                out = [RockPy.core.utils.tuple2str(series)]

            else:
                out = [RockPy.core.utils.tuple2str(b) for b in series]
            return "_".join(out).replace("\"", "").replace(" ", "")
        else:
            return ""

    @classmethod
    def get_add_block(cls, additional):
        """
        Args:
            additional:
        """
        if additional:
            return RockPy.core.utils.tuple2str(additional)

    @property
    def new_filenames(self):
        out = []
        for f in range(self.nfiles):
            measurement_block = self.get_measurement_block(self.sgroups[f], self.snames[f], self.mtypes[f],
                                                           self.ftype[f])
            sample_block = self.get_sample_block(self.mass[f], self.massunit[f],
                                                 self.diameter[f], self.diameterunit[f],
                                                 self.height[f], self.heightunit[f])
            series_block = self.get_series_block(self.series[f])
            add_block = self.get_add_block(self.additional[f])  # todo implement

            blocks = [measurement_block, sample_block, series_block, add_block]
            # work through blocks backwards throw out blocks that are empty,
            # if ones is not empty, stop.
            for i, block in enumerate(blocks[::-1]):
                if not block:
                    blocks.pop()
                else:
                    break

            suffix = self.suffix[f]

            if not suffix:
                suffix = 0

            if isinstance(suffix, (int, float)):
                suffix = "%03.0f" % suffix
            fname = "#".join(map(str, blocks)) + ".%s" % suffix

            fname = fname.replace("None", "")
            out.append(fname)
        return out

    def getImportHelper(self, snames=None, mtypes=None):
        """Generates ImportInfo instances for each sample, mtype.

        Args:
            snames (str): filters the generator to only create the passed snames
            mtypes (str): filters the generator to only create matching mtypes

        Returns:
            yields:
        """

        snames = RockPy.to_tuple(snames)
        mtypes = RockPy.to_tuple(mtypes)

        for ih in self._gen_dicts:
            if all(i for i in snames) and ih["snames"] not in snames:
                continue
            if all(i for i in mtypes) and ih["mtypes"] not in mtypes:
                continue
            a = self.__class__.from_dict(**ih)
            yield a

    @property
    def nsnames(self):
        return len(self.snames)

    @property
    def nfiles(self):
        return len(self.fpath) if self.fpath[0] != None else 1

    @property
    def _gen_dicts(self):
        """generator that returns a dictionary for all samples of the instance

        Returns:
            dict: **generator**
        """
        for f in range(self.nfiles):
            for sname in self.snames[f]:
                if sname is None:
                    continue
                for mtype in self.mtypes[f]:
                    if mtype is None:
                        continue
                    yield dict(snames=sname, mtypes=mtype, sgroups=self.sgroups[f],
                               mass=self.mass[f], massunit=self.massunit[f],
                               diameter=self.diameter[f], height=self.height[f],
                               diameterunit=self.diameterunit[f], heightunit=self.heightunit[f],
                               lengthunit=self.lengthunit[f],
                               # file path and file type
                               fpath=self.fpath[f], ftype=self.ftype[f], dialect=self.dialect[f],
                               idx=self.suffix[f],
                               series=self.series[f],
                               comment=self.comment[f], )

    @property
    def gen_measurement_dict(self):
        for ih in self._gen_dicts:
            ih["sname"] = ih.pop("snames")
            ih["mtype"] = ih.pop("mtypes")
            yield ih

    @property
    def gen_sample_dict(self):
        """generator returns dictionary with all sample infos for each sample in
        the instance

        Returns:
            dict: **generator**
        """
        generated = []
        for ih in self._gen_dicts:
            if ih["snames"] in generated:
                continue
            ih["sname"] = ih.pop("snames")
            ih["mtype"] = ih.pop("mtypes")
            generated.append(ih["sname"])
            yield {k: v for k, v in ih.items() if k in ("sname",
                                                        "mass", "massunit",
                                                        "height", "diameter", "lengthunit",
                                                        "sgroups", "series")} #todo check if adding series here broke stuff

    def return_file_infos(self):
        """Returns a dictionary with the sample infos of the ImportHelper with
        the sname as keys.

        Returns:
            dict:
        """
        out = {}

        out['snames'] = list_or_item(self.snames[0])
        out['sgroups'] = list_or_item(self.sgroups[0])
        out['mtypes'] = list_or_item(self.mtypes[0])
        out['ftype'] = self.ftype[0]
        out['fpath'] = self.fpath[0]
        out['dialect'] = self.dialect[0]

        out['mass'] = self.mass[0]
        out['mass_unit'] = self.massunit[0]

        out['height'] = self.height[0]
        out['height_unit'] = self.heightunit[0]

        out['diameter'] = self.diameter[0]
        out['diameter_unit'] = self.diameterunit[0]

        out["series"] = {}

        if self.series[0]:
            for s in self.series[0]:
                out["series"].update(series_to_dict(s))

        out['comment'] = self.comment[0]
        out['additional'] = self.additional[0]
        out['suffix'] = self.suffix[0]


        return out

def connect_to_IRMdb():
    pass


if __name__ == '__main__':
    ih = ImportHelper.from_file('/Users/mike/Dropbox/science/_projects/hexagonal_pyrrhotite/data/HYS/Phexa_MSM16667-6a_HYS_VSM##(T,200,C)_(cool,1,).026')
    print(ih)