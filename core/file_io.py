import RockPy


class minfo():
    @staticmethod
    def extract_tuple(s):
        s = s.strip('(').strip(')').split(',')
        return tuple(s)

    def extract_series(self, s):
        s = self.extract_tuple(s)
        s = tuple([s[0], float(s[1]), s[2]])
        return s

    @staticmethod
    def tuple2str(tup):
        """
        takes a tuple and converts it to text, if more than one element, brackets are put around it
        """
        if tup is None:
            return ''

        tup = RockPy._to_tuple(tup)

        # if type(tup) == list:
        #     if len(tup) == 1:
        #         tup = tup[0]
        #     else:
        #         tup = tuple(tup)
        if len(tup) == 1:
            return str(tup[0])
        else:
            return str(tup).replace('\'', ' ').replace(' ', '')

    def measurement_block(self, block):
        sgroups, samples, mtypes, ftype = block.split('_')
        # names with , need to be replaced
        if not '(' in samples and ',' in samples:
            samples = samples.replace(',', '.')
            RockPy3.logger.warning('sample name %s contains \',\' will be replaced with \'.\'' % samples)

        self.sgroups, self.samples, self.mtypes, self.ftype = self.extract_tuple(sgroups), self.extract_tuple(
            samples), self.extract_tuple(mtypes), ftype
        self.mtypes = tuple(RockPy3.abbrev_to_classname(mtype) for mtype in RockPy3._to_tuple(self.mtypes))
        self.ftype = RockPy3.abbrev_to_classname(ftype)

    def sample_block(self, block):
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
        # old style series block: e.g. mtime(h)_0,0_h;mtime(m)_0,0_min;speed_1100,0_rpm
        if not any(s in block for s in ('(', ')')) or ';' in block:
            block = block.replace(',', '.')
            block = block.replace('_', ',')
            block = block.replace(';', '_')

        series = block.split('_')
        if not series:
            self.series = None
        self.series = [self.extract_series(s) for s in series if s]

    def add_block(self, block):
        if block:
            self.additional = block
        else:
            self.additional = ''

    def comment_block(self, block):
        self.comment = block

    def get_measurement_block(self):
        block = deepcopy(self.storage[0])
        block[2] = [abbreviate_name(mtype).upper() for mtype in RockPy3._to_tuple(block[2]) if mtype]
        block[3] = abbreviate_name(block[3]).upper()
        if not all(block[1:]):
            raise ImportError('sname, mtype, ftype needed for minfo to be generated')
        return '_'.join((self.tuple2str(b) for b in block))

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
            out = [self.tuple2str(b) for b in block]
            return '_'.join(out)

    def get_add_block(self):
        if self.additional:
            out = tuple(''.join(map(str, a)) for a in self.additional)
            return self.tuple2str(out)

    def is_readable(self):
        if not os.path.isfile(self.fpath):
            return False
        if all(self.storage[0][1:]):
            return True
        else:
            return False

    def __init__(self, fpath,
                 sgroups=None, samples=None,
                 mtypes=None, ftype=None,
                 mass=None, height=None, diameter=None,
                 massunit=None, lengthunit=None, heightunit=None, diameterunit=None,
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
        massunit
        lengthunit
        heightunit
        diameterunit
        series
        comment
        folder
        suffix
        read_fpath: bool
            if true the path will be read for info
        kwargs
        """
        if 'mtype' in kwargs and not mtypes:
            mtypes = kwargs.pop('mtype')
        if 'sgroup' in kwargs and not sgroups:
            mtypes = kwargs.pop('sgroup')
        if 'sample' in kwargs and not samples:
            mtypes = kwargs.pop('sample')

        blocks = (self.measurement_block, self.sample_block, self.series_block, self.add_block, self.comment_block)
        additional = tuple()

        sgroups = RockPy3._to_tuple(sgroups)
        sgroups = tuple([sg if sg != 'None' else None for sg in sgroups])

        if mtypes:
            mtypes = tuple(RockPy3.abbrev_to_classname(mtype) for mtype in RockPy3._to_tuple(mtypes))
        if ftype:
            ftype = RockPy3.abbrev_to_classname(ftype)

        self.__dict__.update({i: None for i in ('sgroups', 'samples', 'mtypes', 'ftype',
                                                'mass', 'height', 'diameter',
                                                'massunit', 'lengthunit', 'heightunit', 'diameterunit',
                                                'series', 'additional', 'comment', 'folder', 'suffix')
                              })
        self.fpath = fpath

        if read_fpath and fpath:  # todo add check for if path is readable
            self.folder = os.path.dirname(fpath)
            f, self.suffix = os.path.splitext(os.path.basename(fpath))
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
        for i in ('sgroups', 'samples', 'mtypes', 'ftype',
                  'mass', 'height', 'diameter',
                  'massunit', 'lengthunit', 'heightunit', 'diameterunit',
                  'series', 'additional', 'comment', 'folder'):

            if locals()[i]:
                if isinstance(locals()[i], (tuple, list, set)):
                    if not all(locals()[i]):
                        continue
                setattr(self, i, locals()[i])

        if self.additional is None:
            self.additional = ''
        if kwargs:
            for k, v in kwargs.items():
                if v:
                    print(k, v, self.additional)
                    self.additional += '{}:{}'.format(k, v)

        if suffix:
            self.suffix = suffix

        if type(self.suffix) == int:
            self.suffix = '%03i' % self.suffix

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
        name after new RockPy3 convention
        """

        # if not self.fpath:
        #     RockPy3.logger.error('%s is not a file' %self.get_measurement_block())
        #     return
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
        idict = {'fpath': self.fpath, 'ftype': self.ftype, 'idx': self.suffix, 'series': self.series}
        samples = RockPy3._to_tuple(self.samples)
        for i in samples:
            for j in self.mtypes:
                mtype = RockPy3.abbrev_to_classname(j)
                idict.update({'mtype': mtype, 'sample': i})
                yield idict

    @property
    def sample_infos(self):
        sdict = dict(mass=self.mass, diameter=self.diameter, height=self.height,
                     mass_unit=self.massunit, height_unit=self.heightunit, diameter_unit=self.diameterunit,
                     samplegroup=self.sgroups)

        samples = RockPy3._to_tuple(self.samples)
        for i in samples:
            sdict.update({'name': i})
            yield sdict
