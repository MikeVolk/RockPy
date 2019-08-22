from RockPy.core.utils import extract_inheritors_from_cls
from RockPy.core.ftype import Ftype


def __implemented__(cls):  # todo move into RockPy core has nothing to do with measurement
    """
    Dictionary of all implemented filetypes.

    Looks for all subclasses of RockPy3.core.ftype.ftypes
    generating a dictionary of implemented machines : {implemented out_* method : machine_class}

    Returns
    -------

    dict: classname:
    """
    implemented = {cl.__name__.lower(): cl for cl in extract_inheritors_from_cls(cls)}
    return implemented


if __name__ == '__main__':
    from RockPy.core.measurement import Measurement
    print(__implemented__(Ftype))
    print(__implemented__(Measurement))