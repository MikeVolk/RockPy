import RockPy
import  RockPy.core.utils as core_utils

def __implemented__(cls):  # todo move into RockPy core has nothing to do with measurement
    """Dictionary of all implemented filetypes.

    Looks for all subclasses of RockPy3.core.ftype.ftypes generating a
    dictionary of implemented machines : {implemented out_* method :
    machine_class}

    Args:
        cls:

    Returns:
        classname:: **dict**
    """
    implemented = {cl.__name__.lower(): cl for cl in core_utils.extract_inheritors_from_cls(cls)}
    return implemented