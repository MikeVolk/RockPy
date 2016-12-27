from contextlib import contextmanager


@contextmanager
def ignored(*exceptions):
    """
    ignores certain exceptions

    Parameters
    ----------
    exceptions: errors
        the errors to be ignored

    """
    try:
        yield
    except exceptions:
        pass


def tuple2list_of_tuples(item) -> list:
    """
    Takes a list of tuples or a tuple and returns a list of tuples

    Parameters
    ----------
       item: list, tuple

    Returns
    -------
       list
          Returns a list of tuples, if data is a tuple it converts it to a list of tuples
          if data == a list of tuples will just return data
    """

    # check if item is a list -> each item in item has to be converted to a tuple
    if isinstance(item, list):
        for i, elem in enumerate(item):
            if not type(elem) == tuple:
                item[i] = (elem,)

    if not isinstance(item, (list, tuple)):
        item = tuple([item])

    if isinstance(item, tuple):
        item = [item, ]

    return item


def _to_tuple(oneormoreitems):
    """
    convert argument to tuple of elements

    Parameters
    ----------
        oneormoreitems: single number or string or list of numbers or strings

    Returns
    -------
        tuple of elements
    """
    return tuple(oneormoreitems) if hasattr(oneormoreitems, '__iter__') and type(oneormoreitems) is not str else (
        oneormoreitems,)


def extract_tuple(s: str) -> tuple:
    """
    Extracts a tuple from a string, brackets ('[]()') are removed first

    e.g. "(HYS, COE)" -> ('hys','coe')
    e.g. "[HYS, COE]" -> ('hys','coe')

    Parameters
    ----------
    s str
        string to be tupeled

    Returns
    -------
        tuple
    """
    s = s.translate(str.maketrans("", "", "(){}[]")).split(',')
    return tuple(s)


def split_num_alph(item):
    '''
    splits a string with numeric and str values into a float and a string

    Parameters
    ----------
    item: str
        The string that to be split

    Returns
    -------
        float, str
    '''
    # replace german decimal comma
    item.replace(',', '.')

    # cycle through all items in the string and stop at the first non numeric
    for i, v in enumerate(item):
        if not v in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.'):
            break

    if not i == len(item) - 1:
        return float(item[:i]), item[i:].strip()
    else:
        return float(item), None


if __name__ == '__main__':
    print(split_num_alph('20.3'))
