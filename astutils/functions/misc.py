import ast
from typing import Dict
import time


def _read(fn, *args):
    kwargs = {"encoding": "iso-8859-1"}
    with open(fn, *args, **kwargs) as f:
        return f.read()


def _measure_time(func, *args, **kwargs):
    """Utility function"""

    time_beg = time.time()
    ret = func(*args, **kwargs)
    time_end = time.time()

    return ret, time_end - time_beg


def ducktype_ast_class(d: Dict, name: str) -> bool:
    """Takes in input a dictionary, and check if it could be converted to and ast.AST subclass.

    Args:
        d (Dict): A Dictionary.
        name (str): The name of the field.

    Returns:
        bool: flag
    """

    def is_sublist(lst1, lst2):
        def get_all_in(one, another):
            for element in one:
                if element in another:
                    yield element

        for x1, x2 in zip(get_all_in(lst1, lst2), get_all_in(lst2, lst1)):
            if x1 != x2:
                return False
        return True

    if isinstance(d[name], dict) and name in dir(ast):
        if is_sublist(d[name].keys(), getattr(ast, name)._fields):
            return True
    return False
