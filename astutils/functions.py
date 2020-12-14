import ast
import json
from typing import Tuple, Dict


def _read(fn, *args):
    kwargs = {"encoding": "iso-8859-1"}
    with open(fn, *args, **kwargs) as f:
        return f.read()


def ast_parse(
    filename: str,
    mode: str = "exec",
    type_comments: bool = False,
    feature_version: Tuple[int, int] = None,
) -> ast.AST:
    """Takes in input the path to a Python file, and return an Abstract Syntax Tree.

    Args:
        filename (str): Path to a Python file
        mode (str, optional): Can be 'exec' or 'func_type'. if mode is 'func_type', the input syntax is modified to correspond to PEP 484 “signature type comments”. Defaults to 'exec'.
        type_comments (bool, optional): If type_comments=True is given, the parser is modified to check and return type comments as specified by PEP 484 and PEP 526. Defaults to False.
        feature_version (Tuple[int, int], optional): A tuple (major, minor) (for example (3, 4)) used to decide which version of the Python grammar to use during the parsing. Defaults to None.

    Returns:
        ast.AST: The Abstract Syntax Tree obtained by the input Python file.
    """

    return ast.parse(_read(filename))


def ast_unparse(ast_tree: ast.AST) -> str:
    """Takes in input an Abstract Syntax Tree and return the string representation of the code.

    Args:
        ast_tree (ast.AST): An Abstract Syntax Tree representing a Python program.

    Returns:
        str: A string that represent the code associated with the input Abstract Syntax Tree.
    """

    return ast.unparse(ast_tree)


def ast2dict(ast_tree: ast.AST) -> Dict:
    """Takes in input an Abstract Syntax Tree and return a Dictionary based representation of the code.

    Args:
        ast_tree (ast.AST): An Abstract Syntax Tree representing a Python program.

    Returns:
        Dict: A Dictionary represintig the input Abstract Syntax Tree.
    """

    if isinstance(ast_tree, ast.AST):

        class_name = ast_tree.__class__.__name__
        ret = {class_name: {}}
        for field in ast_tree._fields:
            ret[class_name][field] = ast2dict(ast_tree.__dict__[field])

    elif isinstance(ast_tree, list):
        ret = []
        for element in ast_tree:
            ret.append(ast2dict(element))
    else:
        ret = ast_tree

    return ret


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


def dict2ast(d: Dict) -> ast.AST:
    """Takes in input a Dictionary representing a Python program and return an Abstract Syntax Tree.

    Args:
        d (Dict): A Dictionary representing a Python program.

    Raises:
        Exception: If the input structure is wrong.

    Returns:
        ast.AST: The Abstract Syntax Tree obtained by the input Python file.
    """

    if isinstance(d, dict):
        ret = []
        for k in d:
            if ducktype_ast_class(d, k):

                Class = getattr(ast, k)
                kwargs = {}
                for subk in d[k]:
                    kwargs[subk] = dict2ast(d[k][subk])
                ret.append(Class(**kwargs))
            else:
                raise Exception
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    elif isinstance(d, list):
        ret = []
        for element in d:
            ret.append(dict2ast(element))
        return ret

    else:
        return d


def dict2json(d: Dict) -> json:
    """Takes in input a Dictionary representing a Python program and return a Json string.

    Args:
        d (Dict): A Dictionary representing a Python program.

    Returns:
        json: A Json string representing the input Python program.
    """

    return json.dumps(d)


def ast2json(ast_tree: ast.AST) -> json:
    """Takes in input an Abstract Syntax Tree representing a Python program and return a Json string.

    Args:
        ast_tree (ast.AST): An Abstract Syntax Tree representing a Python program.

    Returns:
        json: A Json string representing the input Python program.
    """

    return json.dumps(ast2dict(ast_tree))
