import ast
import json
from typing import Dict
from astutils.misc import ducktype_ast_class


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


def ast2dict(ast_tree: ast.AST) -> Dict:
    """Takes in input an Abstract Syntax Tree and return a Dictionary based representation of the code. **WARNING:** the tabulation information will be lost. If these fields are important for you, check ```ast2heap```.

    Args:
        ast_tree (ast.AST): An Abstract Syntax Tree representing a Python program.

    Returns:
        Dict: A Dictionary represintig the input Abstract Syntax Tree.
    """

    if isinstance(ast_tree, ast.AST):

        class_name = ast_tree.__class__.__name__

        if len(ast_tree._fields) > 0:
            ret = {class_name: {}}
            for field in ast_tree._fields:
                ret[class_name][field] = ast2dict(ast_tree.__dict__[field])
        else:
            ret = class_name

    elif isinstance(ast_tree, list):
        ret = []
        for element in ast_tree:
            ret.append(ast2dict(element))
    else:
        ret = ast_tree

    return ret


def ast2json(ast_tree: ast.AST) -> json:
    """Takes in input an Abstract Syntax Tree representing a Python program and return a Json string.

    Args:
        ast_tree (ast.AST): An Abstract Syntax Tree representing a Python program.

    Returns:
        json: A Json string representing the input Python program.
    """

    return json.dumps(ast2dict(ast_tree))
