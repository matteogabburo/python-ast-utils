import ast
from typing import Tuple
import sys
from astutils.functions.misc import _read


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

    return ast.parse(
        _read(filename),
        mode=mode,
        type_comments=type_comments,
        feature_version=feature_version,
    )


def ast_parse_from_string(
    source: str,
    input_text: str = None,
    mode: str = "exec",
    type_comments: bool = False,
    feature_version: Tuple[int, int] = None,
) -> ast.AST:
    """Takes in input the path to a Python file, and return an Abstract Syntax Tree.

    Args:
        source (str): A string representing a Python source code.
        mode (str, optional): Can be 'exec' or 'func_type'. if mode is 'func_type', the input syntax is modified to correspond to PEP 484 “signature type comments”. Defaults to 'exec'.
        type_comments (bool, optional): If type_comments=True is given, the parser is modified to check and return type comments as specified by PEP 484 and PEP 526. Defaults to False.
        feature_version (Tuple[int, int], optional): A tuple (major, minor) (for example (3, 4)) used to decide which version of the Python grammar to use during the parsing. Defaults to None.

    Returns:
        ast.AST: The Abstract Syntax Tree obtained by the input Python file.
    """

    return ast.parse(
        source, mode=mode, type_comments=type_comments, feature_version=feature_version
    )


def ast_unparse(ast_tree: ast.AST) -> str:
    """Takes in input an Abstract Syntax Tree and return the string representation of the code.

    Args:
        ast_tree (ast.AST): An Abstract Syntax Tree representing a Python program.

    Raises:
        Exception: If the current python version is < (3,9).

    Returns:
        str: A string that represent the code associated with the input Abstract Syntax Tree.
    """
    if sys.version_info >= (3, 9):
        return ast.unparse(ast_tree)
    else:
        raise Exception("Not implemented in Python version < 3.9")
