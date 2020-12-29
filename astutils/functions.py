import ast
import json
from typing import Tuple, Dict, List
import re
from tokenize import Number
import copy
import sys
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


# heap global strings
HEAP_ID = "_heap_id"
HEAP_CHILDREN = "_heap_children"
HEAP_TYPE = "_heap_type"
HEAP_VALUE = "_heap_value"
HEAP_CODE = "_heap_code"
HEAP_PLACEHOLDER = "_heap_placeholder"
CHILD_PLACEHOLDER = "<child={}><dlihc>"
NUMBER_PLACEHOLDER = "<num={}><mun>"
HEAP_TOKENS = "_heap_tokens"
TMP_SEQ_SEPARATOR = ">#@>@@§"
SOURCE_MAP_CHUNKS_SIZE = 128


def build_source_map(source: str) -> Dict[int, str]:
    """Generate an efficent map of an input source code to be used from ```get_source_segment```.

    Args:
        source (str, optional): The original code used to build the AST. If not None the final heap will contain a field with the original code associated with the node.

    Returns:
        Dict[int, str]: The map of the code.
    """

    lines = source.split("\n")
    source_map = {}
    for lineno, line in enumerate(lines):

        source_map[lineno + 1] = [line[i:i+SOURCE_MAP_CHUNKS_SIZE] for i in range(0, len(line), SOURCE_MAP_CHUNKS_SIZE)]

        if len(source_map[lineno + 1]) > 0 and len(source_map[lineno + 1][-1]) < SOURCE_MAP_CHUNKS_SIZE:
            source_map[lineno + 1][-1] += "\n"
        else:
            source_map[lineno + 1].append("\n")
        
        
    return source_map


def get_source_segment(
    source_map: Dict[int, str], tree: ast.AST, padded: bool = True
) -> str:
    """Given an ast tree and a source_map generated by ```build_source_map```, return the code associated to the ast tree.

    Args:
        source_map (Dict[int,str]): The map of the code generated by ```build_source_map```.
        tree (ast.AST): An Abstract Syntax Tree representing a Python program.
        padded (bool, optional): If true the output string will contain the tabulation. Defaults to True.

    Raises:
        Exception: Raised if the input tree is not an ast.AST instance

    Returns:
        str: A string that represents the input tree.
    """

    def _get_tokens(line, col_offset=None, end_col_offset=None):

        
        buck_col_offset = 0
        buck_end_col_offset = len(line) - 1

        # get the right x of the source_map
        if col_offset is not None:
            buck_col_offset, col_offset = col_offset // SOURCE_MAP_CHUNKS_SIZE,  col_offset % SOURCE_MAP_CHUNKS_SIZE

        if end_col_offset is not None:
            buck_end_col_offset, end_col_offset =  end_col_offset // SOURCE_MAP_CHUNKS_SIZE, end_col_offset % SOURCE_MAP_CHUNKS_SIZE 

        pieces = [line[i] for i in range(buck_col_offset, buck_end_col_offset+1)] # line[buck_col_offset: buck_end_col_offset+1]

        if col_offset:
            pieces[0] = pieces[0][col_offset:]
        
        if end_col_offset:
            pieces[-1] = pieces[-1][:end_col_offset]

        return ''.join(pieces)
    
    if isinstance(tree, ast.AST):

        dict_tree = tree.__dict__
        if "lineno" in dict_tree:

            lineno = dict_tree["lineno"]
            end_lineno = dict_tree["end_lineno"]
            col_offset = dict_tree["col_offset"]
            end_col_offset = dict_tree["end_col_offset"]

            if end_lineno == lineno:
                return _get_tokens(source_map[lineno], col_offset, end_col_offset)

            ret = [source_map[i][0] for i in range(lineno, end_lineno+1)]
            
            if not padded:
                ret[0] = _get_tokens(ret[0], col_offset=col_offset)
            else:
                span = _get_tokens(ret[0], col_offset=col_offset)
                if not span.isspace():
                    ret[0] = " " * col_offset + span

            ret[-1] = _get_tokens(ret[-1], end_col_offset=end_col_offset)
                
            return "".join(ret)

        else:
            return None
    else:
        raise Exception

    return None


def ast2heap(
    ast_tree: ast.AST,
    source: str = None,
    positional: bool = True,
    not_considered_leaves: List = [],
) -> List:
    """Takes in input an Abstract Syntax Tree representing a Python program and return it represented with an heap structure. The resulting structure is defined by a list of nodes. Each node is composed as follows:

    ```
    heap_node = {
        _heap_id: (int) the "id" of the node
        _heap_type: (str) generally it is a non-terminal of the grammar (grammar reference https://docs.python.org/3/library/ast.html)
        _heap_value: [Optional] (dict) a dictionary containing the values of an ast node (generally can be both a terminal and a non-terminal of the grammar). 
        _heap_children: [Optional] (list[int]) a list of "_heap_ids". Each id contained in this list is a children of this node
        _heap_code: [Optional] (str) contain the source code associate with this specific node. Initialized only if the source parameter is initialized
    }
    ```

    Args:
        ast_tree (ast.AST): An Abstract Syntax Tree representing a Python program.
        source (str, optional): The original code used to build the AST. If not None the final heap will contain a field with the original code associated with the node.
        positional (str, optional): Considered only if the source parameter is initialized. When positional is True, the field ```_heap_code``` of the final model will contain alias that replace the original value of each subtoken (For example, ```@11``` is a placeolder for the node with ```_heap_id=11```).
        not_considered_leaves (list, optional): A list containing all the not desidered types. This could be used to reduce the heap size keeping only the wanted nodes. Defaults to [].

    Returns:
        List: An heap representing the input AST.
    """

    CHILD_PLACEHOLDER_BEG = CHILD_PLACEHOLDER.split('{')[0]

    def _build_heap(
        tree: ast.AST,
        source,
        source_map,
        positional,
        heap,
        not_considered_leaves,
        field,
        parent,
    ):

        if isinstance(tree, ast.AST):

            node_id = len(heap)

            # if the current tree is not the root
            if parent:
                if HEAP_CHILDREN not in parent:
                    parent[HEAP_CHILDREN] = []
                parent[HEAP_CHILDREN].append(node_id)

            class_name = tree.__class__.__name__
            heap_node = {HEAP_ID: node_id, HEAP_TYPE: class_name}

            if source:

                heap_node[HEAP_CODE] = get_source_segment(source_map, tree, padded=True)
                if positional:

                    # conditions
                    def is_node_abstract(node):
                        return node[HEAP_CODE] is None

                    def is_not_root(heap):
                        return len(heap) > 0

                    def is_prec_node_abstract(node):
                        return HEAP_PLACEHOLDER in node

                    def __put_code_placeholder(source, to_replace, node_id):

                        def __find_separator(m):
                            return m.group(0) + TMP_SEQ_SEPARATOR

                        code_l = source.split(CHILD_PLACEHOLDER_BEG)

                        code_l[-1] = code_l[-1].replace(
                            to_replace, CHILD_PLACEHOLDER.format(node_id), 1
                        )

                        return CHILD_PLACEHOLDER_BEG.join(code_l)

                    # handle numbers to avoid problems during the tree construction
                    def __replace(m):
                        return NUMBER_PLACEHOLDER.format(m.group(0))

                    if heap_node[HEAP_CODE]:
                        heap_node[HEAP_CODE] = re.sub(
                            Number, __replace, heap_node[HEAP_CODE]
                        )

                    if not is_not_root(heap):
                        heap_node[HEAP_CODE] = re.sub(Number, __replace, source)
                        

                    elif is_node_abstract(heap_node) and is_not_root(heap):

                        if not is_node_abstract(parent):
                            heap_node[HEAP_PLACEHOLDER] = parent[HEAP_CODE]
                        else:
                            assert parent[HEAP_PLACEHOLDER]
                            heap_node[HEAP_PLACEHOLDER] = parent[HEAP_PLACEHOLDER]

                    elif is_not_root(heap) and is_prec_node_abstract(heap):
                        assert parent[HEAP_PLACEHOLDER]
                        # replace the first occurrence of the current text in the placeholder
                        parent[HEAP_PLACEHOLDER] = __put_code_placeholder(
                            parent[HEAP_PLACEHOLDER], heap_node[HEAP_CODE], node_id
                        )

                    else:
                        # replace the first occurrence of the current text in the code
                        assert HEAP_CODE in parent
                        if parent[HEAP_CODE]:
                            parent[HEAP_CODE] = __put_code_placeholder(
                                parent[HEAP_CODE], heap_node[HEAP_CODE], node_id
                            )
                        else:
                            parent[HEAP_PLACEHOLDER] = __put_code_placeholder(
                                parent[HEAP_PLACEHOLDER], heap_node[HEAP_CODE], node_id
                            )

            heap.append(heap_node)

            if len(tree._fields) > 0:
                for field in tree._fields:
                    if field not in not_considered_leaves:
                        _build_heap(
                            tree.__dict__[field],
                            source,
                            source_map,
                            positional,
                            heap,
                            not_considered_leaves,
                            field,
                            heap_node,
                        )

        elif isinstance(tree, tuple) or isinstance(tree, list):
            for element in tree:
                _build_heap(
                    element,
                    source,
                    source_map,
                    positional,
                    heap,
                    not_considered_leaves,
                    field,
                    parent,
                )

        else:
            # append to the last inserted node
            if HEAP_VALUE not in heap[-1]:
                heap[-1][HEAP_VALUE] = {}
            heap[-1][HEAP_VALUE][field] = tree

    source_map = None
    if source:
        # build the source map
        source_map = build_source_map(source)

    heap = []
    _build_heap(
        ast_tree,
        source,
        source_map,
        positional,
        heap,
        not_considered_leaves,
        None,
        None,
    )

    return heap


def scompone(heap: list) -> list:
    """Given an heap representation of a tree generated by ```ast2heap```, return the list of subheaps where each sub heap correspond to a block of code starting without identation.

    Args:
        heap (list): A heap generated with ```astutils.ast2heap()```.

    Returns:
        list: A list of subheaps.
    """

    # make a copy of the input heap and work on it
    _heap = copy.deepcopy(heap)

    def _dfs_build(heap, root):

        sub_heap = [root]

        if HEAP_CHILDREN in root:
            for child_id in root[HEAP_CHILDREN]:
                sub_heap += _dfs_build(heap, heap[child_id])

        return sub_heap

    sub_heaps = []
    if HEAP_CHILDREN in _heap[0]:
        for child_id in _heap[0][HEAP_CHILDREN]:
            sub_heaps.append(_dfs_build(_heap, _heap[child_id]))

    if len(sub_heaps) == 0:
        # no sub heaps
        return [_heap]
    else:
        return sub_heaps


def heap2code(heap: list, inplace: bool = False) -> str:
    """Given an heap generated by ```ast2heap```, return a string representing the original code from wich the heap was originated.

    Args:
        heap (list): A heap generated with ```astutils.ast2heap()``` with ```source not None``` and ```positional == True```.
        inplace (bool, optional): If True, the input heap will be modified by the function. Default to False.

    Returns:
        list: A string representing the original code.
    """

    NUMBER_PLACEHOLDER_BEG = NUMBER_PLACEHOLDER.split("{")[0]
    NUMBER_PLACEHOLDER_END = NUMBER_PLACEHOLDER.split("}")[1]

    if not inplace:
        # make a copy of the input heap and work on it
        _heap = copy.deepcopy(heap)
    else:
        _heap = heap

    def _dfs_build(root, heap, id_offset, parent):

        # if the heap was generated from an empty file, the heap will contains only the root node without the field "HEAP_CODE"
        if HEAP_CODE not in root and len(heap) == 1:
            return ""

        assert HEAP_CODE in root

        def has_children(node):
            return HEAP_CHILDREN in node

        if has_children(root):

            for heap_node_id in root[HEAP_CHILDREN]:

                # add the offset
                offsetted_node_id = heap_node_id - id_offset
                assert heap[offsetted_node_id][HEAP_ID] == heap_node_id

                partial_source = _dfs_build(
                    heap[offsetted_node_id], heap, id_offset, root
                )

                if root[HEAP_CODE]:
                    root[HEAP_CODE] = root[HEAP_CODE].replace(
                        CHILD_PLACEHOLDER.format(heap_node_id), partial_source, 1
                    )
                else:
                    assert root[HEAP_PLACEHOLDER]
                    root[HEAP_PLACEHOLDER] = root[HEAP_PLACEHOLDER].replace(
                        CHILD_PLACEHOLDER.format(heap_node_id), partial_source, 1
                    )
        else:
            # is a leaf
            pass

        # handle numbers to avoid problems during the tree construction
        def __replace(m):
            return m.group(0)[
                len(NUMBER_PLACEHOLDER_BEG) : -len(NUMBER_PLACEHOLDER_END)
            ]

        if root[HEAP_CODE]:
            return re.sub(NUMBER_PLACEHOLDER.format(Number), __replace, root[HEAP_CODE])
        assert root[HEAP_PLACEHOLDER]
        return re.sub(
            NUMBER_PLACEHOLDER.format(Number), __replace, root[HEAP_PLACEHOLDER]
        )

    assert isinstance(heap, list)
    return _dfs_build(_heap[0], _heap, _heap[0][HEAP_ID], None)


def heap2tokens(heap: list, inplace: bool = False) -> list:
    """Given an heap generated by ```ast2heap```, return a list of tuple containing the ordered list of tokens and their indexes and their grammar type.

    Args:
        heap (list): A heap generated with ```astutils.ast2heap()``` with ```source not None``` and ```positional == True```.
        inplace (bool, optional): If True, the input heap will be modified by the function. Default to False.

    Returns:
        list: A list of tuple where each tuple is ```(segment of code, ast_node_id, ast_node_type)```.
    """

    NUMBER_PLACEHOLDER_BEG = NUMBER_PLACEHOLDER.split("{")[0]
    NUMBER_PLACEHOLDER_END = NUMBER_PLACEHOLDER.split("}")[1]
    CHILD_PLACEHOLDER_BEG = CHILD_PLACEHOLDER.split('{')[0]
    CHILD_PLACEHOLDER_END = CHILD_PLACEHOLDER.split('}')[1]

    if not inplace:
        # make a copy of the input heap and work on it
        _heap = copy.deepcopy(heap)
    else:
        _heap = heap

    def _dfs_build(root, heap, parent, id_offset):

        # if the heap was generated from an empty file, the heap will contains only the root node without the field "HEAP_CODE"
        if HEAP_CODE not in root and len(heap) == 1:
            return [("", 0, root[HEAP_TYPE])]

        assert HEAP_CODE in root

        def has_children(node):
            return HEAP_CHILDREN in node

        def _code2tokens(code, node_id, node_type):
            
            code_l = code.replace(CHILD_PLACEHOLDER_BEG, TMP_SEQ_SEPARATOR + CHILD_PLACEHOLDER_BEG).replace(CHILD_PLACEHOLDER_END, CHILD_PLACEHOLDER_END + TMP_SEQ_SEPARATOR).split(TMP_SEQ_SEPARATOR)

            return [(tok, node_id, node_type) for tok in code_l if tok != ""]

        if root[HEAP_CODE]:
            root[HEAP_TOKENS] = _code2tokens(
                root[HEAP_CODE], root[HEAP_ID], root[HEAP_TYPE]
            )
        else:
            root[HEAP_TOKENS] = _code2tokens(
                root[HEAP_PLACEHOLDER], root[HEAP_ID], root[HEAP_TYPE]
            )

        if has_children(root):

            for heap_node_id in root[HEAP_CHILDREN]:

                # add the offset
                offsetted_node_id = heap_node_id - id_offset
                assert heap[offsetted_node_id][HEAP_ID] == heap_node_id

                partial_tokens = _dfs_build(
                    heap[offsetted_node_id], heap, root, id_offset
                )
                for i in range(len(root[HEAP_TOKENS])):

                    token, _, _ = root[HEAP_TOKENS][i]
                    if token == CHILD_PLACEHOLDER.format(heap_node_id):
  
                        root[HEAP_TOKENS] = (
                            root[HEAP_TOKENS][:i]
                            + partial_tokens
                            + root[HEAP_TOKENS][i + 1 :]
                        )
                        break

        else:
            # is a leaf
            pass

        # handle numbers to avoid problems during the tree construction
        return [
            (
                token.replace(NUMBER_PLACEHOLDER_BEG, "").replace(NUMBER_PLACEHOLDER_END, ""),
                node_id,
                node_type,
            )
            for token, node_id, node_type in root[HEAP_TOKENS]
        ]

    return _dfs_build(_heap[0], _heap, None, _heap[0][HEAP_ID])


def heap2ast(heap: list) -> ast.AST:

    raise NotImplementedError
