import ast
from typing import Dict, List
import re
from tokenize import Number
import copy

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
CHILD_PLACEHOLDER_BEG = CHILD_PLACEHOLDER.split('{')[0]
CHILD_PLACEHOLDER_END = CHILD_PLACEHOLDER.split('}')[1]

NUMBER_PLACEHOLDER_BEG = NUMBER_PLACEHOLDER.split("{")[0]
NUMBER_PLACEHOLDER_END = NUMBER_PLACEHOLDER.split("}")[1]
CHILD_PLACEHOLDER_BEG = CHILD_PLACEHOLDER.split('{')[0]
CHILD_PLACEHOLDER_END = CHILD_PLACEHOLDER.split('}')[1]


def get_class_name(item):
    return item.__class__.__name__


class CodeSegment:

    def __init__(self, code_segment):

        self.code_segment = code_segment

        self.segments = None
        self.positions = None
        
        self._segments_map = {}
        self._positions_map = {}
        self._type_map = {}

    @staticmethod
    def _mask_numbers(code, regex_pattern):
        # handle numbers to avoid problems during the tree construction
        def _replace(m):
            return NUMBER_PLACEHOLDER.format(m.group(0))

        return regex_pattern.sub( _replace, code)

    @staticmethod
    def _unmask_numbers(code):
        return code.replace(NUMBER_PLACEHOLDER_BEG, "").replace(NUMBER_PLACEHOLDER_END, "")

    def anonimize_numbers(self, regex_pattern):

        if self.code_segment:
            self.code_segment = self._mask_numbers(self.code_segment, regex_pattern)

    def make_segments(self, regex_pattern):

        def _replace(m):
            return TMP_SEQ_SEPARATOR + m.group(0) + TMP_SEQ_SEPARATOR

        pieces = []
        for piece in regex_pattern.sub( _replace, self.code_segment).split(TMP_SEQ_SEPARATOR):
            if piece != '':
                pieces.append(piece)

        self.segments = pieces

    def replace_next(self, sep, to_replace, node_id, node_type):

        code_l = self.code_segment.split(sep)

        if to_replace.code_segment in code_l[-1]:

            code_l[-1] = code_l[-1].replace(
                to_replace.code_segment, CHILD_PLACEHOLDER.format(node_id), 1
            )

            self._segments_map[CHILD_PLACEHOLDER.format(node_id)] = to_replace.code_segment
            self._positions_map[CHILD_PLACEHOLDER.format(node_id)] = node_id
            self._type_map[CHILD_PLACEHOLDER.format(node_id)] = node_type

            self.code_segment = sep.join(code_l) 

    def resolve_tuple(self, node_id=None, node_type=None, heap=None):

        resolved = []

        for segment in self.segments:
            
            curr_tuple = []
            if segment in self._segments_map:

                if heap:
                    # build tuples recursively
                    child_id = self._positions_map[segment]
                    resolved += (heap[child_id].get_tuple(heap=heap))
                    
                else:
                    curr_tuple.append(self._segments_map[segment])
                    if node_id:
                        curr_tuple.append(self._positions_map[segment])
                    if node_type:
                        curr_tuple.append(self._type_map[segment])
                    
                    resolved.append(curr_tuple)
            else:
                curr_tuple.append(self._unmask_numbers(segment))
                if node_id:
                    curr_tuple.append(node_id)
                if node_type:
                    curr_tuple.append(node_type)

                resolved.append(curr_tuple)

        return resolved


class HeapNode:

    def __init__(self, node_id, parent, node_type, node_value=None, code=None, children=[]):
        
        self.node_id = node_id
        self.node_type = node_type
        self.node_value = node_value
        self.children = children
        self.parent = parent
        self.code = code
        self.placeholder = None

        self.depth = 0 if self.parent is None else self.parent.depth + 1
        self.walk = [node_id] if self.parent is None else self.parent.walk + [node_id]

        self._size_tree = 1
        
    def size_tree(self):
        return self._size_tree

    def _incr_size_tree(self, size):
        self._size_tree += size

    def print_descr(self):

        for attr in self.__dict__:
            value = self.__getattribute__(attr)
            if isinstance(value, HeapNode):
                print("\t", attr, ":", {'id:': value.node_id, 'node': value})

            else:
                print("\t", attr, ":", value)

    def is_root(self):
        return True if self.parent is None else False

    def is_abstract(self):
        return True if self.code is None else False

    def add_child(self, node_id):
        
        self.children.append(node_id)

    def add_child_to_parent(self, node_id):

        if self.parent:
            self.parent.add_child(node_id)

    def add_value(self, field, value):

        if self.node_value is None:
            self.node_value = {}
        self.node_value[field] = value

    def add_code_segment(self, code_segment):
        self.code = CodeSegment(code_segment) if isinstance(code_segment, str) else code_segment

    @staticmethod
    def _update_code(source, to_replace, node_id, node_type):
        source.replace_next(CHILD_PLACEHOLDER_END, to_replace, node_id, node_type)

    def setup_code(self):

        if not self.is_root():
            
            if self.is_abstract() and self.parent.is_abstract():
                self.placeholder = self.parent.placeholder

            elif self.is_abstract() and not self.parent.is_abstract():
                self.placeholder = self.parent.code
            
            elif not self.is_abstract() and self.parent.is_abstract():
                self._update_code(self.parent.placeholder, self.code, self.node_id, self.node_type)

            elif not self.is_abstract() and not self.parent.is_abstract():
                self._update_code(self.parent.code, self.code, self.node_id,  self.node_type)

    def anonimize_numbers(self, regex_pattern):
        
        if self.code:
            self.code.anonimize_numbers(regex_pattern)

    def make_segments(self, regex_pattern):

        if self.code:
            self.code.make_segments(regex_pattern)

    def get_tuple(self, heap=None):

        return self.code.resolve_tuple(self.node_id, self.node_type,  heap=heap) 


class HeapCapsule():

    def __init__(self, ast_tree: ast.AST, source=None):

        self._numbers_pattern = re.compile(Number)
        self._child_pattern = re.compile(CHILD_PLACEHOLDER.format("[0-9]+"))
        self._ast_tree = ast_tree
        self._source = source
        self._heap = {}

        self.source_map = None
        if source:
            # build the source map
            self._source_map = build_source_map(source)


class Heap:

    def __init__(self, ast_tree: ast.AST=None, source=None, positional=True, not_considered_leaves=[], padded=True, heap_capsule=None):

        if heap_capsule is None:
            assert ast_tree is not None
            self._capsule = HeapCapsule(ast_tree=ast_tree, source=source)
        else:
            self._capsule = heap_capsule

        self._root_id = None

        self._inorder_view = []

        self.positional = positional
        self.padded = padded 
        self.not_considered_leaves = not_considered_leaves

        if heap_capsule is None:
            # build the heap
            self._build_heap(ast_tree,source,None,None)

    def _get_capsule(self):
        return self._capsule

    def _create_subheap(self, sub_root_id):
        
        sub_heap = Heap(positional=self.positional,
                        not_considered_leaves=self.not_considered_leaves,
                        padded=self.padded,  
                        heap_capsule=self._get_capsule())

        sub_heap._root_id = sub_root_id
        return sub_heap

    def _get_heap(self):
        return self._capsule._heap

    def _have_source(self):
        return True if self._capsule._source else False

    def _get_source(self):
        return self._capsule._source

    def _add_node_heap(self, node_id, heap_node):
        self._capsule._heap[node_id] = heap_node
 
    def _get_source_map(self):
        return self._capsule._source_map

    def _get_numbers_pattern(self):
        return self._capsule._numbers_pattern

    def _get_child_pattern(self):
        return self._capsule._child_pattern

    def _build_heap(self, tree: ast.AST, source, field, parent):

        if isinstance(tree, ast.AST):
            """ Is an AST intermediate node"""

            # get the node id
            node_id = len(self._get_heap())
            if self._root_id is None:
                self._root_id = node_id

            # new node
            heap_node = HeapNode(node_id, parent, get_class_name(tree))

            # add the current node to the children list of its parent
            heap_node.add_child_to_parent(node_id)

            """
            BEG CRITICAL (efficiency)
            """

            if self._capsule._source:
                heap_node.add_code_segment(get_source_segment(self._get_source_map(), tree, padded=self.padded))

                # replace segments of code of children nodes with some alias
                if self.positional:

                    if heap_node.is_root():
                        heap_node.add_code_segment(source)
            
                    heap_node.anonimize_numbers(self._get_numbers_pattern())
                    heap_node.setup_code()

            """
            END CRITICAL (efficiency)
            """

            # add the new node to the heap
            self._add_node_heap(node_id, heap_node)

            # if has some children and the type of the cildren is accepted,
            # then continue to build the heap
            if len(tree._fields) > 0:
                for field in tree._fields:
                    if field not in self.not_considered_leaves:
                        self._build_heap(tree.__dict__[field],source, field, heap_node)
                        
                        """
                        BEG CRITICAL (efficiency)
                        """

                        # build the segments to be used for the tuples creation
                        if self._have_source():
                            heap_node.make_segments(self._get_child_pattern())

                        """
                        END CRITICAL (efficiency)
                        """
            
            if parent is not None:
                parent._incr_size_tree(heap_node.size_tree())

            if not heap_node.is_abstract():
                self._inorder_view.append(node_id)

        elif isinstance(tree, tuple) or isinstance(tree, list):
            """ Is a list or a tuple of something  """

            for element in tree:
                self._build_heap(element,source, field, parent)

        else:
            """ Is a leaf """
            parent.add_value(field, tree)

    def get_root(self):
        return self.get_node(self._root_id) if self._root_id is not None else None

    def get_node(self, node_id):
        return self._get_heap()[node_id]

    def get_heap_tuples(self):
        return self.get_root().get_tuple(heap=self._get_heap())

    def get_subtree(self, node_id):
        
        if node_id in self._get_heap():
            return self._get_heap()[node_id]
        else:
            raise ValueError("node_id not found")

    def get_size(self):
        return self.get_root().size_tree()

    def scompone(self, min_size=1, max_size=None):
        
        if max_size is None:
            max_size = self.get_size()

        heaps = []
        for sub_root in self._get_heap():

            sub_size = self.get_node(sub_root).size_tree()
            if  sub_size >= min_size and sub_size <= max_size:
                heaps.append(self._create_subheap(sub_root))

        return heaps        
    

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
        source_map[lineno + 1] = line + "\n"

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

    if isinstance(tree, ast.AST):

        dict_tree = tree.__dict__
        if "lineno" in dict_tree:

            lineno = dict_tree["lineno"]
            end_lineno = dict_tree["end_lineno"]
            col_offset = dict_tree["col_offset"]
            end_col_offset = dict_tree["end_col_offset"]

            if end_lineno == lineno:
                return source_map[lineno][col_offset:end_col_offset]

            ret = [source_map[i] for i in range(lineno, end_lineno + 1)]
            if not padded:
                ret[0] = ret[0][col_offset:]
            else:

                if not ret[0][:col_offset].isspace():
                    ret[0] = " " * col_offset + ret[0][col_offset:]

            ret[-1] = ret[-1][:end_col_offset]

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
    numbers_pattern = re.compile(Number)

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
                        """heap_node[HEAP_CODE] = re.sub(
                            Number, __replace, heap_node[HEAP_CODE]
                        )"""
                        heap_node[HEAP_CODE] = numbers_pattern.sub( __replace, heap_node[HEAP_CODE])

                    if not is_not_root(heap):
                        #heap_node[HEAP_CODE] = re.sub(Number, __replace, source)
                        heap_node[HEAP_CODE] = numbers_pattern.sub( __replace, source)

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
