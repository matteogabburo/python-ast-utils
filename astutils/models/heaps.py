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
        self.tokenized_segments = None
        
        self._segments_map = {}
        self._tokenized_segments_map = {}
        self._positions_map = {}
        self._type_map = {}

        self.n_tokenized_tokens = None

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

    @staticmethod
    def _get_segments(code, regex_pattern):

        def _replace(m):
            return TMP_SEQ_SEPARATOR + m.group(0) + TMP_SEQ_SEPARATOR

        pieces = []
        for piece in regex_pattern.sub( _replace, code).split(TMP_SEQ_SEPARATOR):
            if piece != '':
                pieces.append(piece)

        return pieces

    def make_segments(self, regex_pattern):
        self.segments = self._get_segments(self.code_segment, regex_pattern)

    def replace_next(self, sep, to_replace, node_id, node_type):

        if to_replace.code_segment in self.code_segment:

            code_l = self.code_segment.split(sep)
            if to_replace.code_segment in code_l[-1]:

                code_l[-1] = code_l[-1].replace(
                    to_replace.code_segment, CHILD_PLACEHOLDER.format(node_id), 1
                )

                self.code_segment = sep.join(code_l) 

            else:
                self.code_segment = self.code_segment.replace(to_replace.code_segment, CHILD_PLACEHOLDER.format(node_id), 1)

            self._segments_map[CHILD_PLACEHOLDER.format(node_id)] = to_replace.code_segment
            self._positions_map[CHILD_PLACEHOLDER.format(node_id)] = node_id
            self._type_map[CHILD_PLACEHOLDER.format(node_id)] = node_type

            return True
        else:
            return False
        

    def resolve_tuple(self, node_id=None, node_type=None, heap=None):

        resolved = []

        if self.segments is not None:
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

    def resolve_tokenized_tuple(self, heap, node_id=None, node_type=None):

        resolved = []

        def _build_tuple(segment):
            curr_tuple = [segment]
            if node_id:
                curr_tuple.append(node_id)
            if node_type:
                curr_tuple.append(node_type) 
            
            return curr_tuple

        for segment in self.tokenized_segments:

            # is a key for the map vocabulary
            if isinstance(segment, str) and segment in self._positions_map:
                
                if heap:
                    # build tuples recursively
                    child_id = self._positions_map[segment]
                    accumulator = heap[child_id].get_tokenized_tuple(heap=heap)

                    if isinstance(accumulator, list):      
                        resolved += accumulator
                    else:
                        raise NotImplementedError("Some error has occured, check the code")    
                else:
                    raise NotImplementedError("Initialize the heap parameter")

            else:
                
                if isinstance(segment, str):
                    resolved.append(_build_tuple(segment))

                elif isinstance(segment, list): 
                    for sub_seq in segment:
                        resolved.append(_build_tuple(sub_seq))


        return resolved



    def tokenize(self, regex_pattern, fn_tokenize, *args, **kwargs):

        counter = 0
        
        for segment in self.segments:

            if segment not in self._segments_map:

                if self.tokenized_segments is None:
                    self.tokenized_segments = []
                
                self.tokenized_segments.append(fn_tokenize(self._unmask_numbers(segment), *args, **kwargs))
                counter += len(self.tokenized_segments[-1])

            else:
                
                if self.tokenized_segments is None:
                    self.tokenized_segments = []
                self.tokenized_segments.append(segment)

        self.n_tokenized_tokens = counter
            
     
class HeapNode:

    def __init__(self, node_id, parent, node_type, node_value=None, code=None, children=None):
        
        self.node_id = node_id
        self.node_type = node_type
        self.node_value = node_value
        self.children = children
        self.parent = parent
        self.code = code
        self.placeholder = None

        self.abstract_depth = 0 if self.parent is None else self.parent.abstract_depth + 1
        self.abstract_walk = [node_id] if self.parent is None else self.parent.abstract_walk + [node_id]
        self.depth = None
        self.walk = None

        self._size_tree = 1
        
        self._is_tokenized=False
        self._n_tokenized_tokens=None
        self._ntokens_count_updated = False

        self.first_parent_node = None
        
    def update_depth(self):

        if not self.is_abstract():
            self.depth = 0 if self.parent is None else self.parent.depth + 1
        else:
            self.depth = 0 if self.parent is None else self.parent.depth 
    
    def update_walk(self):

        if not self.is_abstract():
            self.walk = [self.node_id] if self.parent is None else self.parent.walk + [self.node_id]
        else:
            self.walk = [self.node_id] if self.parent is None else self.parent.walk

    def is_tokenized(self):
        return self._is_tokenized

    def get_walk(self):
        return self.walk


    def _add_ntokens(self, ntokens):

        if not self.is_abstract():

            if self._n_tokenized_tokens is None:
                self._n_tokenized_tokens = ntokens
            else:
                self._n_tokenized_tokens += ntokens

            global VISITED
            VISITED.append(self.node_id)
                

    def update_ntokens(self):

        if not self.is_abstract() and not self._ntokens_count_updated:
        
            if not self._ntokens_count_updated:
                self._add_ntokens(self.code.n_tokenized_tokens)
                self._ntokens_count_updated = True

            def _update_parent_node(candidate, ntokens):
                
                candidate._add_ntokens(ntokens)
                
                if candidate.is_root() or candidate is None:
                    return None

                candidate = candidate.parent 
                while candidate.is_abstract():
                    candidate = candidate.parent 

                return candidate

            candidate = self.first_parent_node
            while candidate is not None:

                candidate = _update_parent_node(candidate, self.code.n_tokenized_tokens)




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
        
        if self.children is None:
            self.children = []
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

    def _update_code(self, node, to_replace, node_id, node_type):

        if not node.is_root():

            success = False
            if not node.is_abstract() and not node.parent.is_abstract():
                success = node.parent.code.replace_next(CHILD_PLACEHOLDER_END, to_replace, node_id, node_type)
                
            elif not node.is_abstract() and node.parent.is_abstract():
                success = node.parent.placeholder.replace_next(CHILD_PLACEHOLDER_END, to_replace, node_id, node_type)

            if not success:
                self._update_code(node.parent, to_replace, node_id, node_type)
            else:
                self.first_parent_node = node.parent

    def setup_code(self):

        if not self.is_root():
            
            if self.is_abstract() and self.parent.is_abstract():
                self.placeholder = self.parent.placeholder

            elif self.is_abstract() and not self.parent.is_abstract():
                self.placeholder = self.parent.code
            
            elif not self.is_abstract() and self.parent.is_abstract():
                self._update_code(self, self.code, self.node_id, self.node_type)

            elif not self.is_abstract() and not self.parent.is_abstract():
                self._update_code(self, self.code, self.node_id,  self.node_type)

    def anonimize_numbers(self, regex_pattern):
        
        if self.code:
            self.code.anonimize_numbers(regex_pattern)

    def num_tokenized_tokens(self):
        return self._n_tokenized_tokens

    def make_segments(self, regex_pattern):

        if not self.is_abstract():
            self.code.make_segments(regex_pattern)

    def get_tuple(self, heap=None):

        return self.code.resolve_tuple(self.node_id, self.node_type,  heap=heap) 
    
    def get_tokenized_tuple(self, heap):

        return self.code.resolve_tokenized_tuple(heap, self.node_id, self.node_type) 

    def tokenize(self, regex_pattern, fn_tokenize, *args, force=False, **kwargs):

        if not self.is_abstract() and (not self.is_tokenized() or force):
            self.code.tokenize(regex_pattern, fn_tokenize, *args, **kwargs)
            self._is_tokenized = True

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

    def __init__(self, ast_tree: ast.AST=None,
                 source=None,
                 positional=True, 
                 not_considered_leaves=[], 
                 padded=True, 
                 heap_capsule=None
                 ):

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

            if self._have_source():
                heap_node.add_code_segment(get_source_segment(self._get_source_map(), tree, padded=self.padded))

                # replace segments of code of children nodes with some alias
                if self.positional:

                    if heap_node.is_root():
                        heap_node.add_code_segment(source)
            
                    heap_node.anonimize_numbers(self._get_numbers_pattern())
                    heap_node.setup_code()

                    # update walk and depth
                    heap_node.update_walk()
                    heap_node.update_depth()

                    """ BEG FUTURE IMPROVEMENTS, to remove or optimize this part """
                    # build the segments to be used for the tuples creation
                    heap_node.make_segments(self._get_child_pattern())
                    """ END FUTURE IMPROVEMENTS """

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

                        """ BEG FUTURE IMPROVEMENTS, to remove or optimize this part """
                        if self._have_source():
                            # build the segments to be used for the tuples creation
                            heap_node.make_segments(self._get_child_pattern())
                        """ END FUTURE IMPROVEMENTS """
            
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

    def _check_if_possible_to_return_values(self):
        if not self._is_tree_empty() and not self._only_root() and self._is_code_embedded():
            return True
            
        elif self._is_tree_empty():
            return False
        
        elif self._only_root() and not self._is_code_embedded():
            return False

        elif self._is_code_embedded():
            return True

        elif not self._is_code_embedded():
            raise Exception("Source code was not passed as parameter during the Heap construction, this method is unavailable")

        else:
            raise NotImplementedError

    def get_heap_tuples(self):

        if self._check_if_possible_to_return_values():
            return self.get_root().get_tuple(heap=self._get_heap())
        else:
            return ""
            
    def get_heap_tokenized_tuples(self, fn_tokenize, *args, **kwargs):

        if self._check_if_possible_to_return_values():
            self.tokenize(fn_tokenize=fn_tokenize, *args, **kwargs)
            return self.get_root().get_tokenized_tuple(heap=self._get_heap())
        else:
            return ""

    def get_subtree(self, node_id):
        
        if node_id in self._get_heap():
            return self._get_heap()[node_id]
        else:
            raise ValueError("node_id not found")

    def get_size(self):
        return self.get_root().size_tree()

    def decompose(self, min_size=1, max_size=None, measure='nnodes'):
        
        if max_size is None:
            max_size = self.get_size()

        heaps = []
        for node_id in self._get_heap():

            if not self.get_node(node_id).is_abstract():

                if measure == 'nnodes':
                    sub_size = self.get_node(node_id).size_tree()
                elif measure == 'ntokens':
                    sub_size = self.get_node(node_id).num_tokenized_tokens()
                else:
                    pass

                if sub_size >= min_size and sub_size <= max_size:
                    heaps.append(self._create_subheap(node_id))

        return heaps     

    def gredy_decomposition(self, min_size=1, max_size=None, mode='max', measure='nnodes'):

        if max_size is None:
            max_size = self.get_size()

        def _max_decomposition(node_id):

            res = []

            curr_node = self.get_node(node_id)
            if not curr_node.is_abstract() and curr_node.num_tokenized_tokens():

                if measure == 'nnodes':
                    sub_size = curr_node.size_tree()
                elif measure == 'ntokens':
                    sub_size = curr_node.num_tokenized_tokens()
                else:
                    raise ValueError("Not supported measure")

                curr_node.print_descr()

                if sub_size >= min_size and sub_size <= max_size: 

                    res += [node_id]
                    return res
                
            if curr_node.children is not None:
                for child_id in curr_node.children:
                    res += _max_decomposition(child_id)

            return res    

        def _min_decomposition(node_id, walk=[], banned=[]):

            res = []

            walk.append(node_id)

            curr_node = self.get_node(node_id)
            if not curr_node.is_abstract():

                if measure == 'nnodes':
                    sub_size = curr_node.size_tree()
                elif measure == 'ntokens':
                    sub_size = curr_node.num_tokenized_tokens()
                else:
                    raise ValueError("Not supported measure")

            if curr_node.children is not None:
                for child_id in curr_node.children:
                    new_res, walk, banned = _min_decomposition(child_id, walk, banned)
                    res += new_res
                    walk = walk[:-1]
            
            if not curr_node.is_abstract() and curr_node.num_tokenized_tokens():
                if sub_size >= min_size and sub_size <= max_size and node_id not in banned: 
                    res += [node_id]
                    banned += walk
            
            return res, walk, banned              

        heaps_nodes = []
        if mode == 'max':
            heaps_nodes = _max_decomposition(self._root_id)

        elif mode == 'min':
            heaps_nodes, _, _ = _min_decomposition(self._root_id)

        else:
            raise ValueError("this mode is not supported (chose between 'max' and 'min'")
        
        return [self._create_subheap(sub_root) for sub_root in heaps_nodes]

    def _is_tree_empty(self):
        return self.get_root().size_tree() == 0

    def _only_root(self):
        return self.get_root().size_tree() == 1

    def _is_code_embedded(self):

        return True if self._capsule._source else False

    def tokenize(self, fn_tokenize, *args, **kwargs):

        for node_id in self._get_heap():
            
            self.get_node(node_id).tokenize(self._get_child_pattern(), fn_tokenize, *args, **kwargs)
            
        for node_id in self._get_heap():
            
            self.get_node(node_id).update_ntokens()


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