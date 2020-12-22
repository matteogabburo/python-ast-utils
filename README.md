# python-ast-utils [![Build Status](https://travis-ci.com/matteogabburo/python-ast-utils.svg?branch=main)](https://travis-ci.com/matteogabburo/python-ast-utils)

Some utility functions used to handle Python AST.

## Installation

- Important: To use all the functions, ```Python3.9``` is required.

#### pip + git
```
pip install git+https://github.com/matteogabburo/python-ast-utils.git
```

## Examples

#### Parse a Python file:
```.py
import astutils
ast = astutils.ast_parse("path/to/a/file.py")
```

#### Parse a string representing a Python file:
```.py
import astutils
sourcecode = "def hello(name):\n\tprint('hello', name)\nhello('John')\n"
ast = astutils.ast_parse_from_string(sourcecode)
```

#### Unparse a Python AST (only from Python3.9):
```.py
import astutils
ast = astutils.ast_parse("path/to/a/file.py")
py_program_str = astutils.ast_unparse(ast)
```

#### AST2Dict:
```.py
import astutils
ast = astutils.ast_parse("path/to/a/file.py")
dict_ast = astutils.ast2dict(ast)
```

#### Dict2AST:
```.py
import astutils
ast1 = astutils.ast_parse("path/to/a/file.py")
dict_ast = astutils.ast2dict(ast1)
ast2 = astutils.dict2ast(ast)
# ast1 == ast2
```

#### dict2Json:
```.py
import astutils
ast = astutils.ast_parse("path/to/a/file.py")
dict_ast = astutils.ast2dict(ast)
json_ast = astutils.dict2json(dict_ast)
```

#### AST2Json:
```.py
import astutils
ast = astutils.ast_parse("path/to/a/file.py")
json_ast = astutils.ast2json(ast)
```

#### AST2Heap:
```.py
import astutils
ast = astutils.ast_parse("path/to/a/file.py")
heap_ast = astutils.ast2heap(ast)
```

#### Heap2Code:
```.py
import astutils
sourcecode = "def hello(name):\n\tprint('hello', name)\nhello('John')\n"
ast = astutils.ast_parse_from_string(sourcecode)
heap_ast = astutils.ast2heap(ast, source=sourcecode)
code = astutils.heap2code(heap_ast)
assert(sourcecode == code)
```

#### Heap2Tokens:
```.py
import astutils
sourcecode = "def hello(name):\n\tprint('hello', name)\nhello('John')\n"
ast = astutils.ast_parse_from_string(sourcecode)
heap_ast = astutils.ast2heap(ast, source=sourcecode)
tokens = astutils.heap2tokens(heap_ast)
assert(sourcecode == "".join([tok for tok, node_id, node_type in tokens]))
```