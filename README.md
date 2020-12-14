# python-ast-utils

Some utility functions used to handle Python AST.

#### Parse a Python file:
```.py
import astutils
ast = astutils.ast_parse("path/to/a/file.py")
```

#### Unparse a Python AST:
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
assert ast1 == ast2
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