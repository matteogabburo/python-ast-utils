import astutils
import pytest

from astutils.functions import ast_parse
from astutils.functions import ast_unparse
from astutils.functions import ast2dict
from astutils.functions import dict2ast
from astutils.functions import dict2json
from astutils.functions import ast2json


@pytest.mark.parametrize(
    "path",
    [
        ("path/that/not/exist/foo_1_levels.json"),
    ]
)
def test_ast_parse(path):

    testargs = ["prog"]
    with patch.object(sys, 'argv', testargs):    

        with pytest.raises(Exception) as exc_info:
            @pyovra.head(root_conf=str(tmp_path / path))
            def app(pyovra_cfg, *args, **kwargs) -> dict:
                return pyovra_cfg
            app()
        assert exc_info.type == expected_exception