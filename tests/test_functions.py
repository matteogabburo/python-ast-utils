#! /usr/bin/python

import pytest
import os
import astutils

RESOURCES_PATH = "./tests/resources"
MAX_PY_VERSION = (3, 9)


def _read(filename):
    with open(filename, "r") as f:
        text = f.read()
    return text


def _check_instances(class_a, class_b):

    if class_a == class_b:
        return True

    if "__dict__" in dir(class_a) and "__dict__" in dir(class_b):

        d_a = class_a.__dict__
        d_b = class_b.__dict__

        if len(d_a) != len(d_b):
            return False

        for k in d_a:

            if k not in d_b:
                return False

            elif "__dict__" in dir(d_a[k]):
                if not check_instances(d_a[k], d_b[k]):
                    return False

            elif d_a[k] != d_b[k]:
                return False

        return True
    else:
        return class_a != class_b


def _get_resources(folder_path):
    def _get_min_py_version(filename):
        words = filename.split(".")[0].split("_")
        for i, word in enumerate(words):
            if word == "minpyversion":
                version = (int(words[i + 1]), int(words[i + 2]))
                break
        return version

    ret = [
        (os.path.join(folder_path, filename), _get_min_py_version(filename))
        for filename in os.listdir(folder_path)
    ]
    return ret


def _ast_parse_parameters(folder_path):

    resources = _get_resources(folder_path)
    modes = ["exec"]  # , 'func_type']
    type_comments = [True, False]

    tests = []
    for fname, min_pyversion in resources:
        for mode in modes:
            for type_comment in type_comments:
                for feature_version in [
                    (major, minor)
                    for major in range(min_pyversion[0], MAX_PY_VERSION[0] + 1)
                    for minor in range(min_pyversion[1], MAX_PY_VERSION[1] + 1)
                ]:
                    tests.append((fname, mode, type_comment, feature_version))
    return tests


@pytest.mark.parametrize(
    "fname,mode,type_comments,feature_version", _ast_parse_parameters(RESOURCES_PATH),
)
def test_ast_parse(fname, mode, type_comments, feature_version):
    astutils.ast_parse(
        fname, mode=mode, type_comments=type_comments, feature_version=feature_version
    )


@pytest.mark.parametrize(
    "fname,mode,type_comments,feature_version", _ast_parse_parameters(RESOURCES_PATH),
)
def test_ast_parse_from_string(fname, mode, type_comments, feature_version):
    astutils.ast_parse_from_string(
        _read(fname),
        mode=mode,
        type_comments=type_comments,
        feature_version=feature_version,
    )


@pytest.mark.parametrize(
    "fname,mode,type_comments,feature_version", _ast_parse_parameters(RESOURCES_PATH),
)
def test_ast_unparse(fname, mode, type_comments, feature_version):
    ast_tree = astutils.ast_parse(
        fname, mode=mode, type_comments=type_comments, feature_version=feature_version
    )
    astutils.ast_unparse(ast_tree)


@pytest.mark.parametrize(
    "fname,mode,type_comments,feature_version", _ast_parse_parameters(RESOURCES_PATH),
)
def test_ast2dict_dict2ast(fname, mode, type_comments, feature_version):
    ast_tree1 = astutils.ast_parse(
        fname, mode=mode, type_comments=type_comments, feature_version=feature_version
    )
    ast_dict = astutils.ast2dict(ast_tree1)
    ast_tree2 = astutils.dict2ast(ast_dict)
    ast_dict2 = astutils.ast2dict(ast_tree2)

    assert _check_instances(ast_dict, ast_dict2)


@pytest.mark.parametrize(
    "fname,mode,type_comments,feature_version", _ast_parse_parameters(RESOURCES_PATH),
)
def test_ast2heap_heap2code(fname, mode, type_comments, feature_version):
    ast_tree = astutils.ast_parse(
        fname, mode=mode, type_comments=type_comments, feature_version=feature_version
    )
    sourcecode = _read(fname)

    ast_heap = astutils.ast2heap(ast_tree, source=sourcecode)
    source_code_1 = astutils.heap2code(ast_heap)

    assert _check_instances(sourcecode, source_code_1)


@pytest.mark.parametrize(
    "fname,mode,type_comments,feature_version", _ast_parse_parameters(RESOURCES_PATH),
)
def test_ast2heap_heap2tokens(fname, mode, type_comments, feature_version):
    ast_tree = astutils.ast_parse(
        fname, mode=mode, type_comments=type_comments, feature_version=feature_version
    )
    sourcecode = _read(fname)

    ast_heap = astutils.ast2heap(ast_tree, source=sourcecode)
    tokens = astutils.heap2tokens(ast_heap)

    source_code_1 = "".join([token[0] for token in tokens])

    assert _check_instances(sourcecode, source_code_1)
