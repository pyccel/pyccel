# pylint: disable=missing-function-docstring, missing-module-docstring

import pytest
from modules import epyccel_IfTernaryOperator
from utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_epyccel_IfTernaryOperator_mod(language):
    return epyccel_module_with_fallback(epyccel_IfTernaryOperator, language)


# wp suffix means With Parentheses
# ------------------------------------------------------------------------------
def test_f1(epyc_epyccel_IfTernaryOperator_mod):
    f1 = epyccel_IfTernaryOperator.f1
    f = epyc_epyccel_IfTernaryOperator_mod.f1

    # ...
    assert f(6) == f1(6)
    assert f(4) == f1(4)
    # ...


# ------------------------------------------------------------------------------


def test_f2(epyc_epyccel_IfTernaryOperator_mod):
    f2 = epyccel_IfTernaryOperator.f2
    f = epyc_epyccel_IfTernaryOperator_mod.f2

    # ...
    assert f(6) == f2(6)
    assert f(4) == f2(4)
    # ...


# ------------------------------------------------------------------------------
def test_f3(epyc_epyccel_IfTernaryOperator_mod):
    f3 = epyccel_IfTernaryOperator.f3
    f3wp = epyc_epyccel_IfTernaryOperator_mod.f3wp
    f = epyc_epyccel_IfTernaryOperator_mod.f3
    fwp = epyc_epyccel_IfTernaryOperator_mod.f3wp

    # ...
    assert f(6) == f3(6)
    assert f(4) == f3(4)

    assert fwp(6) == f3wp(6)
    assert fwp(4) == f3wp(4)
    # ...


# ------------------------------------------------------------------------------


def test_f4(epyc_epyccel_IfTernaryOperator_mod):
    f4 = epyccel_IfTernaryOperator.f4
    f4wp = epyc_epyccel_IfTernaryOperator_mod.f4wp
    f = epyc_epyccel_IfTernaryOperator_mod.f4
    fwp = epyc_epyccel_IfTernaryOperator_mod.f4wp

    # ...
    assert f(6) == f4(6)
    assert f(4) == f4(4)

    assert fwp(6) == f4wp(6)
    assert fwp(4) == f4wp(4)
    # ...


# ------------------------------------------------------------------------------
def test_f5(epyc_epyccel_IfTernaryOperator_mod):
    f5 = epyccel_IfTernaryOperator.f5
    f5wp = epyc_epyccel_IfTernaryOperator_mod.f5wp
    f = epyc_epyccel_IfTernaryOperator_mod.f5
    fwp = epyc_epyccel_IfTernaryOperator_mod.f5wp

    # ...
    assert f(6) == f5(6)
    assert f(4) == f5(4)
    assert f(5) == f5(5)

    assert fwp(6) == f5wp(6)
    assert fwp(4) == f5wp(4)
    assert fwp(5) == f5wp(5)
    # ...


# ------------------------------------------------------------------------------
def test_f6(epyc_epyccel_IfTernaryOperator_mod):
    f6 = epyccel_IfTernaryOperator.f6
    f6wp = epyc_epyccel_IfTernaryOperator_mod.f6wp
    f = epyc_epyccel_IfTernaryOperator_mod.f6
    fwp = epyc_epyccel_IfTernaryOperator_mod.f6wp

    # ...
    assert f(6) == f6(6)
    assert f(4) == f6(4)
    assert f(5) == f6(5)

    assert fwp(6) == f6wp(6)
    assert fwp(4) == f6wp(4)
    assert fwp(5) == f6wp(5)
    # ...


# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "language",
    (
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(
                    reason="Copy of a list not yet supported (required to handle the generated temporary)."
                ),
                pytest.mark.fortran,
            ],
        ),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(
                    reason="Copy of a list not yet supported (required to handle the generated temporary)."
                ),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_f7(language):
    def f7(x: "int"):
        a = [1.0, 2.0, 3.0] if x < 5 else [1.5, 6.5, 7.5]
        return a[0]

    def f7wp(x: "int"):
        a = (
            [1.0, 2.0, 3.0]
            if x < 5
            else ([1.5, 6.5, 7.5] if x > 5 else [3.1, 9.5, 2.8])
        )
        return a[0]

    f = epyccel(f7, language=language)
    fwp = epyccel(f7wp, language=language)

    # ...
    assert f(6) == f7(6)
    assert f(4) == f7(4)

    assert fwp(6) == f7wp(6)
    assert fwp(4) == f7wp(4)
    # ...


# ------------------------------------------------------------------------------


def test_f8(epyc_epyccel_IfTernaryOperator_mod):
    f8 = epyccel_IfTernaryOperator.f8
    f8wp = epyc_epyccel_IfTernaryOperator_mod.f8wp
    f = epyc_epyccel_IfTernaryOperator_mod.f8
    fwp = epyc_epyccel_IfTernaryOperator_mod.f8wp

    # ...
    assert f(6) == f8(6)
    assert f(4) == f8(4)

    assert fwp(6) == f8wp(6)
    assert fwp(4) == f8wp(4)
    # ...


# ------------------------------------------------------------------------------


def test_f9(epyc_epyccel_IfTernaryOperator_mod):
    f9 = epyccel_IfTernaryOperator.f9
    f9wp1 = epyc_epyccel_IfTernaryOperator_mod.f9wp1
    f9wp2 = epyc_epyccel_IfTernaryOperator_mod.f9wp2
    f = epyc_epyccel_IfTernaryOperator_mod.f9
    fwp1 = epyc_epyccel_IfTernaryOperator_mod.f9wp1
    fwp2 = epyc_epyccel_IfTernaryOperator_mod.f9wp2
    # ...
    assert f(6) == f9(6)
    assert f(4) == f9(4)

    assert fwp1(6) == f9wp1(6)
    assert fwp1(4) == f9wp1(4)

    assert fwp2(6) == f9wp2(6)
    assert fwp2(4) == f9wp2(4)
    # ...


# ------------------------------------------------------------------------------


def test_f10(epyc_epyccel_IfTernaryOperator_mod):
    f10 = epyccel_IfTernaryOperator.f10
    f10wp1 = epyc_epyccel_IfTernaryOperator_mod.f10wp1
    f10wp2 = epyc_epyccel_IfTernaryOperator_mod.f10wp2
    f = epyc_epyccel_IfTernaryOperator_mod.f10
    fwp1 = epyc_epyccel_IfTernaryOperator_mod.f10wp1
    fwp2 = epyc_epyccel_IfTernaryOperator_mod.f10wp2
    # ...
    assert f(6) == f10(6)
    assert f(4) == f10(4)

    assert fwp1(6) == f10wp1(6)
    assert fwp1(4) == f10wp1(4)

    assert fwp2(6) == f10wp2(6)
    assert fwp2(4) == f10wp2(4)
    # ...


# ------------------------------------------------------------------------------


def test_f11(epyc_epyccel_IfTernaryOperator_mod):
    f11 = epyccel_IfTernaryOperator.f11
    f = epyc_epyccel_IfTernaryOperator_mod.f11
    # ...
    assert f(6) == f11(6)
    assert f(-4) == f11(-4)
    # ...


# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "language",
    (
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(
                    reason="Copy of a list not yet supported (required to handle the generated temporary)."
                ),
                pytest.mark.fortran,
            ],
        ),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(
                    reason="Copy of a list not yet supported (required to handle the generated temporary)."
                ),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_f12(language):
    def f12(x: "int"):
        a = [1.0, 2.0, 3.0, 4.0] if x < 5 else [1.5, 6.5, 7.5]
        return a[0]

    def f12wp(x: "int"):
        a = (
            [1.0, 2.0, 3.0]
            if x < 5
            else ([1.5, 6.5, 7.5] if x > 5 else [3.1, 9.5, 2.8, 2.9])
        )
        return a[0]

    f = epyccel(f12, language=language)
    fwp = epyccel(f12wp, language=language)

    # ...
    assert f(6) == f12(6)
    assert f(4) == f12(4)

    assert fwp(6) == f12wp(6)
    assert fwp(4) == f12wp(4)
    # ...


# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "language",
    (
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(reason="Can't return a string"),
                pytest.mark.fortran,
            ],
        ),
        pytest.param(
            "c",
            marks=pytest.mark.c,
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_f13(language):
    def f13(b: bool):
        a = "hello" if b else "world!"
        return a

    def f13wp(b1: bool, b2: bool):
        a = "hello" if b1 else ("world" if b2 else "hello world")
        return a

    f = epyccel(f13, language=language)
    fwp = epyccel(f13wp, language=language)

    # ...
    assert f(True) == f13(True)
    assert f(False) == f13(False)

    assert fwp(True, True) == f13wp(True, True)
    assert fwp(True, False) == f13wp(True, False)
    assert fwp(False, True) == f13wp(False, True)
    assert fwp(False, False) == f13wp(False, False)
    # ...


# ------------------------------------------------------------------------------
