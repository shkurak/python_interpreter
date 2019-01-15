import pytest
import sys

from . import cases
from . import vm_scorer
from . import vm_runner
from . import vm

IDS = [test.name for test in cases.TEST_CASES]
TESTS = [test.text_code for test in cases.TEST_CASES]
SCORER = vm_scorer.Scorer(TESTS)
SCORES = [SCORER.score(test) for test in TESTS]


def test_version():
    """
    To do this task you need python3.6.3
    """
    assert '3.6.3' == sys.version.split(' ', maxsplit=1)[0]


def test_stat():
    """
    Shows stat for all passed test cases
    Usage:
        $ pytest test_public.py::test_stat -s
    """
    vm_scorer.dump_tests_stat(sys.stdout, SCORER)


@pytest.mark.parametrize("test,score", zip(cases.TEST_CASES, SCORES), ids=IDS)
def test_dis(test: cases.Case, score: float):
    """
    Shows dis for test cases
        Usage for all TEST_CASES:
            $ pytest test_public.py::test_dis -s
        Usage for some test case in TEST_CASES:
            $ pytest test_public.py::test_dis[<SOME_TEST_CASE_NAME>] -s
    :param test: test case to check
    :param score: score for test calculated compared to others
    """
    vm_runner.compile_code(test.text_code)
    sys.stdout.write("Score: {}\n".format(score))


@pytest.mark.parametrize("test,score", zip(cases.TEST_CASES, SCORES), ids=IDS)
def test_all_cases(test: cases.Case, score: float):
    """
    Compare all test cases with etalon
    :param test: test case to check
    :param score: score for test if passed
    """
    code = vm_runner.compile_code(test.text_code)
    globals_context = {}
    vm_out, vm_err, vm_exc = vm_runner.execute(code, vm.VirtualMachine().run)
    py_out, py_err, py_exc = vm_runner.execute(code, eval, globals_context, globals_context)

    assert vm_out == py_out

    if py_exc is not None:
        assert isinstance(vm_exc, type(py_exc))
    else:
        assert vm_exc is None

    # Write to stderr for subsequent parsing
    sys.stderr.write("{}\n".format(score))
