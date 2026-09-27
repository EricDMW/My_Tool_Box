"""Package-level checks: versions, lazy imports and optional dependencies."""

import subprocess
import sys

import env_lib
import toolkit


def test_versions_are_in_sync():
    assert env_lib.__version__ == toolkit.__version__


def _imported_modules(statement: str) -> set:
    code = f"import sys; {statement}; print(' '.join(sorted(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_import_env_lib_is_lightweight():
    modules = _imported_modules("import env_lib")
    for heavy in ("torch", "pygame", "pymunk", "scipy"):
        assert heavy not in modules, heavy


def test_import_toolkit_is_lightweight():
    modules = _imported_modules("import toolkit")
    for heavy in ("torch", "gymnasium", "env_lib", "tkinter", "matplotlib"):
        assert heavy not in modules, heavy


def test_plotkit_does_not_need_env_lib():
    modules = _imported_modules("import toolkit.plotkit")
    assert "env_lib" not in modules
    assert "gymnasium" not in modules
