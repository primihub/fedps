import platform
import sys


def _get_sys_info():
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info():
    deps = [
        "pip",
        "setuptools",
        "fedps",
        "scikit-learn",
        "numpy",
        "scipy",
        "datasketches",
        "pyzmq",
    ]

    from importlib.metadata import PackageNotFoundError, version

    deps_info = {}
    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info


def show_versions():
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print("\nPython dependencies:")
    for k, stat in deps_info.items():
        print("{k:>13}: {stat}".format(k=k, stat=stat))


def import_is_scalar_nan():
    try:
        # sklearn >= 1.5
        from sklearn.utils._missing import is_scalar_nan
    except ImportError:
        from sklearn.utils import is_scalar_nan
    return is_scalar_nan


def import_safe_assign():
    try:
        # sklearn >= 1.5
        from sklearn.utils._indexing import _safe_assign
    except ImportError:
        from sklearn.utils import _safe_assign
    return _safe_assign
