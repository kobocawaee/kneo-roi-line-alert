"""Make the `openni` (2.3.0, 2018) Python binding import on Python 3.14+.

openni.utils.CEnumMeta instantiates enum members inside the metaclass __new__.
Since Python 3.14, ctypes finishes setting up simple types (`_type_`) in the
metaclass __init__, so at __new__ time the class is still abstract and
`cls2(value)` raises "TypeError: abstract class". We move that work to __init__,
which also works on older Pythons. Must run before importing openni.openni2.
"""

from __future__ import annotations

import ctypes

_patched = False


def import_openni2():
    """Return the `openni.openni2` module with the 3.14 compatibility patch applied."""
    global _patched
    if not _patched:
        from openni import utils

        simple_type = type(ctypes.c_int)

        def _new(meta, name, bases, namespace):
            return simple_type.__new__(meta, name, bases, namespace)

        def _init(cls2, name, bases, namespace):
            simple_type.__init__(cls2, name, bases, namespace)
            if namespace.get("__module__") != utils.__name__:
                namespace["_values_"].clear()
                for member in list(namespace["_names_"].keys()):
                    if member.startswith("_"):
                        continue
                    setattr(cls2, member, cls2(namespace[member]))
                    namespace["_names_"][member] = namespace[member]
                    namespace["_values_"][namespace[member]] = member

        utils.CEnumMeta.__new__ = staticmethod(_new)
        utils.CEnumMeta.__init__ = _init
        _patched = True

    from openni import openni2
    return openni2
