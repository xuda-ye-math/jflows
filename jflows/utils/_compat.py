"""Small public-keyword compatibility helpers for :mod:`jflows.utils`.

The utility API uses compact, semantic names (for example ``dt`` and
``steps``), while accepting the original keyword names during the migration.
Keeping the translation in an ordinary Python wrapper leaves the numerical
kernel and its JAX trace unchanged.
"""

from __future__ import annotations

from functools import wraps


def legacy_keywords(**aliases: str):
    """Accept deprecated keyword aliases without changing positional calls.

    ``aliases`` maps an old name to its canonical replacement.  Supplying both
    spellings is rejected explicitly instead of silently choosing one.  The
    wrapped function's canonical signature remains visible through
    ``inspect.signature`` because :func:`functools.wraps` preserves
    ``__wrapped__``.
    """

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            for old, new in aliases.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    raise TypeError(
                        f"{function.__name__}() received both {new!r} and its "
                        f"deprecated alias {old!r}"
                    )
                kwargs[new] = kwargs.pop(old)
            return function(*args, **kwargs)

        mapping = ", ".join(f"``{old}`` -> ``{new}``" for old, new in aliases.items())
        note = (
            "\n\n    Deprecated keyword aliases: " + mapping +
            ". They remain accepted for backward compatibility; do not pass "
            "both spellings."
        )
        wrapped.__doc__ = (function.__doc__ or "") + note
        return wrapped

    return decorate


def inherit_implementation_doc(public, implementation) -> None:
    """Expose an implementation's full docs through its compatibility wrapper."""
    marker = "\n\n    Deprecated keyword aliases:"
    _, found, alias_text = (public.__doc__ or "").partition(marker)
    suffix = f"{found}{alias_text}" if found else ""
    public.__doc__ = (implementation.__doc__ or "") + suffix
