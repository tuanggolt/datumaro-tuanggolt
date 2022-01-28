# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

def ensure_type(t, c=None):
    c = c or t
    def _converter(v):
        if not isinstance(v, t):
            v = c(v)
        return v
    return _converter

def ensure_type_kw(c):
    def converter(arg):
        if isinstance(arg, c):
            return arg
        else:
            return c(**arg)
    return converter

def optional_cast(cls, *, conv=None):
    """
    Equivalent to:
    attrs.converters.pipe(
        attrs.converters.optional(ensure_type(t, c)),
    )

    But provides better performance (mostly, due to less function calls). It
    may sound insignificant, but in datasets, there are millions of calls,
    and function invocations start to hit.
    """

    conv = conv or cls

    def _conv(v):
        if v is not None and not isinstance(v, cls):
            v = conv(v)
        return v
    return _conv

def cast_with_default(cls, *, conv=None, factory=None):
    """
    Equivalent to:
    attrs.converters.pipe(
        attrs.converters.optional(ensure_type(t, c)),
        attrs.converters.default_if_none(factory=f),
    )

    But provides better performance (mostly, due to less function calls). It
    may sound insignificant, but in datasets, there are millions of calls,
    and function invocations start to hit.
    """

    factory = factory or cls
    conv = conv or cls

    def _conv(v):
        if v is None:
            v = factory()
        elif not isinstance(v, cls):
            v = conv(v)
        return v
    return _conv

def not_empty(inst, attribute, x):
    assert len(x) != 0, x
