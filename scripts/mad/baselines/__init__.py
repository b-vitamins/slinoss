"""Optional MAD baseline adapters.

Modules in this package register one external mixer when explicitly imported through
``scripts.mad.run --mixer-module``.  Importing :mod:`scripts.mad.mixers` itself never
pulls an optional research dependency into the SLinOSS environment.
"""
