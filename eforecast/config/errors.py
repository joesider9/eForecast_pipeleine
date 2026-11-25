# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The two errors the configuration service raises, and nothing else.

``ServiceError`` is wanted in 31 places and ``WorkspaceNotConfigured`` in 7, including
``run_model``, which has no business with case documents at all. While both were defined in
``service``, a module that only wanted to say "that name is not allowed" had to import the
workspace and the document editing operations with them. Now it does not: this module imports
nothing.

What it does *not* yet buy is a cheaper import. ``eforecast/config/__init__.py`` imports
``loader`` and ``schema`` eagerly, which costs 158 ms of pydantic and schema building, and every
``eforecast.config.X`` import pays it -- measured, after this split, as 99 ms cumulative for
``errors``, ``workspace``, ``case_document`` and ``service`` alike. Make that package import lazy
and asking for an exception class becomes free. Until then the gain here is structural, not fast.
"""
from __future__ import annotations


class ServiceError(ValueError):
    """A user-facing problem (bad name, unknown case, path outside the allowed roots)."""


class WorkspaceNotConfigured(RuntimeError):
    """There is nowhere to keep users, tenants and cases, and guessing is not an option.

    Raised where the workspace is first needed, never at import, so ``--help``, the schema
    commands and ordinary imports keep working on a machine that has no workspace yet.
    """
