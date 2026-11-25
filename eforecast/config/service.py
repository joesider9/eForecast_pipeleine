# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Configuration service: kept as the name 60 modules already import.

This module was 620 lines holding two unrelated things -- the case document (validate it, read
and write it) and the workspace (where cases live, who may look at them) -- plus the two errors
everything else wanted. Thirty modules imported it; 31 places wanted ``ServiceError`` alone, and
``run_model`` was one of them -- so a module with no interest in case documents imported the
editing operations to get an exception class.

The three parts now live where they belong:

``eforecast.config.errors``
    ``ServiceError``, ``WorkspaceNotConfigured``. Imports nothing.
``eforecast.config.case_document``
    Validation, preview, documents on disk, the presets of a variable.
``eforecast.config.workspace``
    ``Workspace``, ``workspace_root``, browsing and the allowed roots.

Nothing moved out of reach: every name this module used to export is re-exported here, so no
caller had to change in the commit that split it. New code should import from the three modules
above; this one exists so that the split cost nothing to land, and so that `python -m` entry
points and installed units keep working.

It does not make imports cheaper: ``eforecast/config/__init__.py`` imports ``loader`` and
``schema`` eagerly, so every ``eforecast.config.X`` import pays 99 ms whichever of the four it
asks for. That is the package's cost, not this module's, and it is a separate change.

One thing a re-export cannot carry is a *patch*: ``monkeypatch.setattr(service, 'X', ...)``
rebinds the name here, while the function that reads it now lives in ``workspace``. Patch the
module that defines the value.
"""
from __future__ import annotations

import os                                        # noqa: F401  (tests patch service.os.path)

from eforecast.config.case_document import (SECTION_MODELS, apply_data_types, defaults, derived_facts, load_case,
                                            merge_sections, preview_static_data, save_case, save_sections, schema,
                                            sync_document, validate, validate_section, validation_errors,
                                            variable_presets, write_document)
from eforecast.config.errors import ServiceError, WorkspaceNotConfigured
from eforecast.config.workspace import (CASE_NAME, DEFAULT_WORKSPACE, LAB_WORKSPACES, MANAGING_ROLES, TENANT_ROLES,
                                        WORKSPACE_REMEDY, Workspace, _default_workspace, _within, browse,
                                        browse_roots, inside_roots, workspace_root)

__all__ = [
    # errors
    "ServiceError", "WorkspaceNotConfigured",
    # the case document
    "SECTION_MODELS", "apply_data_types", "defaults", "derived_facts", "load_case", "merge_sections",
    "preview_static_data", "save_case", "save_sections", "schema", "sync_document", "validate",
    "validate_section", "validation_errors", "variable_presets", "write_document",
    # the workspace
    "CASE_NAME", "DEFAULT_WORKSPACE", "LAB_WORKSPACES", "MANAGING_ROLES", "TENANT_ROLES", "WORKSPACE_REMEDY",
    "Workspace", "browse", "browse_roots", "inside_roots", "workspace_root",
]
