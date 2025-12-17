import importlib
import os
import pkgutil

from .base import ProjectionEntry, ProjectionOperator, create_projection_map, project


def _auto_import_projections():
    current_dir = os.path.dirname(__file__)

    for _, module_name, _ in pkgutil.iter_modules([current_dir]):
        if module_name not in ["base", "__init__"]:
            importlib.import_module(f".{module_name}", __name__)


_auto_import_projections()

__all__ = ["project", "ProjectionOperator", "create_projection_map", "ProjectionEntry"]
