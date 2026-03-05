# new/runner/cubicasa_detector/__init__.py
try:
    from .jobs import run_cubicasa_for_plan
except ImportError:
    run_cubicasa_for_plan = None  # e.g. torch not installed when only using raster_processing

__all__ = ["run_cubicasa_for_plan"]