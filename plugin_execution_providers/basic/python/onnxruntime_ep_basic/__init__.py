from __future__ import annotations
import pathlib

__all__ = ['get_library_path', 'get_ep_name', 'get_ep_names']

module_dir = pathlib.Path(__file__).parent

def get_library_path() -> str:
    candidate_paths = [
        module_dir / 'basic_plugin_ep.dll',
        module_dir / 'libbasic_plugin_ep.so',
    ]

    paths = [p for p in candidate_paths if p.is_file()]

    assert len(paths) == 1, f"Did not find exactly one library path."

    return str(paths[0])

def get_ep_name() -> str:
    return "BasicPluginExecutionProvider"

def get_ep_names() -> list[str]:
    return [get_ep_name()]
