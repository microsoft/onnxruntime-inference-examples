from __future__ import annotations
import pathlib

__all__ = ['get_library_path', 'get_ep_name', 'get_ep_names']

module_dir = pathlib.Path(__file__).parent

def get_library_path() -> str:
    return str(module_dir / "basic_plugin_ep.dll")

def get_ep_name() -> str:
    return "BasicPluginEp"

def get_ep_names() -> list[str]:
    return [get_ep_name()]
