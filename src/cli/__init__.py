"""
src/cli/__init__.py

命令行模块
"""

from src.cli.menu import show_menu
from src.cli.parser import create_parser

__all__ = ["create_parser", "show_menu"]
