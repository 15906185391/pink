"""
magicbot 子包初始化，可在此导出主要类或工具。
"""
# 示例：假设 magicbot.py 中有 MagicBot 类
# Ensure magicbot.py exists in this directory, or update the import to the correct module name
# Example: If the file is named magic_bot.py, use:
# from .magic_bot import MagicBot
from .magicbot_gen1 import MagicBotGen1
from .magicbot import MagicBot

__all__ = [
    "MagicBotGen1",
    "MagicBot",
]