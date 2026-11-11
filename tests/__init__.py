"""测试包：保证从仓库根目录运行 python -m unittest discover -s tests 时可直接 import。"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 异常路径的单测会触发 logger.exception；只静音本项目 logger，保持测试输出干净
logging.getLogger("mini_agent").setLevel(logging.CRITICAL)
