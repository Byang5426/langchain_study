import os
from pathlib import Path
from dotenv import load_dotenv
import getpass

from util.const import const

# 项目根目录（更稳妥）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 加载 .env
load_dotenv(PROJECT_ROOT / ".env")


def load_key(keyname: str) -> str:
    """
    优先从环境变量 / .env 读取
    不存在则交互式输入并写入 .env
    """
    value = os.getenv(keyname)
    if value:
        return value

    # 兜底：交互式输入
    value = getpass.getpass(const.Config_Load_Tip.format(keyname)).strip()
    if not value:
        raise ValueError(f"{keyname} cannot be empty")

    # 写回 .env（追加方式）
    env_path = PROJECT_ROOT / ".env"
    with open(env_path, "a", encoding="utf-8") as f:
        f.write(f"\n{keyname}={value}")

    # 同步到当前进程
    os.environ[keyname] = value
    return value