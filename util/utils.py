import functools
import os
from typing import Dict

import os

from config.config import load_key
from util.const import const

KEY_MAPPING: Dict[str, str] = {
    "OPENAI_API_KEY": const.Config_OpenAI_API_Key,
    "OPENAI_BASE_URL": const.Config_OpenAI_Base_URL,
    "HUGGINGFACEHUB_API_TOKEN": const.Config_HuggingFaceHub_API_Token,
    "LANGSMITH_TRACING": const.Config_LANGSMITH_TRACING,
    "LANGSMITH_API_KEY": const.Config_LANGSMITH_API_KEY,
}


def import_keys(overwrite: bool = False) -> None:
    """

    :param overwrite:
    :return:
    """
    for env_key, config_key in KEY_MAPPING.items():
        if not overwrite and os.getenv(env_key):
            continue
        os.environ[env_key] = load_key(config_key)


def skip_if_not_run(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("-" * 50)
        # 兼容 need_run 作为 位置参数 或 关键字参数
        need_run = kwargs.get("need_run")

        # 如果没传 keyword，就尝试从位置参数中取
        if need_run is None and len(args) >= 2:
            need_run = args[1]

        if not need_run:
            print(f"不需要运行, {func.__name__}")
            return None

        return func(*args, **kwargs)

    return wrapper


if __name__ == '__main__':
    import_keys()
