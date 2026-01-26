import os
from typing import Dict

import os

from config.config import load_key
from util.const import const

KEY_MAPPING: Dict[str, str] = {
    "OPENAI_API_KEY": const.Config_OpenAI_API_Key,
    "OPENAI_BASE_URL": const.Config_OpenAI_Base_URL,
    "HUGGINGFACEHUB_API_TOKEN": const.Config_HuggingFaceHub_API_Token,
}


def import_keys(overwrite: bool = False) -> None:
    """"""
    for env_key, config_key in KEY_MAPPING.items():
        if not overwrite and os.getenv(env_key):
            continue
        os.environ[env_key] = load_key(config_key)

if __name__ == '__main__':
    import_keys()
