"""
    @Author: baiYang
    @Time: 2026/1/29 19:15
    @FILE:damo
"""
from langchain_core.tools import tool
from langchain import chat_models

model = chat_models.init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0
)


@tool
def multiply(a: int, b: int) -> int:
    """把'a'和'b'相乘。

    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """加上“a”和“b”。

    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """除以'a'和'b'。

    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a / b


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
