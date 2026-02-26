"""
    @Author: baiYang
    @Time: 2026/2/26 11:51
    @FILE:tools
"""
from typing import Optional

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}城市天气良好"


@tool
def calculator(operation: str, a: float, b: float) -> str:
    """
    执行数学计算

    参数:
        operation: "add", "subtract", "multiply", "divide"
        a: 第个数字
        b: 第二个数字
    """
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "除数不能为0"
    }
    result = ops.get(operation, "未知操作")
    return f"{a}和{b}执行了{operation}操作，结果为{result}"


@tool
def web_search(query: str, num_results: Optional[int] = 3) -> str:
    """
    搜索网页

    参数:
        query: 搜索关键词
        num_results: 搜索结果数量，默认 3
    """
    return f"搜索了{query}，返回了{num_results}条结果"


