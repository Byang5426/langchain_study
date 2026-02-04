"""
    @Author: baiYang
    @Time: 2026/2/2 15:40
    @FILE:langchain1
"""
import functools
import inspect
import json
from typing import List, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.tools import tool

from util.utils import import_keys, skip_if_not_run


def init_model():
    """
    初始化model
    :return:
    """
    model = init_chat_model(model="gpt-4.1",
                            # api_key="",
                            temperature=0.9,
                            max_tokens=1024,
                            # model_kwargs={}
                            )

    # 0.0-0.3：需要一致性、准确性的任务（数据提取、分类、代码生成）
    # 0.5-0.7：平衡创造性和一致性（聊天、问答）
    # 0.8-1.5：创造性任务（写作、头脑风暴）
    # 1.5-2.0：高度创造性（诗歌、故事创作）

    return model


@skip_if_not_run
def start_project_prompt(model: BaseChatModel, need_run=True):
    """
    invoke 三种方式
    - 格式 1：纯字符串
    - 格式 2：字典列表（推荐）
    - 格式 3：消息对象
    :return:
    """

    # 格式1 纯字符串
    str_prompt = "请用中文简洁回答：什么是python"
    str_resp = model.invoke(str_prompt)
    print("纯字符串", str_resp.content)

    # 格式2 字典列表
    dict_prompt = [{"role": "system", "content": "你是一个助手"},
                   {"role": "user", "content": "请用中文回答：什么是python"}]
    dict_resp = model.invoke(dict_prompt)
    print("字典列表", dict_resp.content)
    dict_prompt.append({"role": "assistant", "content": dict_resp.content})
    dict_prompt.append({"role": "user", "content": "我上一个问题是什么"})
    dict_resp = model.invoke(dict_prompt)
    print("字典列表,带历史", dict_resp.content)

    # 格式3 消息对象
    msg_prompt = [SystemMessage(content="你是一个助手"),
                  HumanMessage(content="请用中文回答：什么是python"), AIMessage()]
    msg_resp = model.invoke(msg_prompt)
    print("消息对象", msg_resp.content)

    msg_prompt.append(AIMessage(content=msg_resp.content))
    msg_prompt.append(HumanMessage(content="我上一个问题是什么"))
    msg_resp = model.invoke(msg_prompt)
    print("消息对象,带历史", msg_resp.content)


@skip_if_not_run
def start_project_response(model: BaseChatModel, need_run=True):
    """
    响应值的不同字段，以及含义
    - response.content              # str - AI 的回复文本
    - response.response_metadata    # dict - 响应元数据
    - response.id                   # str - 消息唯一 ID
    - response.usage_metadata       # dict - Token 使用情况
    - response.additional_kwargs    # dict - 其他额外信息
    :param model:
    :return:
    """
    if not need_run:
        current_func_name = inspect.currentframe().f_code.co_name  # 如果不需要运行直接返回
        if not need_run:
            print(f"不需要运行,{current_func_name}")
            return

    prompt = [
        {"role": "system", "content": "你是一个助手，会用简短的话回答我的问题"},
        {"role": "user", "content": "请用中文回答：什么是python"}
    ]

    rep = model.invoke(prompt)
    print(rep)
    print("*" * 50)

    print("主要内容，回复文本", rep.content)
    print("元数据", rep.response_metadata)
    print("ID", rep.id)

    print("使用情况(总)", rep.usage_metadata)
    print("使用情况(总token)", rep.usage_metadata.get("total_tokens"))
    print("使用情况(提示词token)", rep.usage_metadata.get("prompt_tokens"))
    print("使用情况(回复token)", rep.usage_metadata.get("completion_tokens"))

    print("其他信息", rep.additional_kwargs)


@skip_if_not_run
def start_project_prompt_template(model: BaseChatModel, need_run=True):
    """
    提示模板
    - PromptTemplate
    - ChatPromptTemplate
    :param model:
    :param need_run:
    :return:
    """

    # PromptTemplate格式化方法： from_template
    prompt_template_from_template = PromptTemplate.from_template(
        "将以下文本翻译成{language}：\n{text}"
    )
    print("from_template格式化", prompt_template_from_template.format(language="中文", text="hello world"))
    print("from_template 激活显示", prompt_template_from_template.invoke({"language": "英文", "text": "hello world"}))
    print("from_template输入变量", prompt_template_from_template.input_variables)
    print("*" * 50)

    # PromptTemplate格式化方法：显式输入变量
    prompt_template_explicit = PromptTemplate(
        input_variables=["language", "text"],
        template="将以下文本翻译成{language}：\n{text}"
    )
    print("显式输入变量格式化", prompt_template_explicit.format(language="英文", text="你好"))
    print("显式输入变量激活显示", prompt_template_explicit.invoke({"language": "中文", "text": "hello world"}))
    print("显式输入变量输入变量", prompt_template_explicit.input_variables)
    print("*" * 50)

    # PromptTemplate格式化方法：预填充
    prompt_template_prefill = PromptTemplate.from_template(
        "你是一个{role}，请{task}"
    )
    print("预填充输入变量前", prompt_template_prefill.input_variables)
    prompt_template_prefill = prompt_template_prefill.partial(role="sql助手")
    print("预填充格式化", prompt_template_prefill.format(task="写一个DDL"))
    print("预填充激活显示", prompt_template_prefill.invoke({"role": "Python 导师", "task": "写一个DDL"}))
    print("预填充输入变量后", prompt_template_prefill.input_variables)
    print("*" * 50)

    # ChatPromptTemplate 格式化方法：元组
    chat_prompt_template_tuples = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{role}"),
            ("user", "{question}")
        ]
    )
    print("ChatPromptTemplate格式化-元组",
          chat_prompt_template_tuples.format_messages(role="Python 导师", question="什么是装饰器？"))
    print("ChatPromptTemplate激活显示-元组",
          chat_prompt_template_tuples.invoke({"role": "Python 导师", "question": "什么是装饰器？"}))
    print("ChatPromptTemplate输入变量", chat_prompt_template_tuples.input_variables)
    print("*" * 50)

    # ChatPromptTemplate 格式化方法：message
    system_template = SystemMessagePromptTemplate.from_template(
        "你是一个{role}"
    )
    human_template = HumanMessagePromptTemplate.from_template(
        "{question}"
    )
    chat_prompt_template_message = ChatPromptTemplate.from_messages(
        [
            system_template,
            human_template
        ]
    )
    print("ChatPromptTemplate格式化-message",
          chat_prompt_template_message.format_messages(role="Python 开发人员", question="什么是装饰器？"))
    print("ChatPromptTemplate激活显示-message",
          chat_prompt_template_message.invoke({"role": "Python 开发人员", "question": "什么是装饰器？"}))
    print("ChatPromptTemplate输入变量", chat_prompt_template_message.input_variables)
    print("*" * 50)

    # ChatPromptTemplate 格式化方法：set
    chat_prompt_template_set = ChatPromptTemplate.from_messages(
        [
            {"role": "system", "content": "你是一个{role}"},
            {"role": "user", "content": "{question}"}
        ]
    )

    print("ChatPromptTemplate格式化-set",
          chat_prompt_template_set.format_messages(role="Python 爬虫人员", question="什么？"))
    print("ChatPromptTemplate激活显示-set",
          chat_prompt_template_set.invoke({"role": "Python 爬虫人员", "question": "什么？"}))
    print("ChatPromptTemplate输入变量", chat_prompt_template_set.input_variables)
    print("*" * 50)

    # ChatPromptTemplate 格式化方法：预加载
    chat_prompt_template_prefill = ChatPromptTemplate.from_messages(
        [("system", "你是{role}，目标用户是{audience}"),
         ("user", "{task}")]
    )

    print("ChatPromptTemplate预加载格式化前", chat_prompt_template_prefill.input_variables)
    chat_prompt_template_prefill = chat_prompt_template_prefill.partial(
        role="客服专员",
        audience="普通用户"
    )
    customer_support_template = chat_prompt_template_prefill.format_messages(task="如何退款")
    print("ChatPromptTemplate预加载格式化", customer_support_template)
    print("ChatPromptTemplate预加载激活显示", chat_prompt_template_prefill.invoke({"task": "如何退款"}))
    print("ChatPromptTemplate预加载格式化后", chat_prompt_template_prefill.input_variables)
    print("*" * 50)


@skip_if_not_run
def start_project_lecl(need_run=True):
    """
    LECL 通过管道服以及invoke来进行一层层的传递
    :param model:
    :param need_run:
    :return:
    """
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}"),
        ("user", "{question}")
    ])

    model = init_chat_model(model="gpt-3.5-turbo", temperature=0.9)

    LECL = template | model
    print(template.input_variables, "\n")
    print(LECL.invoke({"role": "Python 爬虫人员", "question": "什么是抓取？"}))


@skip_if_not_run
def start_project_multiple_rounds_dialogue(model: BaseChatModel, need_run=True):
    def separate_messages(messages: List[BaseMessage], max_num=3):
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        user_messages = [m for m in messages if isinstance(m, HumanMessage)]

        if len(user_messages) > max_num:
            user_messages = user_messages[:max_num]

        return system_messages + user_messages

    messages: List[BaseMessage] = [
        SystemMessage(content="你是{role}，目标用户是{audience}"),
        HumanMessage(content="我的名字叫张山，很高兴认识你"),
    ]

    template = ChatPromptTemplate.from_messages(messages)
    chain = template | model

    req = {"role": "客服专员", "audience": "普通用户"}
    res_1 = chain.invoke(req)

    print("第一轮回复", res_1.content)

    messages.append(AIMessage(content=res_1.content))
    messages.append(HumanMessage(content="我叫什么名字？"))
    messages = separate_messages(messages)
    template = ChatPromptTemplate.from_messages(messages)
    chain = template | model
    res_2 = chain.invoke(req)

    print("第二轮回复", res_2.content)


@skip_if_not_run
def start_project_tools(model: BaseChatModel, need_run=True):
    """
    工具使用
    :param model:
    :param need_run:
    :return:
    """

    # 单参数工具
    @tool
    def get_weather(city: str) -> str:
        """获取指定城市的天气"""
        return f"{city}城市天气良好"

    # 多参数工具
    @tool
    def calculator(operation: str, a: float, b: float) -> str:
        """
        执行数学计算

        参数:
            operation: "add", "subtract", "multiply", "divide"
            a: 第一个数字
            b: 第二个数字
        """
        return f"{a}和{b}执行了{operation}操作，结果为{eval(operation)}"

    # 可选参数
    @tool
    def web_search(query: str, num_results: Optional[int] = 3) -> str:
        """
        搜索网页

        参数:
            query: 搜索关键词
            num_results: 返回结果数量，默认 3
        """
        return f"搜索了{query}，返回了{num_results}条结果"

    @tool
    def search_flights(origin: str, destination: str, date: str) -> str:
        """
        搜索航班信息

        参数:
            origin: 出发城市，如"北京"
            destination: 目的地城市，如"上海"
            date: 出发日期，格式 YYYY-MM-DD

        返回:
            可用航班的 JSON 列表
        """
        res = [
            {
                "flight_number": "CA123",
                "departure_time": "2023-04-01T08:00:00",
                "arrival_time": "2023-04-01T10:00:00",
                "price": 500
            },
            {
                "flight_number": "UA456",
                "departure_time": "2023-04-01T09:00:00",
                "arrival_time": "2023-04"}
        ]
        try:
            return json.dumps(res)
        except Exception as e:
            return str(e)

    tools = [get_weather, calculator, web_search]
    model_with_tools = model.bind_tools(tools)
    res = model_with_tools.invoke("北京天气如何？")

    if res.tool_calls:
        print("AI 调用了工具", res.tool_calls)
    else:
        print("AI 直接返回", res.content)


@skip_if_not_run
def start_project_create_agent(model: BaseChatModel, need_run=True):
    """
    创建AI Agent
    :param model:
    :param need_run:
    :return:
    """

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

    tools = [get_weather, calculator, web_search]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个助手，你需要调用工具来帮助回答用户问题。"
    )

    # 正确的 invoke 输入格式
    response = agent.invoke({
        "messages": HumanMessage(content="北京天气如何？")
    })

    # 多轮对话
    second_response = agent.invoke({
        "messages": [AIMessage(content=response['messages'][-1].content),
                     HumanMessage(content="上海呢？")
                     ]
    })

    # 打印响应
    print("=== Agent 响应 ===")
    print(f"响应内容: {response['messages'][-1].content}")
    print(f"响应内容: {second_response['messages'][-1].content}")
    print("*" * 50)

    # 查看完整历史
    print("=== 完整历史 ===")
    for msg in second_response['messages']:
        print(f"{msg.__class__.__name__}: {msg.content}")
    print("*" * 50)

    # 查看工具调用
    print("=== 工具调用 ===")
    used_tools = []
    for msg in response['messages']:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                used_tools.append(tc['name'])
    print(f"使用的工具: {used_tools}")
    print("*" * 50)

    # 流式输出
    for chunk in agent.stream(
            {'messages': HumanMessage("你叫什么名字？")},
            stream_mode="values"):
        # chunk 是状态更新
        chunk["messages"][-1].pretty_print()
        # 处理最新消息
        # print(latest_msg.content)

    return ""


if __name__ == '__main__':
    import_keys()
    model = init_model()
    start_project_prompt(model, need_run=False)
    start_project_response(model, need_run=False)
    start_project_prompt_template(model, need_run=False)
    start_project_lecl(need_run=False)
    start_project_multiple_rounds_dialogue(model, need_run=False)
    start_project_tools(model, need_run=False)
    start_project_create_agent(model, need_run=True)
