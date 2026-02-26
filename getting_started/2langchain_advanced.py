"""
    @Author: baiYang
    @Time: 2026/2/4 16:41
    @FILE:langchain_advanced
"""
from typing import List

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import false

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
def start_langchain_memory(model: BaseChatModel, need_run=True):
    """
    携带记忆
    - 短期记忆
    - 长期记忆
    :return:
    """
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "conversation_1"}}

    first_resp = agent.invoke({
        "messages": [AIMessage(content="你好"), HumanMessage(content="我叫张三，你好")]}
        , config=config)

    second_resp = agent.invoke({
        "messages": [HumanMessage(content="我叫什么")]}
        , config=config
    )

    print(first_resp["messages"][-1].content)
    print("=== Agent 响应 ===")
    print(f"响应内容: {first_resp['messages'][-1].content}")
    print(f"响应内容: {second_resp['messages'][-1].content}")
    print("*" * 50)


@skip_if_not_run
def start_langchain_multiplayer_memory(model: BaseChatModel, thread_ids: List[str] = ["conversation_1"], need_run=True):
    """
    多人聊天，内存记忆
    :param model:
    :return:
    """
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=InMemorySaver(),
    )

    for ThreadId in thread_ids:
        # 创建多个会话配置
        config_user1 = {"configurable": {"thread_id": ThreadId}}
        resp_user1 = agent.invoke({"messages": [HumanMessage(content="你好")]}, config=config_user1)
        if resp_user1:
            print(f"{ThreadId}会话: {resp_user1['messages'][-1].content}")

@skip_if_not_run
def start_langchain_context_manager(model: BaseChatModel, need_run=True):
    """
    使用上下文管理器
    :param model:
    :return:
    """


if __name__ == '__main__':
    import_keys()
    model = init_model()
    start_langchain_memory(model, need_run=False)
    start_langchain_multiplayer_memory(model, thread_ids=["user1", "user2"], need_run=False)
