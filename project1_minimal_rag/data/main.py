"""
@Author: baiYang
@Time: 2026/1/26 19:56
@File: main.py
"""
from typing import Optional

import bs4
from langchain.agents import create_agent
from langchain_community.document_loaders import \
    WebBaseLoader  # https://docs.langchain.com/oss/python/integrations/document_loaders
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import \
    InMemoryVectorStore, VectorStore  # https://docs.langchain.com/oss/python/integrations/vectorstores
from langchain_openai import ChatOpenAI, \
    OpenAIEmbeddings  # https://docs.langchain.com/oss/python/integrations/text_embedding#interface
from langchain_text_splitters import RecursiveCharacterTextSplitter, \
    TextSplitter  # https://docs.langchain.com/oss/python/integrations/splitters
from langchain_core.tools import tool

from util.utils import import_keys

tools = []


class MiniRag:
    def __init__(self, chat_model=None, embedding=None):
        self.chat_model = chat_model
        self.embedding = embedding
        self.vector_store: Optional[VectorStore] = None
        self.text_splitter: Optional[TextSplitter] = None
        self.agent: Optional[Runnable] = None

    def set_chat_model(self, model_name: str = "gpt-4.1"):
        self.chat_model = ChatOpenAI(model=model_name)
        return self.chat_model

    def set_embedding(self, model_name: str = "text-embedding-3-large"):
        self.embedding = OpenAIEmbeddings(model=model_name)
        return self.embedding

    def set_text_splitters(self):
        # 文本结构、长度、文档结构
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        return self.text_splitter

    def create_vector_store(self):
        if self.embedding is None:
            raise RuntimeError("Embedding must be initialized before creating vector store")

        self.vector_store = InMemoryVectorStore(self.embedding)
        return self.vector_store

    @staticmethod
    def load_web_doc(url: str) -> list[Document]:
        bs4_strainer = bs4.SoupStrainer(
            class_=("post-title", "post-header", "post-content")
        )
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()
        print(f"Total characters: {len(docs[0].page_content)}")
        # print(docs[0].page_content[:500])
        return docs

    def split_doc(self, doc: list[Document]) -> list[Document]:
        if self.text_splitter is None:
            raise RuntimeError("TextSplitter is not initialized")
        all_splits = self.text_splitter.split_documents(doc)
        print(f"将博客文章拆分成{len(all_splits)} 子文档。")
        return all_splits

    def vector_store_doc(self, documents: list[Document]):
        document_ids = self.vector_store.add_documents(documents=documents)
        print(document_ids[:3])
        return document_ids

    def build_tools(self):
        retrieve_tool = build_retrieve_tool(self)

        prompt = (
            "你可以使用一个工具，可以从博客文章中提取上下文。"
            "使用该工具帮助回答用户问题。"
        )

        self.agent = create_agent(
            self.chat_model,
            tools=[retrieve_tool],
            system_prompt=prompt,
        )
        return self.agent


def build_retrieve_tool(rag: "MiniRag"):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieve_docs = rag.vector_store.similarity_search(query=query, k=2)
        serialized = "\n\n".join(
            f"源: {doc.metadata}\n内容: {doc.page_content}"
            for doc in retrieve_docs
        )
        return serialized, retrieve_docs

    return retrieve_context


if __name__ == "__main__":
    import_keys()
    rag = MiniRag()

    # 初始化模型、向量模型、存储位置
    rag.set_chat_model()
    rag.set_embedding()
    rag.set_text_splitters()
    rag.create_vector_store()

    # 加载文档
    docs = MiniRag.load_web_doc(
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    )

    # 文档拆分
    split_doc = rag.split_doc(docs)

    # 文档embedding、存储
    document_ids = rag.vector_store_doc(split_doc)

    # 初始化工具
    rag.build_tools()

    # 测试回答
    query = (
        "任务分解的标准方法是什么?\n\n"
        "一旦你得到答案，就去查查这种方法的常见扩展。"
    )

    for event in rag.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
