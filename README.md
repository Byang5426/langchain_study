# langchain_study
# LangChain 1.0 & LangGraph 实战项目推荐指南

本指南为您推荐五个基于 **LangChain 1.0** 和 **LangGraph** 的实战项目。这些项目从入门到进阶，涵盖了从基础 RAG 到复杂多代理协作的核心场景，架构参考了 [ProjectPro](https://www.projectpro.io) 的专业标准，旨在帮助您从理论学习快速转向工业级应用开发。

---

## 架构概览 (Project Architecture Summary)

以下表格汇总了五个项目的核心架构风格和关键技术组件，体现了 LangChain 1.0 (LCEL) 和 LangGraph 在不同复杂场景下的应用。

| 项目 | 核心架构风格 | 关键组件 | 学习重点 |
| :--- | :--- | :--- | :--- |
| **项目一：极简 RAG** | **顺序链 (Sequential Chain)** | LCEL, Retriever, LLM | 熟悉 LangChain 1.0 的 LCEL 管道式开发 |
| **项目二：基础代理** | **状态机 (State Machine)** | Agent Node, Tool Node, Conditional Edge | 掌握 LangGraph 的节点、边和循环控制 |
| **项目三：Adaptive RAG** | **有向无环图 (DAG) / 自我纠错循环** | Router, Grader Agents, Query Rewriter | 实现复杂的条件路由和反馈循环 |
| **项目四：智能客服** | **多代理协作 (Multi-Agent) / 状态持久化** | Supervisor, Specialized Agents, Checkpointers | 构建多角色协作系统和人工干预机制 |
| **项目五：调研助手** | **Plan-and-Execute / 并行处理** | Planner, Researcher, Writer, Parallel Edges | 模拟人类工作流，实现任务拆解与并行执行 |

---

## 入门项目

## 项目一：LCEL 驱动的极简 RAG (基础知识问答机器人)

### 1. 项目简介 (Problem Statement)
构建一个最基础的检索增强生成（RAG）系统，用于回答关于特定文档集的问题。这个项目重点在于熟悉 **LangChain Expression Language (LCEL)** 的管道式开发模式，这是 LangChain 1.0 的核心。

### 2. 项目作用 (Business Impact / Use Case)
*   **快速验证 RAG 概念**: 适用于快速原型开发和概念验证。
*   **基础知识问答**: 可用于小型团队的内部文档问答、个人学习笔记检索等场景。

### 3. 技术栈 (Tech Stack)
*   **框架**: LangChain 1.0 (LCEL)
*   **大模型**: 任意 LLM (如 OpenAI, 免费的 Ollama 模型)
*   **向量数据库**: Chroma (本地轻量级)
*   **数据源**: 本地 Markdown 或 PDF 文件

### 4. 项目架构 (Project Architecture)
该项目采用**顺序链 (Sequential Chain)** 架构，流程清晰，通过 LCEL 的 `|` 运算符将所有组件串联成一个不可变的工作流：
1.  **Document Loader**: 加载本地文档。
2.  **Text Splitter**: 将长文档分割成小块。
3.  **Embedding**: 将文本块转换为向量并存储到 Chroma。
4.  **Retriever**: 根据用户查询检索最相关的文本块。
5.  **Prompt Template**: 格式化检索结果和用户问题。
6.  **Runnable Chain**: 使用 LCEL 将上述组件串联起来，实现一键问答。

### 5. 代码目录结构 (Project Structure)
一个典型的极简 RAG 项目结构如下，核心逻辑集中在 `main.py` 中：
```
minimal_rag/
├── data/
│   └── docs.txt          # 知识库原始文档
├── main.py               # 核心代码：数据加载、向量存储、LCEL Chain 定义与运行
├── requirements.txt      # 项目依赖
└── .env                  # 环境变量（如 API Key）
```

### 6. 核心实现步骤
1.  **数据准备**: 编写代码加载、分割和嵌入文档。
2.  **定义 Chain**: 使用 `|` 运算符将 Retriever、Prompt 和 LLM 链接起来。
3.  **输入/输出**: 运行 Chain 并观察输出。

### 7. 参考代码与资源
*   **官方教程**: [LangChain RAG 快速入门](https://python.langchain.com/docs/use_cases/question_answering/quickstart)
*   **LCEL 教程**: [LangChain Expression Language 指南](https://python.langchain.com/docs/expression_language/)

---

## 项目二：LangGraph 基础工具调用代理 (计算与搜索)

### 1. 项目简介 (Problem Statement)
构建一个能够自主决定是否使用工具的单代理系统。这是学习 LangGraph **节点 (Node)** 和 **条件边 (Conditional Edge)** 的最佳起点。代理可以处理简单问题，遇到需要外部信息或计算时，能自动调用工具并循环执行，直到得出最终答案。

### 2. 项目作用 (Business Impact / Use Case)
*   **简单任务自动化**: 适用于需要结合外部工具（如计算器、天气查询、网页搜索）的自动化任务。
*   **工具集成演示**: 掌握如何将自定义 Python 函数封装为 LLM 可调用的工具。

### 3. 技术栈 (Tech Stack)
*   **框架**: LangChain 1.0, LangGraph
*   **大模型**: 具备 Tool Calling 能力的 LLM
*   **工具**: `Tavily Search` (网页搜索) 或 `Calculator` (计算器)

### 4. 项目架构 (Project Architecture)
该项目采用**状态机 (State Machine)** 架构，包含一个简单的循环，是所有复杂代理系统的基础：
1.  **Agent Node (代理节点)**: 接收用户输入，决定是直接回答还是调用工具。
2.  **Tool Node (工具节点)**: 执行代理选择的工具。
3.  **Conditional Edge (条件边)**: 根据代理的输出（是否需要工具调用）决定下一步是回到 Agent Node（继续思考）还是结束流程。

### 5. 代码目录结构 (Project Structure)
为了清晰分离逻辑，项目结构将核心图定义和工具进行模块化：
```
basic_agent/
├── core/
│   ├── agent_graph.py    # LangGraph 图的定义、节点和边的逻辑
│   └── tools.py          # 封装的工具函数（如搜索、计算）
├── main.py               # 启动文件，运行 Agent
├── requirements.txt
└── .env
```

### 6. 核心实现步骤
1.  **定义图状态**: 仅需包含 `messages` 列表来存储对话历史。
2.  **构建节点**: 编写代理逻辑（调用 LLM）和工具执行逻辑。
3.  **设置条件边**: 实现 `should_continue` 函数，根据 LLM 的响应判断是否进入工具节点。
4.  **运行图**: 观察代理在需要工具时如何进入循环，并在获取信息后退出。

### 7. 参考代码与资源
*   **官方教程**: [LangGraph 基础 Agent 教程](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
*   **GitHub 示例**: [LangGraph Agent with Tools](https://github.com/langchain-ai/langgraph/blob/main/examples/agent_with_tools/agent_with_tools.ipynb)

---

## 进阶项目

## 项目三：Adaptive RAG (自适应检索增强生成系统)

### 1. 项目简介 (Problem Statement)
传统的 RAG 系统在面对复杂查询时往往表现不佳，容易出现检索内容不相关、模型幻觉或回答不完整等问题。**Adaptive RAG** 通过引入“自我纠错”机制，利用 LangGraph 的图结构实现对检索质量、幻觉和回答相关性的实时评估与反馈循环。

### 2. 项目作用 (Business Impact / Use Case)
*   **提升 RAG 准确率**: 适用于对回答准确性要求极高的场景，如金融、法律文档问答。
*   **企业内网搜索**: 解决企业知识库中信息冗余、质量参差不齐导致的检索失败问题。

### 3. 技术栈 (Tech Stack)
*   **框架**: LangChain 1.0 (v0.1+), LangGraph
*   **大模型**: GPT-4o / Claude 3.5 Sonnet
*   **向量数据库**: Chroma / Pinecone
*   **搜索工具**: Tavily Search API
*   **评估器**: LangChain Structured Output (Pydantic)

### 4. 项目架构 (Project Architecture)
该项目采用**有向无环图 (DAG)** 架构，核心逻辑是一个**自我纠错循环**：
1.  **Router (路由)**: 判断用户问题是需要检索本地知识库，还是直接进行网页搜索。
2.  **Retriever (检索)**: 从向量数据库获取相关文档。
3.  **Doc Grader (文档评分)**: 评估检索到的文档是否与问题相关，剔除无关噪音。
4.  **Generator (生成)**: 基于相关文档生成回答。
5.  **Hallucination Grader (幻觉检测)**: 检查生成的回答是否忠实于检索到的文档。
6.  **Answer Grader (回答评分)**: 检查回答是否真正解决了用户的问题。如果不满意，则触发 **Query Rewriter (查询重写)** 重新检索。

### 5. 代码目录结构 (Project Structure)
进阶项目需要更强的模块化，将状态、节点和图的构建分离：
```
adaptive_rag/
├── core/
│   ├── state.py          # LangGraph 状态 (TypedDict) 定义
│   ├── nodes.py          # 所有 RAG 流程节点函数（Router, Graders, Generator）
│   └── graph.py          # LangGraph 图的组装和编译
├── data/
│   └── knowledge_base/   # 知识库原始文件
├── ingest.py             # 数据预处理和向量存储创建脚本
├── main.py               # 运行 Adaptive RAG 流程
├── requirements.txt
└── .env
```

### 6. 核心实现步骤
1.  **定义状态 (State)**: 使用 `TypedDict` 定义图的状态，包含问题、文档列表、生成结果等。
2.  **构建节点 (Nodes)**: 编写检索、分级、生成和重写的函数。
3.  **设置边 (Edges)**: 使用 `add_conditional_edges` 实现基于评分结果的逻辑跳转。
4.  **编译图 (Compile)**: 将节点和边编译为可执行的 Graph。

### 7. 参考代码与资源
*   **官方示例**: [LangGraph Adaptive RAG Notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag.ipynb)
*   **进阶参考**: [Self-RAG 论文实现](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb)

---

## 项目四：Multi-Agent Customer Support (多代理智能客服系统)

### 1. 项目简介 (Problem Statement)
企业级客服需要处理多样化的任务（如查订单、改签机票、技术支持），单一 Prompt 难以胜任。本项目构建一个**多代理协作系统**，通过专门的代理处理特定领域任务，并引入 **Human-in-the-loop (人工干预)** 机制确保高风险操作（如退款）的安全性。

### 2. 项目作用 (Business Impact / Use Case)
*   **自动化办公助手**: 适用于需要多步骤、多角色协作的复杂流程，如 HR 流程、IT 故障排除。
*   **电商/航空客服**: 实现 24/7 自动分流、查询、预订等服务，显著降低人工成本。

### 3. 技术栈 (Tech Stack)
*   **框架**: LangChain 1.0, LangGraph
*   **数据库**: SQLite (用于存储模拟业务数据和对话状态)
*   **工具**: 自定义 Python 函数 (模拟 API 调用)
*   **状态管理**: LangGraph Checkpointers (实现对话中断与恢复)

### 4. 项目架构 (Project Architecture)
该项目采用**多代理协作 (Multi-Agent)** 架构，由一个主管代理和多个专业代理组成：
*   **Supervisor (主管代理)**: 负责理解用户意图并分发任务给子代理。
*   **Specialized Agents (专业代理)**:
    *   *Flight Agent*: 处理机票查询与改签。
    *   *Hotel Agent*: 处理酒店预订。
    *   *Policy Agent*: 回答公司政策相关问题。
*   **Tool Node (工具节点)**: 统一执行所有代理请求的工具调用。
*   **Interrupt (中断机制)**: 在执行敏感操作前挂起图执行，等待人工确认。

### 5. 代码目录结构 (Project Structure)
多代理系统需要清晰地分离每个代理的逻辑和工具：
```
customer_support_agent/
├── agents/
│   ├── supervisor.py     # 主管代理的逻辑和路由
│   ├── flight_agent.py   # 航班查询/改签代理
│   └── policy_agent.py   # 政策问答代理
├── tools/
│   ├── flight_api.py     # 模拟航班 API 调用工具
│   └── db_tools.py       # 数据库查询工具
├── core/
│   ├── graph.py          # LangGraph 图的组装
│   └── state.py          # 共享状态定义
├── main.py               # 启动和交互逻辑
├── requirements.txt
└── .env
```

### 6. 核心实现步骤
1.  **工具定义**: 为每个领域编写详细的工具函数及其描述。
2.  **子图构建**: 为每个专业代理创建独立的子图，提高模块化程度。
3.  **状态持久化**: 配置 `SqliteSaver` 记录每个 Thread 的状态，支持断点续传。
4.  **人工审核界面**: 实现一个简单的交互逻辑，展示待审批的操作并接收用户指令。

### 7. 参考代码与资源
*   **官方教程**: [Build a Customer Support Bot](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/)
*   **开源模板**: [Customer Support Agent by Nir Diamant](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/customer_support_agent_langgraph.ipynb)

---

## 项目五：AI Research & Report Agent (AI 深度调研与报告生成助手)

### 1. 项目简介 (Problem Statement)
撰写一份高质量的行业调研报告需要搜索大量资料、筛选信息并进行逻辑严密的总结。本项目模拟人类研究员的工作流，通过 **Plan-and-Execute (计划与执行)** 模式，自动生成结构化的长篇报告。

### 2. 项目作用 (Business Impact / Use Case)
*   **行业分析**: 快速生成市场趋势、竞争对手分析等报告，辅助商业决策。
*   **学术辅助**: 帮助学生或研究人员快速收集、整理和总结特定主题的文献资料。

### 3. 技术栈 (Tech Stack)
*   **框架**: LangChain 1.0, LangGraph
*   **搜索**: Tavily / Google Serper
*   **长文本处理**: LangChain RecursiveCharacterTextSplitter
*   **输出格式**: Markdown / PDF

### 4. 项目架构 (Project Architecture)
该项目采用 **Plan-and-Execute** 架构，将复杂任务分解为可并行执行的子任务：
1.  **Planner (计划者)**: 将大课题拆解为多个子研究点（如市场规模、竞争对手、技术趋势）。
2.  **Researcher (研究员)**: 针对每个子点进行多轮网页搜索，提取核心事实。
3.  **Reviewer (评审员)**: 检查提取的信息是否充足，是否有矛盾。
4.  **Writer (撰写者)**: 将所有信息整合，按照标准报告格式进行润色和撰写。

### 5. 代码目录结构 (Project Structure)
该结构强调计划、执行和工具的清晰分离：
```
research_agent/
├── core/
│   ├── planner.py        # 任务分解和计划生成逻辑
│   ├── researcher.py     # 研究员（执行）逻辑，包含搜索循环
│   ├── writer.py         # 报告撰写逻辑
│   └── graph.py          # LangGraph 图的组装（包含并行边）
├── tools/
│   └── search_tool.py    # 网页搜索工具
├── main.py               # 启动研究流程
├── requirements.txt
└── .env
```

### 6. 核心实现步骤
1.  **多轮搜索循环**: 利用 LangGraph 的循环结构，直到收集到足够信息才退出。
2.  **并行处理**: 使用 `Send` 协议并行启动多个子研究任务，大幅提升效率。
3.  **长上下文管理**: 动态清理状态中的冗余信息，防止超出 LLM Token 限制。

### 7. 参考代码与资源
*   **项目模板**: [RAG Research Agent Template](https://github.com/langchain-ai/rag-research-agent-template)
*   **深度参考**: [GPT-Researcher (开源标杆项目)](https://github.com/assafelovic/gpt-researcher)

---

## 总结与建议

| 项目 | 难度 | 核心学习点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **项目一：极简 RAG** | **入门** | LCEL 管道、RAG 基础流程 | 快速验证 RAG 概念、基础知识问答 |
| **项目二：基础代理** | **入门** | LangGraph 状态机、工具调用循环 | 简单任务自动化、工具集成 |
| **项目三：Adaptive RAG** | 中级 | 条件路由、自我纠错逻辑 | 知识库问答、企业内网搜索 |
| **项目四：智能客服** | 高级 | 多代理协作、状态持久化、人工干预 | 电商客服、自动化办公助手 |
| **项目五：调研助手** | 中级 | 任务拆解、并行执行、长文本生成 | 行业分析、学术辅助、内容创作 |

**上手建议**: 建议从 **项目一 (极简 RAG)** 开始，熟悉 LangChain 1.0 的 LCEL 语法。随后转到 **项目二 (基础代理)**，掌握 LangGraph 的核心概念（节点、边、状态）。最后再挑战 **项目三 (Adaptive RAG)** 和更复杂的项目。
