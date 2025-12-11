import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, TypedDict, List, Any

import chardet
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
import requests
import pymysql
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.utilities import SQLDatabase
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# 本地模块
from models import DashScopeEmbeddings

# Define LangGraph State
class ChatState(TypedDict):
    messages: List[BaseMessage]
    current_query: str
    tool_results: Dict[str, Any]
    sources: List[Dict[str, str]]
    final_response: str
    agent_scratchpad: List[BaseMessage]  # React推理过程
    tool_calls_made: int  # 工具调用次数
    max_tool_calls: int  # 最大工具调用次数

embeddings_model = DashScopeEmbeddings()


# ======================
# 全局配置与变量
# ======================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化通义千问模型
llm = ChatOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name="qwen-max",
    temperature=0.1,
    top_p=0.9
)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'mains_{datetime.now().strftime("%Y-%m-%d")}.log'
)
logger = logging.getLogger('travel')

# 数据库配置
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'travel',
    'charset': 'utf8mb4'
}

# 全局状态变量
store = {}
yuan = []
title=[]
content=[]
sourcess=[]
answers=''
chroma_collection = None
file_path='./file_path'
tianqi_key='e4b18979496f7b7e'

# ======================
# 工具函数：知识库管理
# ======================

def get_loader(file_path):
    # text=file_path.split('.')[-1].lower()
    text=os.path.splitext(file_path)[1].lower()
    print('文件扩展名：',text)
    if text=='.pdf':
        print('使用PyPDFLoader加载器')
        return PyPDFLoader(file_path)
    elif text in ['.txt','.md']:
        print('使用TextLoader加载器')
        #使用检测文件编码--chardet 进行加载
        try:
            with open(file_path,'rb') as f: #二进制方式读取
                raw_data=f.read(5000) #读取前5000字节
                result=chardet.detect(raw_data)
                #result['encoding']
                encoding=result['encoding'] or 'utf-8'
            #使用指定编码打开文件  --utf8
            return TextLoader(file_path,encoding=encoding)
        except Exception as e:
            #如果编码检测失败，使用默认编码(utf-8)打开文件
            return TextLoader(file_path)
    elif text in ['.doc','.docx']:
        return Docx2txtLoader(file_path)
    else:
        #默认使用UnstructuredPDFLoader
        print('使用UnstructuredPDFLoader加载器')
        return UnstructuredPDFLoader(file_path)

def text_spliter_add(file_path: str):
    """
    将指定文本文件加载、切分并添加到 Chroma 向量数据库中，
    同时将原始文件复制到知识库目录。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(current_dir, 'xiangliang')
    embeddings = DashScopeEmbeddings()
    os.makedirs(kb_dir, exist_ok=True)
    print(f'知识库目录：{kb_dir}')

    try:
        # 1. 加载数据
        loader =get_loader(file_path)
        doc = loader.load()

        global chunks  #切割后的文本块 全局变量，供混合检索调用
        # 2. 切割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(doc)

        # 3. 添加到向量数据库
        chroma_db_path = kb_dir

        if os.path.exists(chroma_db_path):
            print('加载已有数据库')
            vector_db = Chroma(
                persist_directory=kb_dir,
                embedding_function=embeddings,
            )
            test_count = vector_db._collection.count()
            vector_db.add_documents(chunks)
            after_count = vector_db._collection.count()
            print(f'成功追加到知识库中，本次追加了{after_count - test_count}个文本块')
        else:
            print('创建新的数据库')
            vector_db = Chroma(
                documents=chunks,
                embedding_function=embeddings,
                persist_directory=kb_dir,
            )
            print('向量数据库创建成功')

        return vector_db

    except Exception as e:
        error_msg = f'添加文件到知识库失败：{e}'
        print(error_msg)
        logger.error(error_msg)
        logger.exception(e)

        # 尝试重置数据库
        try:
            if os.path.exists(kb_dir):
                shutil.rmtree(kb_dir)
            os.makedirs(kb_dir, exist_ok=True)

            vector_db = Chroma(
                documents=chunks,
                embedding_function=embeddings,
                persist_directory=kb_dir,
            )
            print('已重置并创建新的知识库!')
            return vector_db
        except Exception as reset_e:
            logger.error(f'重置数据库失败：{reset_e}')
            logger.exception(reset_e)
            return None


def init_chroma_db(file_path):
    list_path=os.listdir(file_path)
    print(list_path)
    global chroma_collection
    if chroma_collection is None:
        for i in list_path:
            chroma_collection = text_spliter_add(file_path+'/'+i)


    return chroma_collection

# 混合检索
def hybrid_search(query: str, k: int = 10):
    global chroma_collection, chunks
    if chroma_collection is None:
        chroma_collection = init_chroma_db(file_path)
    if chunks is None:
        raise ValueError("chunks 未初始化")

    # 向量检索
    vector_retriever = chroma_collection.as_retriever(search_kwargs={'k': k})

    # BM25 检索
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # 混合检索
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.8, 0.2]
    )
    raw_results = ensemble_retriever.invoke(query)  # List[Document]

    # 如果结果为空，直接返回
    if not raw_results:
        return []

    # 初始化重排序器
    reranker = DashScopeRerank(
        model="gte-rerank",
        top_n=4,
    )

    # 执行重排序（传入原始文档列表和查询）
    reranked_docs = reranker.compress_documents(documents=raw_results, query=query)

    print('\n混合检索 + 重排序结果\n', reranked_docs)
    return reranked_docs[:k]

# ======================
# React机制工具定义
# ======================

# 工具定义字典
TOOLS_DEFINITION = {
    "chroma_search": {
        "name": "chroma_search",
        "description": "搜索知识库中的景点信息、历史介绍、文化背景等详细内容。当用户询问景点的具体介绍、历史、文化、特色等信息时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询词，应该是具体的景点名称或相关关键词"}
            },
            "required": ["query"]
        }
    },
    "sql_search": {
        "name": "sql_search",
        "description": "查询数据库中的景点基本信息，包括景点名称、位置、标签等。当用户问有哪些景点、景点列表、景点概览时使用。注意：表名是attractions，包含字段：id, name, location, capacity, established_year, tag_name, created_at, updated_at。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL查询语句，必须使用attractions表名"}
            },
            "required": ["query"]
        }
    },
    "weather_search": {
        "name": "weather_search",
        "description": "查询指定城市的天气信息。当用户询问天气、气温、下雨、晴天等天气相关问题时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "要查询天气的城市名称"}
            },
            "required": ["city"]
        }
    }
}

def format_tools_for_prompt():
    """将工具定义格式化为提示词"""
    tools_desc = []
    for tool_name, tool_info in TOOLS_DEFINITION.items():
        desc = f"""
工具名称: {tool_name}
描述: {tool_info['description']}
参数: {tool_info['parameters']}
"""
        tools_desc.append(desc)
    return "\n".join(tools_desc)

# React推理提示词
REACT_PROMPT_TEMPLATE = f"""
你是大花，一位专业的AI导游。你需要通过思考-行动-观察的循环来回答用户的问题。

你有以下工具可以使用：
{format_tools_for_prompt()}

请按照以下格式进行推理：

思考: [分析用户问题，确定需要什么信息，选择合适的工具]

行动: [工具名称]
参数: {{{{'参数名': '参数值'}}}}

观察: [工具执行的结果]

然后重复思考-行动-观察的循环，直到你能够完整回答用户的问题。

重要规则：
1. 每次只能调用一个工具
2. 必须根据观察结果继续思考或给出最终答案
3. 如果遇到|请输出以下文字|标记，直接输出[]中的内容，无需调用工具
4. 最多调用3次工具，如果仍无法回答，请根据已有信息给出回答
5. 当用户问景点详细介绍时，优先使用chroma_search
6. 当用户问有哪些景点时，使用sql_search
7. 当用户问天气时，使用weather_search
8. 使用sql_search工具时，表名必须是attractions，不是scenic_spots

现在开始回答用户问题：
"""

# ======================
# 工具函数定义
# ======================

def init_db():
    """初始化数据库连接"""
    return SQLDatabase.from_uri(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )

# 初始化数据库
db = init_db()

# LangGraph工具函数
def sql_tool(query: str, state: ChatState) -> str:
    """使用 SQLDatabase 执行 SQL 查询"""
    try:
        result = db.run(query)
        if result:
            state["sources"].append({"title": '数据库', "content": result})
            state["tool_results"]["sql_result"] = result
            return result
        else:
            return "查询结果为空"
    except Exception as e:
        logger.error(f"SQL执行错误: {str(e)}")
        return f"SQL执行错误: {str(e)}"

def query_knowledgebase(query: str, state: ChatState) -> Dict[str, Any]:
    """精确匹配景点关键词，避免混淆"""
    docs = hybrid_search(query, k=10)

    # 保留每个文档的原始内容（列表形式）
    contents_list = [d.page_content for d in docs]
    sources = [doc.metadata['source'] for doc in docs]

    print(f"检索到 {len(docs)} 个相关文档片段，来自 {len(set(sources))} 个文件")

    knowledge_sources = [
        {"title": t, "content": c}
        for t, c in zip(sources, contents_list)
    ]
    print(knowledge_sources)

    state["sources"].extend(knowledge_sources)
    context = "\n\n".join(contents_list)

    return {
        "content": context,
        "sources": sources[0] if sources else ""
    }

def chroma_search_tool(query: str, state: ChatState) -> str:
    """Chroma搜索工具"""
    try:
        print(f'开始查询用户问题：{query}')
        result = query_knowledgebase(query, state)
        content = result.get('content')
        source = result.get('sources')
        if content is None:
            content = "未找到相关信息"
        source_name = [os.path.basename(i) for i in source] if isinstance(source, list) else [
            os.path.basename(source)]
        source_info = '\n\n--\n**数据来源:' + ''.join(source_name)
        state["tool_results"]["chroma_result"] = content + source_info
        return content + source_info
    except Exception as e:
        print(f'查询用户问题失败：{e}')
        state["tool_results"]["chroma_error"] = str(e)
        return '查询用户问题失败'

def get_weather_model_client(query: str, state: ChatState) -> str:
    """天气查询工具"""
    global tianqi_key
    print(f'开始查询天气：{query}')
    url = f"https://v2.xxapi.cn/api/weather?city={query}&key={tianqi_key}"
    headers = {
        'User-Agent': 'xiaoxiaoapi/1.0.0 (https://xxapi.cn)'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        global sourcess
        #返回来源
        sourcess.append({"title": '天气api', "content": response.text})
        return response.text

    except requests.exceptions.RequestException as e:
        return f'{{"error": "请求天气API失败: {str(e)}"}}'
    except Exception as e:
        return f'{{"error": "处理天气请求时发生错误: {str(e)}"}}'


# React机制节点函数
def react_reasoning_node(state: ChatState) -> ChatState:
    """React推理节点 - 让大模型决定工具调用"""
    current_query = state["current_query"]
    messages = state["messages"]
    agent_scratchpad = state.get("agent_scratchpad", [])
    tool_calls_made = state.get("tool_calls_made", 0)
    max_tool_calls = state.get("max_tool_calls", 3)

    # 检查是否已经达到最大工具调用次数
    if tool_calls_made >= max_tool_calls:
        state["final_response"] = "已达到最大工具调用次数，请基于已有信息回答用户问题。"
        return state

    print(f"用户问题: {current_query}")

    # 检查是否是历史回顾类问题
    if "|请输出以下文字|" in current_query:
        try:
            start = current_query.find("[") + 1
            end = current_query.find("]")
            if start > 0 and end > start:
                recalled_text = current_query[start:end]
                state["final_response"] = recalled_text
                return state
        except:
            pass

    # 构建历史对话和推理过程
    conversation_history = ""
    for msg in messages[-5:]:  # 只取最近5条消息
        if isinstance(msg, HumanMessage):
            conversation_history += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_history += f"助手: {msg.content}\n"

    # 添加之前的推理步骤
    reasoning_history = ""
    for msg in agent_scratchpad:
        reasoning_history += f"{msg.content}\n"

    # 构建React提示词
    react_prompt = f"""{REACT_PROMPT_TEMPLATE}

对话历史:
{conversation_history}

用户当前问题: {current_query}

之前的推理过程:
{reasoning_history}

请继续进行思考-行动-观察的推理：
"""

    # 使用LLM进行推理
    prompt = ChatPromptTemplate.from_template("{prompt}")
    chain = prompt | llm | StrOutputParser()
    reasoning_result = chain.invoke({"prompt": react_prompt})

    print(f"LLM推理结果: {reasoning_result}")

    # 解析LLM的推理结果
    tool_call, next_thought = parse_reasoning_result(reasoning_result)

    # 将推理结果添加到scratchpad
    agent_scratchpad.append(AIMessage(content=reasoning_result))
    state["agent_scratchpad"] = agent_scratchpad
    state["tool_calls_made"] = tool_calls_made + 1

    # 如果需要调用工具
    if tool_call:
        tool_name = tool_call.get("tool")
        tool_params = tool_call.get("params", {})

        # 执行工具调用
        tool_result = execute_tool(tool_name, tool_params, state)

        # 添加观察结果到scratchpad
        observation = f"观察: {tool_result}"
        agent_scratchpad.append(AIMessage(content=observation))
        state["agent_scratchpad"] = agent_scratchpad

        print(f"工具 {tool_name} 执行结果: {tool_result}")

    # 如果LLM给出了最终答案
    elif next_thought and ("最终答案" in next_thought or "回答" in next_thought):
        final_answer = extract_final_answer(reasoning_result)
        if final_answer:
            state["final_response"] = final_answer
            return state

    return state

def parse_reasoning_result(reasoning_text: str):
    """解析LLM的推理结果，提取工具调用和下一步思考"""
    lines = reasoning_text.split('\n')
    tool_call = None
    next_thought = None

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("行动:"):
            # 提取工具名称
            tool_name = line.replace("行动:", "").strip()
            if tool_name in TOOLS_DEFINITION:
                # 查找下一行的参数
                tool_params = {}
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("参数:"):
                        try:
                            # 简单解析参数
                            params_str = next_line.replace("参数:", "").strip()
                            tool_params = eval(params_str) if params_str else {}
                        except:
                            tool_params = {}

                tool_call = {"tool": tool_name, "params": tool_params}

        elif line.startswith("思考:") and i > 0:
            next_thought = line.replace("思考:", "").strip()

    return tool_call, next_thought

def execute_tool(tool_name: str, params: dict, state: ChatState) -> str:
    """执行指定的工具"""
    try:
        if tool_name == "chroma_search":
            query = params.get("query", state["current_query"])
            result = chroma_search_tool(query, state)
            return result

        elif tool_name == "sql_search":
            query = params.get("query", "SELECT name, location, tag_name FROM attractions LIMIT 10")
            # 确保查询中使用正确的表名
            if "scenic_spots" in query:
                query = query.replace("scenic_spots", "attractions")
            result = sql_tool(query, state)
            return result

        elif tool_name == "weather_search":
            city = params.get("city", "北京")
            result = get_weather_model_client(city, state)
            return result

        else:
            return f"未知的工具: {tool_name}"

    except Exception as e:
        return f"工具执行出错: {str(e)}"

def extract_final_answer(reasoning_text: str) -> str:
    """从推理文本中提取最终答案"""
    lines = reasoning_text.split('\n')
    for line in lines:
        if "最终答案:" in line or "回答:" in line:
            return line.split(":", 1)[-1].strip()

    # 如果没有明确的最终答案标记，返回最后一个非空行
    for line in reversed(lines):
        if line.strip() and not line.startswith(("思考:", "行动:", "参数:", "观察:")):
            return line.strip()

    return ""

def response_generation_node(state: ChatState) -> ChatState:
    """生成最终回复（用于React循环结束后的最终处理）"""
    if "final_response" in state and state["final_response"]:
        return state

    # 如果没有最终回复，基于所有信息生成一个
    now = datetime.now()
    weekday_cn = "星期" + "一二三四五六日"[now.weekday()]
    date_str = now.strftime("%Y-%m-%d")
    riqi = f"{date_str} {weekday_cn}"

    system_prompt = f"""
    你是大花，一位AI导游，专注于提供中国景点的客观、中立介绍。
    日期:今天为{riqi}
    请基于之前的工具调用结果回答用户问题。
    """

    # 构建工具结果信息
    tool_info = ""
    for tool_name, result in state.get("tool_results", {}).items():
        tool_info += f"{tool_name}: {result}\n"

    # 构建推理历史
    reasoning_info = ""
    for msg in state.get("agent_scratchpad", []):
        reasoning_info += f"{msg.content}\n"

    user_message = state["current_query"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "用户问题：{user_message}\n\n工具调用结果：\n{tool_info}\n\n推理过程：\n{reasoning_info}\n\n请给出最终答案：")
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "user_message": user_message,
        "tool_info": tool_info,
        "reasoning_info": reasoning_info
    })

    state["final_response"] = response
    return state

def build_langgraph_workflow():
    """构建React机制的LangGraph工作流"""
    workflow = StateGraph(ChatState)

    # 添加节点
    workflow.add_node("react_reasoning", react_reasoning_node)
    workflow.add_node("response_generation", response_generation_node)

    # 设置入口点
    workflow.set_entry_point("react_reasoning")

    # 添加条件边：检查是否需要继续循环或生成最终答案
    def should_continue(state: ChatState) -> str:
        # 如果已经有最终答案，结束
        if "final_response" in state and state["final_response"]:
            return "end"

        # 如果达到最大工具调用次数，生成最终答案
        if state.get("tool_calls_made", 0) >= state.get("max_tool_calls", 3):
            return "generate_final"

        # 否则继续推理
        return "continue_reasoning"

    # 添加条件边
    workflow.add_conditional_edges(
        "react_reasoning",
        should_continue,
        {
            "continue_reasoning": "react_reasoning",  # 继续循环
            "generate_final": "response_generation",  # 生成最终答案
            "end": END  # 直接结束
        }
    )

    # 最终答案节点结束
    workflow.add_edge("response_generation", END)

    return workflow.compile()


# ======================
# 初始化LangGraph工作流
# ======================

def init_langgraph_workflow():
    """初始化LangGraph工作流"""
    global chroma_collection
    if chroma_collection is None:
        init_chroma_db(file_path)

    return build_langgraph_workflow()

try:
    langgraph_workflow = init_langgraph_workflow()
    logger.info('LangGraph工作流初始化成功')
except Exception as e:
    langgraph_workflow = None
    logger.error(f'LangGraph工作流初始化失败：{e}')


# ======================
# 会话与主逻辑
# ======================

def get_session_history(session_id: str):
    """获取会话历史"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def Chat(message: str, session_id: str = "default"):
    """React机制的LangGraph聊天接口"""
    try:
        # 获取会话历史
        history = get_session_history(session_id)

        # 准备初始状态
        initial_state = {
            "messages": history.messages,
            "current_query": message,
            "tool_results": {},
            "sources": [],
            "final_response": "",
            "agent_scratchpad": [],  # React推理过程
            "tool_calls_made": 0,    # 工具调用次数
            "max_tool_calls": 3      # 最大工具调用次数
        }

        print(f"用户问题: {message}")
        print("开始React推理循环...")

        # 执行React机制的LangGraph工作流
        result = langgraph_workflow.invoke(initial_state)

        # 获取最终回复
        response_text = result.get("final_response", "抱歉，无法生成回复")

        # 获取数据源信息
        source_result = result.get("sources", []).copy()

        # 更新会话历史
        history.add_user_message(message)
        history.add_ai_message(response_text)

        # 更新全局变量（为了保持兼容性）
        global answers, sourcess
        answers = response_text
        sourcess = source_result.copy()

        print(f"最终回复: {response_text}")
        print(f"工具调用次数: {result.get('tool_calls_made', 0)}")
        print(f"数据源数量: {len(source_result)}")

        return response_text, source_result

    except Exception as e:
        print(f'React LangGraph chat错误: {e}')
        logger.error(f'React LangGraph chat错误: {e}')
        logger.exception(e)
        return "抱歉，处理您的请求时出现了错误，请尝试重新表述问题。", []

def exist_user(username,password):
    try:
        conn = pymysql.connect (**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute ("SELECT * FROM user WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone ()

        cursor.close ()
        conn.close ()

        if user:
            print (f"用户验证成功: {user}")
            return True
        else:
            print ("用户不存在或密码错误")
            return False
    except Exception as e:
        print (f"数据库查询错误: {e}")
        return False