import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict

import chardet
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
import requests
import pymysql
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.utilities import SQLDatabase
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma

# 本地模块
from models import DashScopeEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
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


# ======================
# 查询重写与上下文处理
# ======================

def rewrite_query_with_context(current_query: str, chat_history) -> str:
    if hasattr(chat_history, 'messages'):
        messages = chat_history.messages
    elif isinstance(chat_history, list):
        messages = chat_history
    else:
        messages = []

    if not messages:
        return current_query

    print(f'原始消息对象列表: {messages}')
    print(f'chat_history 类型: {type(chat_history)}')

    history_msgs = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "用户"
        elif isinstance(msg, AIMessage):
            role = "助手"
        else:
            continue
        history_msgs.append(f"{role}: {msg.content}")

    history_text = "\n".join(history_msgs[-6:])
    print(f"格式化后的历史对话:\n{history_text}")

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """
    你是专属问题重写助手，仅服务于当前用户，严格基于提供的对话历史工作，不引入任何外部信息或其他用户数据。

    核心规则（优先级从高到低，必须严格遵守）：
    1. 隔离约束：仅使用提供的「当前用户对话历史」，不添加未提及的城市、景点、人物等信息，杜绝跨会话污染。
    2. 补全省略信息（仅补必要内容，不冗余）：
       - 天气类问题：用户未提城市时，补全历史中明确提到的「景点所在城市」（无则不补，不凭空猜测）；
       - 景点类问题：用户省略景点名时，补全历史中最近提到的景点名（无则不补）；
       - 指代类问题：将“它”“这里”“那里”等指代，替换为历史中对应的景点/城市（无则保留原指代）。
    3. 无需重写场景：
       - 问题信息完整（含明确景点名、城市名，无省略或指代）；
       - 列表类问题（如“有哪些景点”“列出XX地区的景点”等）；
       - 纯问候、自我介绍类问题（如“你好”“我是XXX”）。
    4. 历史回顾类问题处理：
       - 触发场景：用户问“我之前说了什么”“你刚才回答的是什么”“重复一下刚才的内容”等；
       - 输出格式：开头必须加「|请输出以下文字|」，内容用[]包裹，如实复述历史对应内容；
       - 示例：用户问“我刚才说什么了”，历史有“用户：想了解长城”，则输出「|请输出以下文字|[你刚才提到想了解长城]」。
    5. 重写后要求：
       - 保持用户原始意图不变，语句通顺自然；
       - 长度不超过原问题2倍，无多余信息；
       - 景点名、城市名清晰无歧义，适配后续工具调用（如 chroma_search、tianqi_search）。
            """),
        ("human", f"""
    当前用户对话历史：
    {history_text}

    当前用户原始问题：
    {current_query}

    请严格按规则输出重写后的问题（无需重写则直接返回原问题，历史回顾类按格式要求输出）：
    """)
    ])

    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({})
    return rewritten
#混合检索
def hybrid_search(query: str, k: int = 10):
    global chroma_collection, chunks
    if chroma_collection is None:
        chroma_collection = init_chroma_db()
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
# 工具函数定义
# ======================

def init_langchain_agent():
    global chroma_collection
    if chroma_collection is None:
        init_chroma_db(file_path)

    # 连接 MySQL 数据库
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )

    def sql_tool(query: str) -> str:
        """使用 SQLDatabase 执行 SQL 查询"""
        try:
            result = db.run(query)
            if result:
                global sourcess
                sourcess.append({"title": '数据库', "content": result})
                return result

            else:
                return "查询结果为空"
        except Exception as e:
            logger.error(f"SQL执行错误: {str(e)}")
            return f"SQL执行错误: {str(e)}"

    def query_knowledgebase(query):
        """精确匹配景点关键词，避免混淆"""
        # docs = chroma_collection.similarity_search(query, k=3)
        docs = hybrid_search(query, k=10)

        #  保留每个文档的原始内容（列表形式）
        contents_list = [d.page_content for d in docs]  # ← 列表！
        sources = [doc.metadata['source'] for doc in docs]  # ← 列表！

        print(f"检索到 {len(docs)} 个相关文档片段，来自 {len(set(sources))} 个文件")


        global sourcess
        sourcess = [
            {"title": t, "content": c}
            for t, c in zip(sources, contents_list)
        ]
        print(sourcess)
        # 如果你还需要拼接的 context 用于 LLM，可以额外保留
        context = "\n\n".join(contents_list)

        return {
            "content": context,
            "sources": sources[0] if sources else ""
        }

    def chroma_search_tool(query):
        try:
            print(f'开始查询用户问题：{query}')
            result = query_knowledgebase(query)
            content = result.get('content')
            source = result.get('sources')
            if content is None:
                content = "未找到相关信息"
            source_name = [os.path.basename(i) for i in source] if isinstance(source, list) else [
                os.path.basename(source)]
            source_info = '\n\n--\n**数据来源:' + ''.join(source_name)
            return content + source_info
        except Exception as e:
            print(f'查询用户问题失败：{e}')
            return '查询用户问题失败'

    def get_weather_model_client(query: str):
        print(f'开始查询天气：{query}')
        url = f"https://v2.xxapi.cn/api/weather?city={query}"
        headers = {
            'User-Agent': 'xiaoxiaoapi/1.0.0 (https://xxapi.cn)'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            global sourcess
            sourcess.append({"title": '天气api', "content": response.text})

            # 解析并清洗数据
            data = response.json()
            if data.get("code") != 200:
                return json.dumps({"error": "天气API返回异常状态码"})

            forecasts = data["data"]["data"]
            # 删除第一天，并保留必要字段
            cleaned = [
                {
                    "date": item["date"],
                    "temperature": item["temperature"],
                    "weather": item["weather"],
                    "wind": item["wind"]
                }
                for item in forecasts[1:]  # 跳过第一天
            ]
            return json.dumps(cleaned, ensure_ascii=False)

        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"请求天气API失败: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"处理天气请求时发生错误: {str(e)}"})

    # 定义工具集
    tools = [
        Tool(
            name="chroma_search",
            func=lambda q: chroma_search_tool(q),
            description="当用户需求介绍时必须立即调用此工具，查询词为用户提到的具体景点名称（如'天安门','后室'），用于获取该景点的详细信息。"
        ),
        Tool(
            name="sql_db_query",
            func=lambda q: sql_tool(q),
            description="只有要求列出景点时使用,用于查询景点都有什么,查询词是根据表结构生成的sql语句"
        ),
        Tool(
            name="tianqi_search",
            func=lambda q: get_weather_model_client(q),
            description="仅当用户需要天气相关信息时使用,查询词为所在城市,请根据返回数据给出今天后天明天的相关的出行建议"
        ),
    ]
    now = datetime.now()
    weekday_cn = "星期" + "一二三四五六日"[now.weekday()]  # weekday(): 周一=0, 周日=6
    date_str = now.strftime("%Y-%m-%d")

    riqi = f"{date_str} {weekday_cn}"
    #主提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
    你是大花，一位AI导游，专注于提供中国景点的客观、中立介绍。
    你拥有返回原文本的功能,当在对话中存在|请输出以下文字|时必须返回[]中的文本,然后输出,不再调用任何工具
    所有回答基于知识库内容.
    你必须综合所有已调用工具返回的信息来回答用户问题。
    不要忽略任何工具返回的内容。
    日期:今天为{riqi},不要回答其他日期
    重要规则:
    1. 当用户询问讲解、介绍等问题时，必须使用 chroma_search 工具
    2. 当用户问到天气相关问题,直接用tianqi_search工具,返回时根据天气告诉用户穿着,饮食建议,如果天气恶略,请劝阻用户前往
    3. 只有用户问景点的简略介绍,或问有哪些景点时才用sql_db_query工具
    4.如果对话存在|请输出以下文字|使用,去除'|请输出以下文字|'剩下文本输出
    **表结构**
        CREATE TABLE attractions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL COMMENT '景点名称',
    location VARCHAR(255) COMMENT '地理位置',
    capacity BIGINT COMMENT '可容纳人数',
    established_year YEAR COMMENT '建立年份',
    tag_name VARCHAR(50) NOT NULL COMMENT '标签名称（如：自然景观、历史古迹）', 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '数据创建时间', 
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='景点基本信息表';
                    """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


# ======================
# 初始化代理
# ======================

try:
    agent_executor = init_langchain_agent()
    logger.info('Langchain代理初始化成功')
except Exception as e:
    agent_executor = None
    logger.error(f'Langchain代理初始化失败：{e}')


# ======================
# 会话与主逻辑
# ======================

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def Chat(message: str, session_id: str = "default"):
    try:
        history = get_session_history(session_id)
        rewritten_query = rewrite_query_with_context(message, history)

        print(f"原始问题: {message}")
        print(f"重写问题: {rewritten_query}")

        response = agent_executor.invoke({"input": rewritten_query})

        history.add_user_message(message)
        history.add_ai_message(response['output'])
        global answers
        global sourcess
        answers=response['output']
        source_result=sourcess.copy()
        sourcess.clear()
        return response['output'],source_result
    except Exception as e:
        print(f'chat错误: {e}')
        return "抱歉，处理您的请求时出现了错误，请尝试重新表述问题。"

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