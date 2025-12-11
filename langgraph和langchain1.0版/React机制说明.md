# React机制说明

## 概述

本项目已成功集成了React机制，让大模型自主决定工具调用并支持多轮循环推理。

## 主要改进

### 1. React推理机制
- **思考-行动-观察循环**：大模型通过分析用户问题，自主选择合适的工具
- **智能工具选择**：根据问题类型自动选择chroma_search、sql_search或weather_search
- **多轮循环支持**：允许最多3次工具调用，支持复杂问题的逐步解决

### 2. 新增状态字段
```python
class ChatState(TypedDict):
    # 原有字段...
    agent_scratchpad: List[BaseMessage]  # React推理过程记录
    tool_calls_made: int  # 工具调用次数
    max_tool_calls: int  # 最大工具调用次数限制
```

### 3. 核心函数

#### react_reasoning_node(state)
- React推理的核心节点
- 分析用户问题并决定工具调用
- 支持循环推理直到获得满意答案

#### parse_reasoning_result(reasoning_text)
- 解析LLM的推理输出
- 提取工具名称和参数
- 识别最终答案

#### execute_tool(tool_name, params, state)
- 执行具体的工具调用
- 支持chroma_search、sql_search、weather_search
- 处理工具执行异常

## 工具定义

### chroma_search
- **用途**：搜索知识库中的景点详细信息
- **场景**：用户询问景点历史、文化背景、特色介绍
- **参数**：query（搜索关键词）

### sql_search
- **用途**：查询数据库中的景点基本信息
- **场景**：用户问有哪些景点、景点列表、概览
- **参数**：query（SQL语句）

### weather_search
- **用途**：查询城市天气信息
- **场景**：用户询问天气、气温、下雨等
- **参数**：city（城市名称）

## 工作流程

```
用户问题 → 查询重写 → React推理循环
                     ↓
              思考-行动-观察
                     ↓
              是否需要更多信息？
                     ↓
            是 → 继续工具调用
            否 → 生成最终答案
```

## 使用示例

```python
from main import Chat

# 基本使用
response, sources = Chat("故宫有什么历史背景？")

# 复杂问题（会自动触发多轮工具调用）
response, sources = Chat("北京今天天气怎么样？适合去故宫游玩吗？")

# 景点列表查询
response, sources = Chat("有哪些著名的历史古迹？")
```

## 日志输出

React机制会输出详细的推理过程：
- 用户问题
- 查询重写结果
- LLM推理结果
- 工具调用情况
- 工具执行结果
- 最终答案

## 配置参数

- **max_tool_calls**：最大工具调用次数（默认3次）
- **temperature**：LLM温度参数（默认0.1）
- **model_name**：使用的模型（默认qwen-max）

## 注意事项

1. 确保知识库已正确初始化
2. 检查API密钥配置
3. 监控工具调用次数避免超限
4. 根据需要调整max_tool_calls参数





## 优势

1. **智能化**：大模型自主决策，无需硬编码规则
2. **灵活性**：支持复杂问题的逐步解决
3. **可扩展**：易于添加新工具和功能
4. **透明性**：完整的推理过程记录
5. **健壮性**：异常处理和回退机制