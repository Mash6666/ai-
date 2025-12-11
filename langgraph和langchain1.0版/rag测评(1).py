# evaluate_with_ragas.py

import os
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量（用于配置API密钥等）
load_dotenv()

# 导入Ragas评估框架相关模块
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset

# 导入自定义模块
from main import hybrid_search, Chat
from models import get_ali_clients

llm_client, embeddings_client = get_ali_clients()


def rags_evaluation():
    # 测试数据：确保每个问题都有有效内容
    test_data = [
        {
            "question": "长城有多长？",
            "ground_truth": "长城的全长约为一万三千多里，也就是大约6700公里。"
        },
        {
            "question": "天安门明代叫什么？",
            "ground_truth": "承天门"
        },
        {
            "question": "长城和故宫哪一个历史更久？",
            "ground_truth": "长城"
        },
        {
            "question": "什么是北京中轴线上唯一的宗教性建筑？",
            "ground_truth": "钦安殿"
        },
        {
            "question": "什么是金凤颁诏？",
            "ground_truth": "明清两朝在天安门城楼上举行的一种隆重的仪式，用于皇帝向全国颁布诏书。"
        },

        {
            "question": "金水河上的桥都有哪些？",
            "ground_truth": "金水河上共有七座汉白玉石桥，这些桥根据其位置和用途有着不同的等级。"
        },
        {
            "question": "故宫什么时候建成的",
            "ground_truth": "建成于明朝永乐十八年（公元1420年）"},
        {
            "question": "长城什么时候开始建造的？",
            "ground_truth": "修建始于春秋战国时期。"
        },
        {
            "question": "颐和园占地有多大？",
            "ground_truth": "290公顷"
        },
        {
            "question": "谁正式把颐和园改成对外开放的公园",
            "ground_truth": "民国时期,北洋政府"
        },
        {
            "question": "乐寿堂的西跨院叫什么？",
            "ground_truth": "扬仁风"
        },
        {
            "question": "德和园的名字取自哪里？",
            "ground_truth": "取自《左传》中的 君子听之以平其心，心平德和"
        },
        {
            "question": "乐寿堂殿内“慈晖懿祉”的匾额意思是什么？",
            "ground_truth": "受母后之深恩，托母后之洪福"
        },
        {
            "question": "排云殿名字出自哪里？",
            "ground_truth": "出自晋代诗人郭璞：“神仙排云出，但见金银台”的诗句"
        }
        ,
        {
            "question": "“蕃厘经纬”的匾额意思是什么？",
            "ground_truth": "幸福无边"
        },
        {
            "question": "智慧海名字出自哪里？",
            "ground_truth": "名称来自《无量寿经》中 如来智慧海，身府无崖底。"
        },
        {
            "question": "朱元璋的坟在哪里？",
            "ground_truth": "取在南京明孝陵"
        },
        {
            "question": "秦始皇为什么修筑长城",
            "ground_truth": "防御北方匈奴的入侵"
        },
        {
            "question": "定陵是谁的陵寝？",
            "ground_truth": "神宗万历朱翊钧和孝端、孝靖两位皇后的合葬墓"
        },
        {
            "question": "定陵地宫挖掘工作是什么时候开始的？",
            "ground_truth": "1956年5月开始的"
            },
        {
            "question": "德和园名字出自哪里？",
            "ground_truth": "君子听之以平其心，心平德和"
        },
    ]

    # 初始化存储列表
    questions = []
    answers = []
    contexts = []  # 每个元素应是一个字符串列表，如 ["text1", "text2"]
    ground_truths = []

    # 遍历测试数据
    for item in test_data:
        question = item["question"].strip()
        if not question:  # 跳过空问题
            continue

        print(f"\n处理问题: {question}")

        # 获取 RAG 系统的回答（Chat 应返回字符串）
        try:
            answer = Chat(message=question)
            if isinstance(answer, (list, tuple)):
                answer = " ".join(str(a) for a in answer)  # 如果返回 list，合并为字符串
            else:
                answer = str(answer)
        except Exception as e:
            print(f"生成回答失败: {e}")
            answer = "无法生成回答"

        print(f"回答: {answer}")

        # 使用 hybrid_search 检索上下文
        try:
            docs = hybrid_search(question)
            if docs:
                # 提取所有检索到的文档内容
                retrieved_texts = [doc.page_content.strip() for doc in docs]
            else:
                retrieved_texts = ["未检索到相关内容"]
        except Exception as e:
            print(f"检索上下文失败: {e}")
            retrieved_texts = ["检索失败"]

        # 添加到对应列表
        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_texts)  # 注意：这里是一个字符串列表
        ground_truths.append(item["ground_truth"])

    # ✅ 所有数据收集完毕后，再构建 dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,         # Ragas 要求 contexts 是 List[str] 的列表
        "ground_truth": ground_truths
    })

    print("\n构建的评估数据集:")
    print(dataset)
    print(dataset[0])

    # 执行 Ragas 评估
    try:
        result = evaluate(
            dataset=dataset,  # 评估数据集
            metrics=[  # 使用的评估指标
                answer_relevancy,  # 答案相关性
                faithfulness,  # 忠实度
                context_precision,  # 上下文精确率
                context_recall,  # 上下文召回率
            ],
            llm=llm_client,  # 用于评估的大语言模型
            embeddings=embeddings_client,  # 用于文本嵌入的模型
            raise_exceptions=False  # 遇到错误不中断，继续执行
        )

        # 转换为 pandas DataFrame
        df = result.to_pandas()
        print("\n评估结果:")
        print(df)

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("evaluation_results", exist_ok=True)
        csv_path = f"evaluation_results/ragas_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n评估结果已保存至: {csv_path}")

        return df

    except Exception as e:
        print(f"评估执行失败: {e}")
        return None


if __name__ == "__main__":
    rags_evaluation()