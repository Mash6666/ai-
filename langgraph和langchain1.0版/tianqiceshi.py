import requests
sourcess= []
def get_weather_model_client(query: str) -> str:
    """天气查询工具"""

    print(f'开始查询天气：{query}')
    url = f"https://v2.xxapi.cn/api/weather?city={query}"
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
a=get_weather_model_client('北京')
print(a)