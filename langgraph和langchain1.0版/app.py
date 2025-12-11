import pymysql
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from main import Chat,store,sourcess,exist_user
app = FastAPI()
templates = Jinja2Templates (directory="templates")
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'travel',
    'charset': 'utf8mb4'
}
# 模拟AI回复
def get_ai_response(user_input: str) -> str:
   return Chat(user_input)

#清除对话
@app.post("/clear")
async def clear_conversation():

    store.clear()  # 清空全局历史
    return {"status": "success"}

@app.get ("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse ("login.html", {"request": request})

# 登录处理
@app.post ("/login")
async def login(request: Request):
    form_data = await request.form ()
    username = form_data.get ("username")
    password = form_data.get ("password")

    if not username or not password:
        return templates.TemplateResponse ("login.html", {
            "request": request,
            "error": "用户名和密码不能为空"
        })

    if exist_user(username, password):
        # 创建会话
        # session_id = str (uuid.uuid4 ())
        # sessions[session_id] = {"username": username}

        # 登录成功，重定向到主页面
        response = RedirectResponse (url="./static/index.html", status_code=302)
        # response.set_cookie (key="session_id", value=session_id)
        return response
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "用户名或密码错误"
        })
# 注册处理
@app.post ("/register")
async def register(request: Request):
    form_data = await request.form ()
    username = form_data.get ("username")
    password = form_data.get ("password")
    confirm_password = form_data.get ("confirm_password")

    if not username or not password:
        return templates.TemplateResponse ("login.html", {
            "request": request,
            "error": "用户名和密码不能为空"
        })
    if password != confirm_password:
        return templates.TemplateResponse ("login.html", {
            "request": request,
            "error": "两次输入的密码不一致"
        })
    try:
        conn = pymysql.connect (**DB_CONFIG)
        cursor = conn.cursor ()
        # 检查用户是否已存在
        cursor.execute ("SELECT * FROM user WHERE username = %s", (username,))
        if cursor.fetchone ():
            return templates.TemplateResponse ("login.html", {
                "request": request,
                "error": "用户名已存在"
            })

        # 插入新用户
        cursor.execute ("INSERT INTO user (username, password) VALUES (%s, %s)", (username, password))
        conn.commit ()

        cursor.close ()
        conn.close ()

        return templates.TemplateResponse ("login.html", {
            "request": request,
            "success": "注册成功，请登录"
        })

    except Exception as e:
        print (f"注册错误: {e}")
        return templates.TemplateResponse ("login.html", {
            "request": request,
            "error": "注册失败，请稍后重试"
        })
# 退出登录
@app.post ("/logout")
async def logout(request: Request):
    # 登录成功，重定向到主页面
    response = RedirectResponse (url="/chat", status_code=302)
    return response



@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")

    if not user_input:
        raise HTTPException(status_code=400, detail="No message provided")

    ai_response ,sources= get_ai_response(user_input)

    # 返回多轮对话历史
    # return {
    #     "response": ai_response,
    # }
    # sources = [
    #     {"title": "杭州市文旅局官网", "content": "西湖位于杭州市中心，是国家5A级景区..."},
    #     {"title": "《中国国家地理》2023年特刊", "content": "西湖十景形成于南宋时期..."},
    #     {"title": "TripAdvisor 用户评论汇总", "content": "95%的游客推荐西湖夜游..."}
    # ]



    return {
        "response": ai_response,
        "sources": sources
    }

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)