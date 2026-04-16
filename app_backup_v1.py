import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

app = Flask(__name__)

# 获取配置信息
API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")
if BASE_URL and not BASE_URL.endswith('/'):
    BASE_URL += '/'
# 构建 chat completions 端点
CHAT_URL = f"{BASE_URL}chat/completions" if BASE_URL else "https://api.openai.com/v1/chat/completions"

# 默认模型名称，如果环境变量中未提供，则使用预设值
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen-max")

# ====== 关键修改：创建一个不使用代理的 Session ======
session = requests.Session()
session.trust_env = False  # 忽略系统环境变量中的 HTTP_PROXY 和 HTTPS_PROXY

@app.route('/')
def index():
    """主页渲染"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """聊天接口，接收用户输入并返回 AI 回答"""
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "请输入内容"}), 400

    try:
        # 准备请求头和负载
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": user_message}
            ]
        }

        # 调用兼容 API，使用我们关闭了代理的 session
        response = session.post(CHAT_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # 如果返回非 200 状态码会抛出异常
        
        response_data = response.json()
        
        # 提取回复文本
        ai_response = response_data['choices'][0]['message']['content']
        return jsonify({"response": ai_response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"AI 服务调用失败: {str(e)}"}), 500

if __name__ == '__main__':
    # 允许在本地运行
    app.run(debug=True, port=5000)
