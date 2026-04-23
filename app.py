import os
import io
import json
import time
import wave
import uuid
import queue
import audioop
import threading
import subprocess
import requests
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 加载 .env 文件中的环境变量
load_dotenv()

app = Flask(__name__)
sock = Sock(app)

# 设置文件路径
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')
PLAYER_LOG_FILE = os.path.join(os.path.dirname(__file__), 'player.log')

def load_settings():
    """从文件加载设置"""
    default_settings = {
        'system_prompt': '',
        'initial_message': '',
        'challenge_image': ''
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                # 合并默认值，确保所有字段都存在
                return {**default_settings, **settings}
        except Exception as e:
            print(f"加载设置失败: {e}")
            return default_settings
    return default_settings

def save_settings(settings):
    """保存设置到文件"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存设置失败: {e}")
        return False


def get_client_ip():
    forwarded_for = request.headers.get('X-Forwarded-For', '')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def append_player_log(user_message, teacher_feedback, status='success', error_message=None):
    record = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'client_ip': get_client_ip(),
        'status': status,
        'user_message': user_message,
        'teacher_feedback': teacher_feedback
    }

    if error_message:
        record['error_message'] = error_message

    try:
        with open(PLAYER_LOG_FILE, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"写入 player.log 失败: {e}")

# 获取配置信息
API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")
if BASE_URL and not BASE_URL.endswith('/'):
    BASE_URL += '/'
# 构建 chat completions 端点
CHAT_URL = f"{BASE_URL}chat/completions" if BASE_URL else "https://api.openai.com/v1/chat/completions"

ASR_MIME_TYPE_TO_FORMAT = {
    'audio/ogg': 'opus',
    'audio/ogg;codecs=opus': 'opus',
    'audio/wav': 'wav',
    'audio/x-wav': 'wav',
    'audio/wave': 'wav',
    'audio/mp4': 'm4a',
    'audio/x-m4a': 'm4a',
    'audio/m4a': 'm4a',
    'audio/aac': 'aac',
    'audio/mpeg': 'mp3',
    'audio/mp3': 'mp3',
    'audio/amr': 'amr'
}

# 默认模型名称，如果环境变量中未提供，则使用预设值
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen-max")


def get_dashscope_api_key():
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("API_KEY")


def detect_asr_format(content_type, filename):
    if content_type:
        mime_type = content_type.split(';', 1)[0].strip().lower()
        detected_format = ASR_MIME_TYPE_TO_FORMAT.get(content_type.lower()) or ASR_MIME_TYPE_TO_FORMAT.get(mime_type)
        if detected_format:
            return detected_format

    if filename and '.' in filename:
        extension = filename.rsplit('.', 1)[1].lower()
        extension_to_format = {
            'wav': 'wav',
            'mp3': 'mp3',
            'm4a': 'm4a',
            'aac': 'aac',
            'amr': 'amr',
            'opus': 'opus',
            'ogg': 'opus'
        }
        detected_format = extension_to_format.get(extension)
        if detected_format:
            return detected_format

    return None


def build_wav_from_pcm(pcm_bytes, sample_width, channels, sample_rate):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def convert_audio_to_wav(audio_bytes, input_format=None):
    ffmpeg_path = '/usr/bin/ffmpeg'
    command = [
        ffmpeg_path,
        '-v', 'error'
    ]

    if input_format:
        command.extend(['-f', input_format])

    command.extend([
        '-i', 'pipe:0',
        '-ac', '1',
        '-ar', '16000',
        '-f', 'wav',
        'pipe:1'
    ])

    process = subprocess.run(
        command,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )

    if process.returncode != 0 or not process.stdout:
        error_message = process.stderr.decode('utf-8', errors='ignore').strip() or 'ffmpeg 转换失败'
        raise RuntimeError(error_message)

    return process.stdout


def normalize_audio_for_asr(audio_bytes, content_type, filename):
    lowered_content_type = (content_type or '').lower()
    lowered_filename = (filename or '').lower()

    if 'webm' in lowered_content_type or lowered_filename.endswith('.webm'):
        return convert_audio_to_wav(audio_bytes, 'webm'), 'wav', 16000

    if 'ogg' in lowered_content_type or 'opus' in lowered_content_type or lowered_filename.endswith('.ogg') or lowered_filename.endswith('.opus'):
        return convert_audio_to_wav(audio_bytes, 'ogg'), 'wav', 16000

    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
            frame_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            pcm_bytes = wav_file.readframes(wav_file.getnframes())
    except wave.Error:
        return convert_audio_to_wav(audio_bytes), 'wav', 16000

    if channels != 1:
        pcm_bytes = audioop.tomono(pcm_bytes, sample_width, 0.5, 0.5)
        channels = 1

    if frame_rate != 16000:
        converted_pcm, _ = audioop.ratecv(pcm_bytes, sample_width, channels, frame_rate, 16000, None)
        pcm_bytes = converted_pcm
        frame_rate = 16000

    normalized_wav = build_wav_from_pcm(pcm_bytes, sample_width, channels, frame_rate)
    return normalized_wav, 'wav', frame_rate


class ASRCallback(RecognitionCallback):
    def __init__(self, event_queue=None):
        self.parts = []
        self.error_message = None
        self.event_queue = event_queue
        self._last_interim_text = ''
        self._last_final_text = ''

    def on_complete(self) -> None:
        if self.event_queue:
            self.event_queue.put({
                'type': 'complete',
                'text': self.get_text()
            })
        return None

    def on_error(self, result) -> None:
        self.error_message = getattr(result, 'message', '语音识别失败')
        if self.event_queue:
            self.event_queue.put({
                'type': 'error',
                'error': self.error_message
            })

    def on_event(self, result) -> None:
        sentence = result.get_sentence() or {}
        text = (sentence.get('text') or '').strip()
        if not text:
            return

        is_final = bool(
            sentence.get('is_end')
            or sentence.get('sentence_end')
            or sentence.get('end_time')
        )

        if is_final:
            if text != self._last_final_text:
                self.parts.append(text)
                self._last_final_text = text
                if self.event_queue:
                    self.event_queue.put({
                        'type': 'final',
                        'text': text,
                        'full_text': self.get_text()
                    })
            self._last_interim_text = ''
            return

        if text == self._last_interim_text:
            return

        self._last_interim_text = text
        if self.event_queue:
            self.event_queue.put({
                'type': 'interim',
                'text': text,
                'full_text': self.get_text(),
                'display_text': self.compose_display_text(text)
            })

    def compose_display_text(self, interim_text=''):
        committed_text = self.get_text()
        return f"{committed_text}{interim_text}".strip()

    def get_text(self):
        if not self.parts:
            return ''

        deduplicated_parts = []
        for part in self.parts:
            if not deduplicated_parts or deduplicated_parts[-1] != part:
                deduplicated_parts.append(part)
        return ''.join(deduplicated_parts).strip()


class StreamingASRSession:
    def __init__(self, session_id, audio_format='opus', sample_rate=16000):
        self.session_id = session_id
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.event_queue = queue.Queue()
        self.callback = ASRCallback(self.event_queue)
        self.recognition = Recognition(
            model='paraformer-realtime-v2',
            format=audio_format,
            sample_rate=sample_rate,
            language_hints=['zh'],
            semantic_punctuation_enabled=False,
            callback=self.callback
        )
        self.started = False
        self.closed = False
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self.started:
                return
            self.recognition.start()
            self.started = True

    def send_audio(self, audio_bytes):
        if not audio_bytes:
            return
        self.start()
        with self._lock:
            if self.closed:
                return
            self.recognition.send_audio_frame(audio_bytes)

    def stop(self):
        with self._lock:
            if self.closed:
                return
            if self.started:
                self.recognition.stop()
            self.closed = True

    def fail(self, error_message):
        self.callback.error_message = error_message
        self.event_queue.put({'type': 'error', 'error': error_message})
        self.closed = True


def transcribe_audio(audio_bytes, audio_format, sample_rate=16000):
    callback = ASRCallback()
    recognition = Recognition(
        model='paraformer-realtime-v2',
        format=audio_format,
        sample_rate=sample_rate,
        language_hints=['zh'],
        semantic_punctuation_enabled=False,
        callback=callback
    )

    recognition.start()
    try:
        chunk_size = 3200
        for index in range(0, len(audio_bytes), chunk_size):
            recognition.send_audio_frame(audio_bytes[index:index + chunk_size])
        recognition.stop()
        time.sleep(1.2)
    except Exception:
        try:
            recognition.stop()
        except Exception:
            pass
        raise

    if callback.error_message:
        raise RuntimeError(callback.error_message)

    return callback.get_text()


def stream_asr_events(session):
    while True:
        event = session.event_queue.get()
        yield event
        if event.get('type') in {'complete', 'error'}:
            break

# ====== 创建具有重试机制和不使用代理的 Session ======
session = requests.Session()
session.trust_env = False  # 忽略系统环境变量中的 HTTP_PROXY 和 HTTPS_PROXY

# 配置重试策略，应对偶发的 ConnectionResetError (10054)
retry_strategy = Retry(
    total=3,  # 总共重试 3 次
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"], # 允许 POST 重试
    backoff_factor=1 # 重试退避时间: 1s, 2s, 4s...
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

@app.route('/')
def index():
    """主页渲染"""
    return render_template('index.html')

@app.route('/settings', methods=['GET'])
def get_settings():
    """获取保存的设置"""
    settings = load_settings()
    return jsonify(settings)

@app.route('/settings', methods=['POST'])
def update_settings():
    """保存设置"""
    data = request.json
    settings = {
        'system_prompt': data.get('system_prompt', ''),
        'initial_message': data.get('initial_message', ''),
        'challenge_image': data.get('challenge_image', '')
    }
    if save_settings(settings):
        return jsonify({"success": True})
    else:
        return jsonify({"error": "保存失败"}), 500

@app.route('/tts', methods=['POST'])
def tts():
    """调用 Qwen TTS 接口，将文本转为语音音频 URL"""
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "缺少文本"}), 400

    api_key = get_dashscope_api_key()
    if not api_key:
        return jsonify({"error": "未配置 API Key"}), 500

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen3-tts-flash",
        "input": {
            "text": text,
            "voice": "Ethan",
            "language_type": "Chinese"
        }
    }

    try:
        response = session.post(
            'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        response_data = response.json()
        audio_url = response_data.get('output', {}).get('audio', {}).get('url')

        if audio_url:
            return jsonify({"audio_url": audio_url})
        else:
            print("TTS Unexpected Response:", response_data)
            return jsonify({"error": "接口未返回音频 URL"}), 500

    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"error": f"语音生成失败: {str(e)}"}), 500

@sock.route('/ws/asr')
def asr_stream(ws):
    api_key = get_dashscope_api_key()
    if not api_key:
        ws.send(json.dumps({'type': 'error', 'error': '未配置 API Key'}, ensure_ascii=False))
        return

    dashscope.api_key = api_key
    session = None
    event_thread = None

    try:
        while True:
            message = ws.receive()
            if message is None:
                break

            if isinstance(message, bytes):
                if not session:
                    ws.send(json.dumps({'type': 'error', 'error': '识别会话尚未开始'}, ensure_ascii=False))
                    continue
                session.send_audio(message)
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                ws.send(json.dumps({'type': 'error', 'error': '无法解析控制消息'}, ensure_ascii=False))
                continue

            message_type = payload.get('type')
            if message_type == 'start':
                if session:
                    ws.send(json.dumps({'type': 'error', 'error': '识别会话已存在'}, ensure_ascii=False))
                    continue

                session = StreamingASRSession(
                    session_id=payload.get('sessionId') or str(uuid.uuid4()),
                    audio_format=payload.get('format') or 'opus',
                    sample_rate=int(payload.get('sampleRate') or 16000)
                )
                session.start()

                def forward_events():
                    for event in stream_asr_events(session):
                        ws.send(json.dumps(event, ensure_ascii=False))

                event_thread = threading.Thread(target=forward_events, daemon=True)
                event_thread.start()
                ws.send(json.dumps({'type': 'started', 'sessionId': session.session_id}, ensure_ascii=False))
            elif message_type == 'stop':
                if session:
                    session.stop()
                break
            else:
                ws.send(json.dumps({'type': 'error', 'error': '未知控制消息'}, ensure_ascii=False))
    except Exception as e:
        if session and not session.closed:
            session.fail(str(e))
        else:
            try:
                ws.send(json.dumps({'type': 'error', 'error': f'语音识别失败: {str(e)}'}, ensure_ascii=False))
            except Exception:
                pass
    finally:
        if session and not session.closed:
            session.stop()
        if event_thread and event_thread.is_alive():
            event_thread.join(timeout=2)


@app.route('/asr', methods=['POST'])
def asr():
    audio_file = request.files.get('audio')
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "缺少音频文件"}), 400

    api_key = get_dashscope_api_key()
    if not api_key:
        return jsonify({"error": "未配置 API Key"}), 500

    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"error": "音频内容为空"}), 400

    dashscope.api_key = api_key

    try:
        normalized_audio_bytes, audio_format, sample_rate = normalize_audio_for_asr(
            audio_bytes,
            audio_file.content_type,
            audio_file.filename
        )
        text = transcribe_audio(normalized_audio_bytes, audio_format, sample_rate)
        if not text:
            return jsonify({"error": "未识别到有效语音内容"}), 422
        return jsonify({"text": text})
    except wave.Error:
        return jsonify({"error": "上传的音频无法解析，请重试"}), 400
    except FileNotFoundError:
        return jsonify({"error": "服务器缺少 ffmpeg，当前仅建议上传 WAV/MP3 音频，或安装 ffmpeg 后再试 M4A/OGG/WEBM。"}), 500
    except Exception as e:
        print(f"ASR Error: {e}")
        return jsonify({"error": f"语音识别失败，请优先尝试 WAV、MP3、M4A 或 OGG。详情: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """聊天接口，接收用户输入并返回 AI 回答"""
    data = request.json
    user_message = data.get('message')
    system_prompt = data.get('system_prompt')

    if not user_message:
        return jsonify({"error": "请输入内容"}), 400

    try:
        # 准备请求头和负载
        headers = {
            "Authorization": f"Bearer {get_dashscope_api_key() or API_KEY}",
            "Content-Type": "application/json",
            "Connection": "close" # 尝试关闭长连接 (Keep-Alive)，减少 10054 错误
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": MODEL_NAME,
            "max_tokens": 1024,
            "messages": messages
        }

        # 调用兼容 API
        response = session.post(CHAT_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # 如果返回非 200 状态码会抛出异常
        
        response_data = response.json()
        
        # 提取回复文本
        ai_response = response_data['choices'][0]['message']['content']
        append_player_log(user_message, ai_response)
        return jsonify({"response": ai_response})

    except requests.exceptions.ConnectionError as ce:
        print(f"Connection Error: {ce}")
        return jsonify({"error": "网络连接中断，请稍后重试"}), 502
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"AI 服务调用失败: {str(e)}"}), 500

if __name__ == '__main__':
    # 允许在本地运行
    app.run(debug=True, port=5000)
