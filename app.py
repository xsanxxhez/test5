from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from ctransformers import AutoModelForCausalLM
import os, json, re, uuid
from datetime import datetime
import requests

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key")

CHATS_DIR = "chats"
os.makedirs(CHATS_DIR, exist_ok=True)

# === Путь и URL к модели ===
MODEL_FILE = "mistral-7b.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# === Загрузка модели при необходимости ===
def download_model_if_needed():
    if not os.path.exists(MODEL_FILE):
        print("Скачиваем модель...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Модель скачана.")
    else:
        print("Модель уже скачана.")

# === Инициализация модели ===
# Загрузка модели

def load_model():
    global model
    download_model_if_needed()
   model = AutoModelForCausalLM.from_pretrained(
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=0,
    context_length=1024
)


# ⚠️ Запускаем прямо при импорте (до запуска сервера)
load_model()

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%d.%m.%Y %H:%M'):
    try:
        return datetime.fromisoformat(value).strftime(format)
    except:
        return value

def get_chat_history(chat_id):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_chat_message(chat_id, user_input, response):
    history = get_chat_history(chat_id)
    history.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "user": user_input,
        "bot": response
    })
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def format_response(text):
    text = re.sub(r'\n\n+', '</p><p>', text.strip())
    return f"<p>{text}</p>"

def generate_response(user_input):
    if not model:
        return "Модель не загружена"
    prompt = f"""Ты — дружелюбный ИИ ассистент. Отвечай кратко и по делу.

Вопрос: {user_input}
Ответ:"""
    try:
        output = model(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9)
        return format_response(output)
    except Exception as e:
        print(f"Ошибка генерации: {e}")
        return "Ошибка генерации"

@app.route("/")
def home():
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
    history = get_chat_history(session['chat_id'])
    chats = []
    for fname in os.listdir(CHATS_DIR):
        if fname.endswith(".json"):
            cid = fname[:-5]
            h = get_chat_history(cid)
            if h:
                chats.append({
                    "id": cid,
                    "title": h[0]["user"][:30],
                    "date": h[0]["timestamp"]
                })
    return render_template("index.html", history=history, chats=chats)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "Пустой запрос"}), 400
    chat_id = session.get("chat_id", str(uuid.uuid4()))
    session["chat_id"] = chat_id
    response = generate_response(user_input)
    save_chat_message(chat_id, user_input, response)
    return jsonify({"response": response})

@app.route("/new_chat")
def new_chat():
    session['chat_id'] = str(uuid.uuid4())
    return redirect(url_for("home"))

@app.route("/load_chat/<chat_id>")
def load_chat(chat_id):
    if os.path.exists(os.path.join(CHATS_DIR, f"{chat_id}.json")):
        session["chat_id"] = chat_id
    return redirect(url_for("home"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
