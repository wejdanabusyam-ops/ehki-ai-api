from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import os
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing.")

OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_store = {}
profile_store = {}

class RequestData(BaseModel):
    user_id: str
    text: str

category_instructions = {
    "chat": "المستخدم يحتاج احتواءً هادئًا واستماعًا.",
    "relaxation": "المستخدم يشعر بقلق أو توتر أو ضغط. قدّم خطوة عملية مؤقتة.",
    "exercise": "المستخدم يطلب تمرينًا. أعطه تمرينًا واضحًا وقصيرًا.",
    "article": "المستخدم يريد معلومة نفسية. اشرح ببساطة بدون تشخيص.",
    "emergency": "المستخدم قد يكون في خطر. ركّز على السلامة الفورية."
}

app_features = """
ميزات تطبيق احكِ:
- Safe Space: مساحة آمنة للتعبير عن المشاعر.
- الشات الآمن مع الطبيب: مناسب للمتابعة الخاصة مع مختص.
- الجلسات الفردية: مناسبة للمشكلات المتكررة أو المؤثرة على الحياة اليومية.
- الجلسات الجماعية: مناسبة للشعور بالوحدة أو الحاجة لدعم جماعي.
- المقالات النفسية: مناسبة لفهم القلق، التوتر، الحزن، النوم.
- التمارين النفسية: مناسبة للتوتر، القلق، التفكير الزائد وصعوبة النوم.
- الدورات: مناسبة لتعلم مهارات نفسية على فترة أطول.
- الخطط العلاجية: مناسبة لمتابعة منظمة عبر مراحل وأهداف.

قواعد الاقتراح:
- لا تقترح كل الميزات.
- اقترح ميزة واحدة أو ميزتين فقط.
- اجعل الاقتراح طبيعيًا وليس إعلانًا.
- لا تقل إن التطبيق يعالج نهائيًا.
"""

def call_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ehki-ai.onrender.com",
        "X-Title": "Ehki AI"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are Anis, an Arabic mental health support assistant. Always answer in Arabic."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.8,
        "max_tokens": 350
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise Exception(response.text)

    data = response.json()
    return data["choices"][0]["message"]["content"]

def save_message(user_id, sender, message_text):
    if user_id not in memory_store:
        memory_store[user_id] = []

    memory_store[user_id].append({
        "sender": sender,
        "text": message_text
    })

def get_recent_memory(user_id, limit=8):
    if user_id not in memory_store:
        return ""

    messages = memory_store[user_id][-limit:]
    memory_text = ""

    for msg in messages:
        if msg["sender"] == "user":
            memory_text += f"المستخدم: {msg['text']}\n"
        else:
            memory_text += f"أنيس: {msg['text']}\n"

    return memory_text

def get_user_profile(user_id):
    if user_id not in profile_store:
        profile_store[user_id] = {
            "name": "",
            "main_issue": "",
            "triggers": "",
            "sleep_notes": "",
            "last_summary": ""
        }

    return profile_store[user_id]

@app.get("/")
def home():
    return {"status": "API is running with OpenRouter"}

@app.post("/chat")
def chat(data: RequestData):
    try:
        vec = vectorizer.transform([data.text])
        prediction = model.predict(vec)[0]

        instruction = category_instructions.get(
            prediction,
            "رد بشكل داعم وعملي بدون تشخيص."
        )

        save_message(data.user_id, "user", data.text)

        recent_memory = get_recent_memory(data.user_id)
        user_profile = get_user_profile(data.user_id)

        profile_text = f"""
اسم المستخدم: {user_profile["name"]}
المشكلة المتكررة: {user_profile["main_issue"]}
المحفزات: {user_profile["triggers"]}
ملاحظات النوم: {user_profile["sleep_notes"]}
آخر ملخص: {user_profile["last_summary"]}
"""

        prompt = f"""
أنت "أنيس"، مساعد ذكي داخل تطبيق احكِ.

مهمتك:
- تقديم دعم عملي ومؤقت للمستخدم.
- استخدام ذاكرة المستخدم إذا كانت مفيدة.
- عدم تكرار نفس النصائح.
- عدم قول "يا صديقي" أو "يا صديقتي".
- عدم التشخيص الطبي.
- الرد من 2 إلى 4 جمل فقط.

أسلوبك:
- عملي، دافئ، ومباشر.
- أعطِ خطوة أو خطوتين قابلة للتطبيق الآن.
- إذا كانت المشكلة متكررة، اربط الرد بها بشكل طبيعي.

حلول مؤقتة:
- كتابة 3 أفكار مزعجة واختيار واحدة فقط.
- تقسيم المشكلة إلى خطوة صغيرة.
- تقليل الهاتف 10 دقائق.
- تغيير المكان أو فتح النافذة.
- شرب ماء وغسل الوجه.
- تمرين grounding.
- المشي لدقيقتين.
- عند النوم: كتابة الأفكار وتأجيلها للغد.
- عند الضغط الدراسي: مهمة واحدة لمدة 10 دقائق.
- عند التوتر الاجتماعي: تجهيز جملة قصيرة قبل الموقف.

ميزات تطبيق احكِ:
{app_features}

إذا كان التصنيف emergency:
- ركّز على السلامة فورًا.
- اطلب من المستخدم ألا يبقى وحده.
- اطلب التواصل مع شخص قريب أو جهة طوارئ أو مختص.
- لا تجعل اقتراحات التطبيق بديلًا عن السلامة الفورية.

التصنيف الداخلي:
{prediction}

تعليمات حسب الحالة:
{instruction}

بروفايل المستخدم:
{profile_text}

المحادثة السابقة:
{recent_memory}

رسالة المستخدم الحالية:
{data.text}
"""

        reply = call_openrouter(prompt)

        save_message(data.user_id, "ai", reply)

        return {"reply": reply}

    except Exception as e:
        print("ERROR:", e)
        return {
            "reply": "حدث خطأ مؤقت في أنيس، حاول مرة أخرى بعد قليل."
        }