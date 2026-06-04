from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import requests
import random

# =========================
# إعدادات Groq
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")

GROQ_MODEL = "llama-3.3-70b-versatile"

# =========================
# تحميل المودل
# =========================

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# FastAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# تخزين الذاكرة
# =========================

memory_store = {}
profile_store = {}

# =========================
# Request Model
# =========================

class RequestData(BaseModel):
    user_id: str
    text: str

# =========================
# تعليمات التصنيف
# =========================

category_instructions = {
    "chat": "رد بشكل طبيعي ومريح.",
    "relaxation": "قدّم خطوة بسيطة لتخفيف التوتر أو القلق.",
    "exercise": "أعط تمرينًا نفسيًا قصيرًا وواضحًا.",
    "article": "اشرح الفكرة ببساطة بدون تشخيص.",
    "emergency": "ركّز على السلامة واطلب التواصل مع شخص قريب أو مختص."
}

# =========================
# ميزات التطبيق
# =========================

app_features = """
ميزات تطبيق احكِ:
- Safe Space
- الشات الآمن مع الطبيب
- الجلسات الفردية
- الجلسات الجماعية
- المقالات النفسية
- التمارين النفسية
- الدورات
- الخطط العلاجية

قواعد الاقتراح:
- لا تقترح أكثر من ميزتين.
- اجعل الاقتراح طبيعيًا.
- لا تجعل الرد يبدو إعلانًا.
"""

# =========================
# التحيات
# =========================

greetings = [
    "مرحبا",
    "مرحباً",
    "اهلا",
    "أهلاً",
    "السلام عليكم",
    "هاي",
    "hi",
    "hello"
]

greeting_replies = [
    "أهلاً 🌷 كيف يومك؟",
    "مرحباً، سعيد بوجودك هنا.",
    "أهلاً ✨ كيف الأمور معك؟",
    "هاي 🌸"
]

# =========================
# استدعاء Groq
# =========================

def call_groq(prompt):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,

        "messages": [
            {
                "role": "system",
                "content": """
أنت أنيس، مساعد ذكي داخل تطبيق احكِ.

أسلوبك:
- طبيعي وهادئ.
- تتحدث بالعربية البسيطة.
- لا تتحدث كطبيب.
- لا تستخدم لغة رسمية.
- لا تكرر كلام المستخدم.
- لا تعيد صياغة المشكلة.
- لا تستخدم عبارات محفوظة مثل:
  أفهم شعورك
  يبدو أنك
  أتفهم ما تمر به

طريقة الرد:
- ركز على الرسالة الحالية فقط.
- أعط فائدة حقيقية أو سؤالًا مفيدًا.
- إذا احتاج المستخدم خطوة عملية فأعطه خطوة صغيرة وواضحة.
- لا تكرر نفس النصيحة.
- لا تخرج عن السياق.
- اجعل الرد قصيرًا من 2 إلى 4 جمل.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": 0.6,
        "max_tokens": 180
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(response.text)

    data = response.json()

    return data["choices"][0]["message"]["content"]

# =========================
# حفظ الرسائل
# =========================

def save_message(user_id, sender, message_text):

    # تجاهل الرسائل القصيرة جدًا
    if len(message_text.strip()) < 6:
        return

    if user_id not in memory_store:
        memory_store[user_id] = []

    memory_store[user_id].append({
        "sender": sender,
        "text": message_text
    })

    # الاحتفاظ بآخر 12 رسالة فقط
    memory_store[user_id] = memory_store[user_id][-12:]

# =========================
# الذاكرة الحديثة
# =========================

def get_recent_memory(user_id, limit=4):

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

# =========================
# بروفايل المستخدم
# =========================

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

# =========================
# Home Route
# =========================

@app.get("/")
def home():
    return {"status": "API is running with Groq"}

# =========================
# Chat Route
# =========================

@app.post("/chat")
def chat(data: RequestData):

    try:

        user_text = data.text.strip()

        # =========================
        # كشف التحيات
        # =========================

        if user_text.lower() in greetings:

            return {
                "reply": random.choice(greeting_replies)
            }

        # =========================
        # تصنيف الرسالة
        # =========================

        vec = vectorizer.transform([user_text])

        prediction = model.predict(vec)[0]

        instruction = category_instructions.get(
            prediction,
            "رد بشكل داعم وعملي بدون تشخيص."
        )

        # =========================
        # حفظ رسالة المستخدم
        # =========================

        save_message(data.user_id, "user", user_text)

        # =========================
        # الذاكرة
        # =========================

        recent_memory = get_recent_memory(data.user_id)

        user_profile = get_user_profile(data.user_id)

        profile_text = f"""
اسم المستخدم: {user_profile["name"]}
المشكلة المتكررة: {user_profile["main_issue"]}
المحفزات: {user_profile["triggers"]}
ملاحظات النوم: {user_profile["sleep_notes"]}
آخر ملخص: {user_profile["last_summary"]}
"""

        # =========================
        # البرومبت
        # =========================

        prompt = f"""
تعليمات إضافية:
{instruction}

بروفايل المستخدم:
{profile_text}

آخر المحادثة:
{recent_memory}

ميزات التطبيق:
{app_features}

قواعد مهمة:
- استخدم الذاكرة لفهم السياق فقط.
- لا تقل أنك تتذكر المستخدم.
- لا تكرر نفس النصيحة السابقة.
- لا تذكر التصنيف الداخلي.
- لا تشرح طريقة تفكيرك.
- إذا اقترحت ميزة من التطبيق اجعلها طبيعية داخل الكلام.

رسالة المستخدم الحالية:
{user_text}
"""

        # =========================
        # الرد
        # =========================

        reply = call_groq(prompt)

        # تنظيف الرد
        reply = reply.strip()

        # منع الردود الطويلة جدًا
        if len(reply) > 500:
            reply = reply[:500]

        # حفظ رد أنيس
        save_message(data.user_id, "ai", reply)

        return {
            "reply": reply
        }

    except Exception as e:

        print("ERROR:", e)

        return {
            "reply": "حدث خطأ مؤقت في أنيس، حاول مرة أخرى بعد قليل."
        }
```
