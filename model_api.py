from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")

GROQ_MODEL = "llama-3.3-70b-versatile"

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
أنت أنيس، المساعد الذكي داخل تطبيق احكِ.

شخصيتك:
- ودود وطبيعي.
- تتحدث بالعربية البسيطة.
- لا تتحدث كروبوت.
- لا تتحدث كطبيب.
- لا تستخدم لغة رسمية.
- لا تكرر كلام المستخدم.
- لا تعيد صياغة المشكلة.
- لا تستخدم عبارات محفوظة مثل:
  أفهم شعورك
  أتفهم ما تمر به
  يبدو أنك

إذا كانت الرسالة تحية فقط:
مرحبا
أهلا
السلام عليكم
هاي

فرد بتحية قصيرة وطبيعية.

مثال:
مرحباً 🌷 كيف يومك؟
أهلاً، سعيد بوجودك هنا.

إذا كانت الرسالة مشكلة:
- ركز على الرسالة الحالية.
- أعط فائدة حقيقية أو سؤالاً مفيداً.
- لا تكرر النصائح السابقة.
- لا تخرج عن السياق.

عند اقتراح ميزة من التطبيق:
- اجعل الاقتراح طبيعياً.
- لا تذكر سبب اختيار الميزة.
- لا تقترح أكثر من ميزتين.

الرد:
- قصير.
- طبيعي.
- مختلف في كل مرة.
- من 2 إلى 4 جمل.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.9,
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

def save_message(user_id, sender, message_text):
    if user_id not in memory_store:
        memory_store[user_id] = []

    memory_store[user_id].append({
        "sender": sender,
        "text": message_text
    })

def get_recent_memory(user_id, limit=3):
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
    return {"status": "API is running with Groq"}

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

مهمتك الأساسية:

- فهم الرسالة الحالية بدقة قبل الرد.
- الرد على ما يقصده المستخدم وليس فقط الكلمات المكتوبة.
- تقديم فائدة حقيقية أو خطوة عملية أو سؤال مفيد.
- استخدام الذاكرة لفهم السياق فقط.
- جعل الرد يبدو طبيعيًا وغير محفوظ.

ممنوعات صارمة:

- لا تكرر كلام المستخدم.
- لا تعيد صياغة المشكلة.
- لا تبدأ الرد بـ:
  "أفهم شعورك"
  "أتفهم ما تمر به"
  "من الطبيعي أن تشعر"
  "يبدو أنك"

- لا تعطِ نصائح لا علاقة لها بسؤاله.
- لا تخرج عن سياق الحديث.
- لا تستخدم نفس النصيحة مرتين في نفس المحادثة.
- لا تمدح المستخدم بشكل مبالغ فيه.
- لا تتعامل معه وكأنك فقط تواسيه.
- لا تكتب ردًا نمطيًا أو محفوظًا.
- لا تذكر التصنيف الداخلي.
- لا تشرح طريقة تفكيرك.

طريقة الرد:

1- افهم الرسالة الحالية أولًا.
2- إذا كان المستخدم يريد حلًا فأعطه حلًا.
3- إذا كان يريد رأيًا فأعطه رأيًا.
4- إذا كان يريد أن يفرغ ما بداخله فاستمع دون إعطاء نصائح عشوائية.
5- إذا احتجت توضيحًا فاسأل سؤالًا واحدًا فقط.

أسلوب الرد:

- طبيعي جدًا.
- ذكي ومختصر.
- مختلف في كل مرة.
- من 2 إلى 4 جمل فقط.
- ركز على الفائدة أكثر من المواساة.

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

إذا كانت رسالة المستخدم مجرد تحية قصيرة:

- لا تطلب شرح المشكلة.
- لا تسأل أسئلة رسمية.
- رد بتحية طبيعية فقط.

تعليمات حسب الحالة:
{instruction}

بروفايل المستخدم:
{profile_text}

قواعد استخدام الذاكرة:

- استخدم الذاكرة لفهم السياق فقط.
- لا تكرر معلومات من الذاكرة للمستخدم.
- لا تقل أنك تتذكر المستخدم.
- إذا سبق أن أعطيت نصيحة مشابهة فاختر طريقة مختلفة.
- إذا كان السؤال الحالي مختلفًا فتجاهل المواضيع القديمة غير المرتبطة.

قبل إرسال الرد:

- تأكد أنك لم تكرر كلام المستخدم.
- تأكد أن الرد مرتبط مباشرة برسالته الأخيرة.
- تأكد أنك لم تكرر نصيحة سابقة.
- إذا بدا الرد عامًا أو محفوظًا فأعد صياغته.

رسالة المستخدم الحالية:
{data.text}
مهم جداً:
- لا تذكر اسم الميزة التي اعتمدت عليها كسبب للرد.
- لا تذكر التصنيف الداخلي.
- لا تشرح للمستخدم لماذا اخترت هذه النصيحة.
- إذا اقترحت ميزة من التطبيق فاذكرها بشكل طبيعي ضمن الحديث.
- لا تقل "أقترح هذه الميزة لأن..."
- لا تكشف آلية اتخاذ القرار أو التفكير الداخلي.
"""

        reply = call_groq(prompt)

        save_message(data.user_id, "ai", reply)

        return {"reply": reply}

    except Exception as e:
        print("ERROR:", e)
        return {
            "reply": "حدث خطأ مؤقت في أنيس، حاول مرة أخرى بعد قليل."
        }