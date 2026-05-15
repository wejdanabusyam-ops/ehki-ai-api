from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import google.generativeai as genai
import json
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

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
ميزات تطبيق احكِ التي يمكنك اقتراحها عند الحاجة:

1. Safe Space:
مساحة آمنة للتعبير عن المشاعر، تشمل محادثات خاصة مع الطبيب، جلسات فردية صوتية، جلسات جماعية، ومجتمع للدعم والنقاش.

2. الشات الآمن مع الطبيب:
مناسب عندما يحتاج المستخدم متابعة مع مختص أو يريد الحديث بشكل خاص. الشات نصي فقط لحماية الخصوصية.

3. الجلسات الفردية:
مناسبة عندما تكون المشكلة متكررة أو تؤثر على النوم أو الدراسة أو العلاقات أو الحياة اليومية.

4. الجلسات الجماعية:
مناسبة عندما يشعر المستخدم بالوحدة أو يريد دعمًا من أشخاص يمرون بتجارب مشابهة.

5. المقالات النفسية:
مناسبة عندما يسأل المستخدم عن معلومات حول القلق، التوتر، الحزن، النوم، أو المشاعر.

6. التمارين النفسية:
تشمل تمارين تنفس، تمارين تفاعلية، وتمارين فيديو. مناسبة للتوتر، القلق، التفكير الزائد، وصعوبة النوم.

7. الدورات:
مناسبة عندما يريد المستخدم تعلم مهارات نفسية على فترة أطول.

8. الخطط العلاجية:
مناسبة عندما تكون المشكلة متكررة أو تحتاج متابعة منظمة من خلال مراحل وأهداف.

قواعد اقتراح ميزات التطبيق:
- لا تقترح كل الميزات دفعة واحدة.
- اقترح ميزة واحدة أو ميزتين فقط حسب الحالة.
- اجعل الاقتراح طبيعيًا داخل الرد وليس كإعلان.
- لا تقل إن التطبيق سيعالج المستخدم نهائيًا.
- إذا كانت الحالة طارئة، الأولوية للسلامة والتواصل مع شخص قريب أو طوارئ أو مختص.
"""

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

def save_or_update_profile(user_id, profile):
    if user_id not in profile_store:
        profile_store[user_id] = profile
    else:
        old_profile = profile_store[user_id]

        for key in profile:
            if profile[key] != "":
                old_profile[key] = profile[key]

        profile_store[user_id] = old_profile

def extract_profile_info(user_message, ai_reply, old_profile):
    prompt = f"""
استخرج معلومات مهمة عن المستخدم من الرسالة والرد.
أعد JSON فقط بدون شرح.

المطلوب:
- name: اسم المستخدم إذا ذكره
- main_issue: المشكلة الأساسية أو المتكررة
- triggers: الأشياء التي تزيد المشكلة مثل امتحانات، ناس، نوم، علاقات
- sleep_notes: أي ملاحظات عن النوم
- last_summary: ملخص قصير جدًا عن آخر حالة للمستخدم

إذا لا توجد معلومة جديدة، اجعل القيمة فارغة "".

البروفايل السابق:
{json.dumps(old_profile, ensure_ascii=False)}

رسالة المستخدم:
{user_message}

رد أنيس:
{ai_reply}

أعد JSON بهذا الشكل فقط:
{{
  "name": "",
  "main_issue": "",
  "triggers": "",
  "sleep_notes": "",
  "last_summary": ""
}}
"""

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except Exception as e:
        print("PROFILE EXTRACTION ERROR:", e)
        return {
            "name": "",
            "main_issue": "",
            "triggers": "",
            "sleep_notes": "",
            "last_summary": ""
        }

@app.get("/")
def home():
    return {"status": "API is running on Render"}

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
- عدم إعادة صياغة المشكلة في كل رد.
- عدم قول "يا صديقي" أو "يا صديقتي".
- عدم التشخيص الطبي.
- الرد من 2 إلى 4 جمل فقط.

أسلوبك:
- عملي، دافئ، ومباشر.
- أعطِ خطوة أو خطوتين قابلة للتطبيق الآن.
- إذا كان اسم المستخدم معروفًا، يمكن استخدامه أحيانًا فقط وليس دائمًا.
- إذا كانت المشكلة متكررة في البروفايل، اربط الرد بها بشكل طبيعي.

حلول مؤقتة ممكنة:
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

ميزات تطبيق احكِ المتاحة:
{app_features}

قواعد مهمة عند اقتراح ميزات التطبيق:
- لا تقترح ميزة من التطبيق في كل رد.
- اقترحها فقط عندما تكون مفيدة فعلًا.
- لا تقترح أكثر من ميزتين.
- إذا المستخدم يحتاج معلومات، اقترح المقالات.
- إذا يحتاج تهدئة أو نوم، اقترح التمارين النفسية أو Safe Space.
- إذا المشكلة متكررة أو مؤثرة، اقترح الخطط العلاجية أو الجلسة الفردية.
- إذا يشعر بالوحدة، يمكن اقتراح الجلسات الجماعية أو المجتمع داخل Safe Space.
- إذا يريد متابعة خاصة، يمكن اقتراح الشات الآمن مع الطبيب.

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

        response = gemini_model.generate_content(prompt)

        reply = (
            response.text
            if hasattr(response, "text") and response.text
            else "أنا هنا معك. جرّب أن تبدأ بخطوة صغيرة الآن."
        )

        save_message(data.user_id, "ai", reply)

        new_profile_info = extract_profile_info(
            data.text,
            reply,
            user_profile
        )

        save_or_update_profile(data.user_id, new_profile_info)

        return {"reply": reply}

    except Exception as e:
        print("ERROR:", e)
        return {"reply": f"حدث خطأ: {str(e)}"}