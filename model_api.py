from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import google.generativeai as genai
import pyodbc
import json

import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

conn_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=(localdb)\\MSSQLLocalDB;"
    "DATABASE=Ehki_FinalDatabase;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

def get_connection():
    return pyodbc.connect(conn_str)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
مناسب عندما يحتاج المستخدم متابعة مع مختص أو يريد الحديث بشكل خاص. الشات نصي فقط لحماية الخصوصية، ولا يسمح بإرسال الصور أو الملفات أو الموقع.

3. الجلسات الفردية:
مناسبة عندما تكون المشكلة متكررة أو تؤثر على النوم أو الدراسة أو العلاقات أو الحياة اليومية. يمكن حجز جلسة مع مختص وتحديد وقت وتاريخ ومتابعة الحالة.

4. الجلسات الجماعية:
مناسبة عندما يشعر المستخدم بالوحدة أو يريد دعمًا من أشخاص يمرون بتجارب مشابهة، ضمن جلسات نقاش جماعي بإشراف مختص.

5. AI Chat:
مناسب للدعم المبدئي، تنظيم الأفكار، واقتراح خطوات مؤقتة. لا يقدم تشخيصًا طبيًا ولا يغني عن المختص.

6. المقالات النفسية:
مناسبة عندما يسأل المستخدم عن معلومات حول القلق، التوتر، الحزن، النوم، المشاعر، أو يريد فهم حالته بشكل أوسع.

7. التمارين النفسية:
تشمل تمارين تنفس، تمارين تفاعلية، وتمارين فيديو. مناسبة للتوتر، القلق، التفكير الزائد، صعوبة النوم، أو الحاجة لتهدئة سريعة.

8. الدورات:
مناسبة عندما يريد المستخدم تعلم مهارات نفسية على فترة أطول، مثل التعامل مع الضغط، تنظيم المشاعر، أو تحسين الوعي النفسي.

9. الخطط العلاجية:
مناسبة عندما تكون المشكلة متكررة أو تحتاج متابعة منظمة من خلال مراحل وأهداف وأسئلة شائعة ومختصين مرتبطين بالخطة.

10. الإشعارات:
يمكن استخدامها لتذكير المستخدم بالجلسات، الردود الجديدة، أو متابعة المحتوى والخطط.

قواعد اقتراح ميزات التطبيق:
- لا تقترح كل الميزات دفعة واحدة.
- اقترح ميزة واحدة أو ميزتين فقط حسب الحالة.
- اجعل الاقتراح طبيعيًا داخل الرد وليس كإعلان.
- لا تقل إن التطبيق سيعالج المستخدم نهائيًا.
- إذا كانت الحالة طارئة أو فيها إيذاء للنفس، الأولوية للسلامة والتواصل مع شخص قريب أو طوارئ أو مختص.
"""

def save_message(user_id, sender, message_text):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO ai_chat_messages (user_id, sender, message_text)
        VALUES (?, ?, ?)
        """,
        user_id,
        sender,
        message_text
    )

    conn.commit()
    conn.close()

def get_recent_memory(user_id, limit=8):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT TOP (?) sender, message_text
        FROM ai_chat_messages
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        limit,
        user_id
    )

    rows = cursor.fetchall()
    conn.close()

    rows.reverse()

    memory_text = ""

    for row in rows:
        sender = row[0]
        text = row[1]

        if sender == "user":
            memory_text += f"المستخدم: {text}\n"
        else:
            memory_text += f"أنيس: {text}\n"

    return memory_text

def get_user_profile(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT name, main_issue, triggers, sleep_notes, last_summary
        FROM ai_user_profiles
        WHERE user_id = ?
        """,
        user_id
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return {
            "name": "",
            "main_issue": "",
            "triggers": "",
            "sleep_notes": "",
            "last_summary": ""
        }

    return {
        "name": row[0] or "",
        "main_issue": row[1] or "",
        "triggers": row[2] or "",
        "sleep_notes": row[3] or "",
        "last_summary": row[4] or ""
    }

def save_or_update_profile(user_id, profile):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        IF EXISTS (SELECT 1 FROM ai_user_profiles WHERE user_id = ?)
        BEGIN
            UPDATE ai_user_profiles
            SET
                name = COALESCE(NULLIF(?, ''), name),
                main_issue = COALESCE(NULLIF(?, ''), main_issue),
                triggers = COALESCE(NULLIF(?, ''), triggers),
                sleep_notes = COALESCE(NULLIF(?, ''), sleep_notes),
                last_summary = COALESCE(NULLIF(?, ''), last_summary),
                updated_at = GETDATE()
            WHERE user_id = ?
        END
        ELSE
        BEGIN
            INSERT INTO ai_user_profiles
            (user_id, name, main_issue, triggers, sleep_notes, last_summary)
            VALUES (?, ?, ?, ?, ?, ?)
        END
        """,
        user_id,
        profile.get("name", ""),
        profile.get("main_issue", ""),
        profile.get("triggers", ""),
        profile.get("sleep_notes", ""),
        profile.get("last_summary", ""),
        user_id,

        user_id,
        profile.get("name", ""),
        profile.get("main_issue", ""),
        profile.get("triggers", ""),
        profile.get("sleep_notes", ""),
        profile.get("last_summary", "")
    )

    conn.commit()
    conn.close()

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
    return {"status": "API is running"}

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
- اقترحها فقط عندما تكون مفيدة فعلًا لحالة المستخدم.
- لا تقترح أكثر من ميزتين في الرد الواحد.
- إذا المستخدم يحتاج معلومات، اقترح المقالات.
- إذا يحتاج تهدئة أو توتر أو نوم، اقترح التمارين النفسية أو Safe Space.
- إذا المشكلة متكررة أو مؤثرة على حياته اليومية، اقترح الخطط العلاجية أو الجلسة الفردية.
- إذا يشعر بالوحدة، يمكن اقتراح الجلسات الجماعية أو المجتمع داخل Safe Space.
- إذا يريد متابعة خاصة، يمكن اقتراح الشات الآمن مع الطبيب.
- اجعل الاقتراح طبيعيًا ومختصرًا.

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