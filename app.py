import streamlit as st
import cv2
from deepface import DeepFace
from transformers import pipeline
import numpy as np
from PIL import Image
from gtts import gTTS
import tempfile
import os
from groq import Groq
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

st.set_page_config(
    page_title="Mental Health AI Companion",
    page_icon="🧠",
    layout="wide"
)

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

st.markdown("""
<style>
    .stApp { background: #f0f4ff; }
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 900;
        color: #3a3a8c;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-title {
        text-align: center;
        color: #6b6b9a;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .section-head {
        background: linear-gradient(90deg, #3a3a8c, #6c63ff);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
    }
    .emotion-card {
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    .tip-item {
        background: white;
        border-left: 5px solid #6c63ff;
        border-radius: 0 12px 12px 0;
        padding: 12px 18px;
        margin: 8px 0;
        font-size: 1rem;
        color: #2d2d5e;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .quote-box {
        background: white;
        border-radius: 15px;
        padding: 18px 24px;
        margin: 1rem 0;
        border-left: 6px solid #6c63ff;
        font-style: italic;
        color: #3a3a8c;
        font-size: 1.05rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .chat-user {
        background: linear-gradient(90deg, #6c63ff, #3a3a8c);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-ai {
        background: white;
        color: #2d2d5e;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        border-left: 4px solid #6c63ff;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stTextArea textarea {
        background: white !important;
        color: #1a1a3e !important;
        border: 2px solid #6c63ff !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
    }
    .stTextInput input {
        background: white !important;
        color: #1a1a3e !important;
        border: 2px solid #6c63ff !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3a3a8c, #6c63ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 35px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #6c63ff;
    }
    section[data-testid="stSidebar"] {
        background: #2d2d5e;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0ff !important;
    }
    .voice-box {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 2px dashed #6c63ff;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #9090b0;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #dde0ff;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

emotion_config = {
    "joy": {
        "emoji": "😊",
        "bg": "linear-gradient(135deg, #ffecd2, #fcb69f)",
        "color": "#7c3a00",
        "wellness": 90,
        "quote": "Happiness is not something ready-made. It comes from your own actions. — Dalai Lama",
        "tips": [
            "Write down 3 things you are grateful for today",
            "Share your happiness with a friend or family member",
            "Use this positive energy to work on a creative project",
            "Celebrate your wins no matter how small they are",
            "Go for a walk and enjoy this beautiful moment fully"
        ],
        "audio": "You are feeling joyful today! That is wonderful. Write down three things you are grateful for. Share your happiness with someone you love. Keep smiling!"
    },
    "sadness": {
        "emoji": "😢",
        "bg": "linear-gradient(135deg, #e0f7fa, #b2ebf2)",
        "color": "#00607a",
        "wellness": 35,
        "quote": "Even the darkest night will end and the sun will rise. — Victor Hugo",
        "tips": [
            "Talk to someone you trust about how you feel",
            "Try a 5 minute deep breathing exercise right now",
            "Watch something funny or uplifting on YouTube",
            "Go outside for fresh air even for just 10 minutes",
            "Write your feelings in a journal to release them",
            "Remember — sadness is temporary and will always pass"
        ],
        "audio": "You are feeling sad right now, and that is completely okay. Talk to someone you trust. Take five deep breaths slowly. Remember, this feeling is temporary and you will feel better soon!"
    },
    "anger": {
        "emoji": "😠",
        "bg": "linear-gradient(135deg, #fce4ec, #f8bbd0)",
        "color": "#7c0020",
        "wellness": 30,
        "quote": "For every minute you are angry, you lose sixty seconds of happiness.",
        "tips": [
            "Take 10 slow deep breaths before doing anything",
            "Go for a brisk walk to release that built up energy",
            "Write down what made you angry in a journal",
            "Drink a glass of cold water slowly and mindfully",
            "Count from 1 to 20 very slowly in your mind",
            "Listen to calming music for at least 10 minutes"
        ],
        "audio": "You are feeling angry right now. Take ten slow deep breaths. Then go for a short walk to release that energy. Responding calmly always gives better results!"
    },
    "fear": {
        "emoji": "😨",
        "bg": "linear-gradient(135deg, #e8f5e9, #c8e6c9)",
        "color": "#1b5e20",
        "wellness": 25,
        "quote": "You gain strength and confidence by every experience in which you face your fear.",
        "tips": [
            "Name 5 things you can SEE around you right now",
            "Name 4 things you can TOUCH around you",
            "Name 3 things you can HEAR right now",
            "Take slow deep breaths for 2 full minutes",
            "Remind yourself — you are safe right now",
            "Talk to someone you trust about your fear",
            "Break your problem into very small manageable steps"
        ],
        "audio": "You are feeling fearful right now. Name five things you can see. Take a slow deep breath in and breathe out slowly. You are safe right now!"
    },
    "surprise": {
        "emoji": "😲",
        "bg": "linear-gradient(135deg, #fff9c4, #fff176)",
        "color": "#5c4a00",
        "wellness": 70,
        "quote": "Life is full of surprises. Being open to unexpected turns is key.",
        "tips": [
            "Take a moment to pause and process what happened",
            "Write down your thoughts and feelings about it",
            "Talk to someone about what surprised you",
            "Channel this fresh energy into something productive",
            "Stay open minded — surprises often bring new opportunities"
        ],
        "audio": "You are feeling surprised! Take a moment to pause and process. Surprises can lead to amazing new opportunities!"
    },
    "neutral": {
        "emoji": "😐",
        "bg": "linear-gradient(135deg, #ede7f6, #d1c4e9)",
        "color": "#311b6e",
        "wellness": 65,
        "quote": "A calm mind is a powerful mind. Use this peace to create something great.",
        "tips": [
            "This is a great time to focus and study productively",
            "Try meditating for 10 minutes to go even deeper",
            "Plan your goals clearly for the rest of the week",
            "Learn something new and interesting today",
            "Reach out to a friend you have not spoken to recently"
        ],
        "audio": "You are feeling calm and neutral today. Use this calm energy to focus on your goals. A calm mind is a powerful mind!"
    },
    "disgust": {
        "emoji": "🤢",
        "bg": "linear-gradient(135deg, #f1f8e9, #dcedc8)",
        "color": "#1b5e00",
        "wellness": 30,
        "quote": "You cannot always control what happens, but you can always control how you respond.",
        "tips": [
            "Remove yourself from whatever is causing discomfort",
            "Take deep breaths to reset your mind and body",
            "Do something you genuinely enjoy to lift your mood",
            "Talk to someone you trust about what is bothering you",
            "Remember — your feelings and boundaries are always valid"
        ],
        "audio": "You are feeling disgusted or uncomfortable. Remove yourself from the situation. Take some deep breaths and do something you enjoy!"
    }
}

def generate_audio(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(tmp.name)
    return tmp.name

def transcribe_audio(audio_bytes, language="English"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    try:
        with open(tmp.name, 'rb') as f:
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", f),
                model="whisper-large-v3",
                language="hi" if language == "Hindi" else "en"
            )
        return transcription.text
    except Exception as e:
        st.error(f"Voice error: {e}")
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass

def get_ai_response(messages, detected_emotion=None, language="English"):
    if language == "Hindi":
        system_prompt = f"""Tu mera sabse accha dost hai — Lily. Tu ek real insaan ki tarah baat karta hai, robot ki tarah bilkul nahi.
Tu hamesha Hindi mein baat karta hai — warm, caring, funny aur motivating tarike se.
{'User ki abhi ki feeling hai: ' + detected_emotion if detected_emotion else ''}
Tu unhe sunta hai, samajhta hai, motivate karta hai aur guide karta hai jaise ek close friend karta hai.
Kabhi bhi boring ya formal mat bana. Chhoti chhoti baatein karo, thoda mazak karo, dil se baat karo.
Agar koi problem hai toh solve karne mein help karo. Hamesha positive aur caring raho.
Response 3-4 lines mein rakh — zyada lamba mat karo."""
    else:
        system_prompt = f"""You are Lily, the user's closest best friend — not an AI, not a robot, a real warm human friend.
You always talk in a warm, caring, funny and motivating way just like a best friend does.
{'The user is currently feeling: ' + detected_emotion if detected_emotion else ''}
You listen deeply, understand their feelings, motivate and guide them just like a best friend would.
Never be boring or formal. Keep it real, add a little humor, speak from the heart.
If they have a problem help them solve it. Always be positive, caring and genuine.
Keep responses to 3-4 lines — not too long, like a real chat."""

    all_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=all_messages,
        max_tokens=300,
        temperature=0.9
    )
    return response.choices[0].message.content

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

classifier = load_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = None
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

with st.sidebar:
    st.markdown("## 🧠 Mental Health AI")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
**📝 Text Analysis**
Type how you feel and our NLP model reads your emotions.

**📷 Face Scan**
Take a photo and DeepFace AI reads your expression.

**🔊 Audio Guidance**
Get a personalised voice message with tips.

**🎤 Voice Chat with Lily**
Talk to Lily using your voice in Hindi or English!
He listens, understands and speaks back to you!
    """)
    st.markdown("---")
    st.markdown("### 🎯 Emotions Detected")
    for e in ["😊 Joy", "😢 Sadness", "😠 Anger",
              "😨 Fear", "😲 Surprise", "😐 Neutral", "🤢 Disgust"]:
        st.markdown(f"- {e}")
    st.markdown("---")
    st.markdown("*Built with Python · DeepFace · HuggingFace · Groq AI*")

st.markdown("<div class='main-title'>🧠 Mental Health AI Companion</div>",
            unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Emotion detection + AI best friend Lily — voice and text in Hindi and English</div>",
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🧠 Emotion Detection", "🤖 Chat with Lily"])

with tab1:
    st.markdown("<div class='section-head'>📝 Step 1 — How are you feeling? Type it below</div>",
                unsafe_allow_html=True)
    user_text = st.text_area("",
                             placeholder="e.g. I feel very anxious and stressed about my exams...",
                             height=120)

    text_emotion = None
    text_score = 0
    if user_text:
        try:
            results = classifier(user_text)
            if isinstance(results[0], list):
                results = results[0]
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            text_emotion = results[0]['label']
            text_score = round(results[0]['score'] * 100, 1)
            c1, c2, c3 = st.columns(3)
            c1.metric("📝 Text Emotion", text_emotion.upper())
            c2.metric("🎯 Confidence", f"{text_score}%")
            c3.metric("🔍 Method", "NLP Transformer")
            st.session_state.detected_emotion = text_emotion
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("<div class='section-head'>📷 Step 2 — Take a face snapshot</div>",
                unsafe_allow_html=True)
    img_file = st.camera_input("")
    face_emotion = None
    face_score = 0
    if img_file:
        img = Image.open(img_file)
        arr = np.array(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        try:
            res = DeepFace.analyze(bgr, actions=['emotion'],
                                   enforce_detection=False)
            face_emotion = res[0]['dominant_emotion']
            face_score = round(res[0]['emotion'][face_emotion], 1)
            c4, c5, c6 = st.columns(3)
            c4.metric("📷 Face Emotion", face_emotion.upper())
            c5.metric("🎯 Confidence", f"{face_score}%")
            c6.metric("🔍 Method", "DeepFace AI")
            st.session_state.detected_emotion = face_emotion
        except Exception as e:
            st.warning(f"Try better lighting: {e}")

    if text_emotion or face_emotion:
        final = text_emotion or face_emotion
        cfg = emotion_config.get(final.lower(), emotion_config["neutral"])

        st.markdown("<div class='section-head'>✨ Step 3 — Your Emotion Result</div>",
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class='emotion-card' style='background:{cfg["bg"]}'>
            <div style='font-size:4.5rem'>{cfg["emoji"]}</div>
            <div style='font-size:2.2rem;font-weight:900;color:{cfg["color"]};margin-top:10px'>
                {final.upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if text_emotion and face_emotion:
            if text_emotion.lower() == face_emotion.lower():
                st.success(
                    f"✅ Both face and text agree — you are feeling **{final.upper()}**")
            else:
                st.info(
                    f"🔀 Mixed — Face: **{face_emotion.upper()}** | Text: **{text_emotion.upper()}**")

        st.markdown(f"<div class='quote-box'>💬 {cfg['quote']}</div>",
                    unsafe_allow_html=True)

        wellness = cfg["wellness"]
        if text_score > 0:
            wellness = min(100, int((wellness + text_score) / 2))

        st.markdown("<div class='section-head'>💚 Your Wellness Score</div>",
                    unsafe_allow_html=True)
        cw1, cw2 = st.columns([4, 1])
        with cw1:
            st.progress(wellness / 100)
        with cw2:
            st.markdown(
                f"<div style='font-size:1.4rem;font-weight:700;color:#3a3a8c;text-align:center'>{wellness}/100</div>",
                unsafe_allow_html=True)

        st.markdown("<div class='section-head'>💡 What you can do right now</div>",
                    unsafe_allow_html=True)
        for i, tip in enumerate(cfg["tips"], 1):
            st.markdown(
                f"<div class='tip-item'>{'⭐' if i == 1 else f'{i}.'} &nbsp; {tip}</div>",
                unsafe_allow_html=True)

        st.markdown("<div class='section-head'>🔊 Personalised Audio Guidance</div>",
                    unsafe_allow_html=True)
        if st.button("▶ Play Audio Suggestion"):
            with st.spinner("Generating your personalised audio..."):
                path = generate_audio(cfg["audio"])
                with open(path, "rb") as f:
                    audio_bytes_tip = f.read()
                st.audio(audio_bytes_tip, format="audio/mp3")
                os.unlink(path)

with tab2:
    st.markdown("""
    <div style='background:white;border-radius:15px;padding:15px;margin-bottom:15px;
    border-left:5px solid #6c63ff;color:#2d2d5e;font-size:0.95rem'>
    👋 Hey! I am <b>Lily</b> — your AI best friend. I am here for you 24/7!<br><br>
    🎤 Press the <b>mic button</b> and speak to me directly!<br>
    ⌨️ Or just <b>type below</b> if you prefer!<br>
    🔊 I will <b>speak back</b> to you automatically!<br>
    🌐 I understand both <b>English and Hindi!</b>
    </div>
    """, unsafe_allow_html=True)

    lang_choice = st.radio(
        "Choose language:",
        ["English", "Hindi"],
        horizontal=True
    )

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-user'>🧑 You: {msg['content']}</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='chat-ai'>🤖 <b>Lily:</b> {msg['content']}</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='section-head'>🎤 Speak to Lily</div>",
                unsafe_allow_html=True)

    if lang_choice == "Hindi":
        st.markdown("**Hindi mein bolne ke liye mic dabao — Lily sun raha hai! 👂**")
    else:
        st.markdown("**Press mic and speak — Lily is listening! 👂**")

    audio_data = mic_recorder(
        start_prompt="🎤 Click to speak",
        stop_prompt="⏹ Click to stop",
        just_once=True,
        use_container_width=True,
        key="Lily_mic"
    )

    if audio_data and audio_data.get('id') != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_data.get('id')
        with st.spinner("Lily is listening..."):
            spoken_text = transcribe_audio(audio_data['bytes'],
                                           language=lang_choice)
        if spoken_text:
            st.success(f"🎤 You said: **{spoken_text}**")
            st.session_state.chat_history.append({
                "role": "user",
                "content": spoken_text
            })
            with st.spinner("Lily is thinking..."):
                ai_reply = get_ai_response(
                    st.session_state.chat_history,
                    detected_emotion=st.session_state.detected_emotion,
                    language=lang_choice
                )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_reply
            })
            st.markdown(
                f"<div class='chat-ai'>🤖 <b>Lily:</b> {ai_reply}</div>",
                unsafe_allow_html=True)
            lang_code = 'hi' if lang_choice == "Hindi" else 'en'
            with st.spinner("Lily is speaking..."):
                path = generate_audio(ai_reply, lang=lang_code)
                with open(path, "rb") as f:
                    Lily_audio = f.read()
                st.audio(Lily_audio, format="audio/mp3")
                os.unlink(path)
            st.success("✅ Lily has responded! Press mic again to continue talking.")
        else:
            if lang_choice == "Hindi":
                st.warning("Kuch samajh nahi aaya! Dobara bolne ki koshish karo.")
            else:
                st.warning("Could not understand! Please speak clearly and try again.")

    st.markdown("<div class='section-head'>⌨️ Or Type to Lily</div>",
                unsafe_allow_html=True)

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        placeholder = "Lily se kuch bhi poocho..." if lang_choice == "Hindi" else "Type anything to Lily..."
        user_msg = st.text_input("", placeholder=placeholder, key="chat_input")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        send = st.button("Send 💬")

    if send and user_msg.strip():
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_msg
        })
        with st.spinner("Lily is thinking..."):
            ai_reply = get_ai_response(
                st.session_state.chat_history,
                detected_emotion=st.session_state.detected_emotion,
                language=lang_choice
            )
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_reply
        })
        lang_code = 'hi' if lang_choice == "Hindi" else 'en'
        with st.spinner("Lily is speaking..."):
            path = generate_audio(ai_reply, lang=lang_code)
            with open(path, "rb") as f:
                Lily_audio = f.read()
            st.audio(Lily_audio, format="audio/mp3")
            os.unlink(path)
        st.success("✅ Lily has responded! Press mic again to continue talking.")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.session_state.last_audio_id = None
            st.success("✅ Lily has responded! Press mic again to continue talking.")

st.markdown(
    "<div class='footer'>🧠 Mental Health AI Companion | Built with Python · Streamlit · DeepFace · HuggingFace · Groq AI</div>",
    unsafe_allow_html=True)