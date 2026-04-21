import streamlit as st
import numpy as np
import cv2
from PIL import Image
import hashlib, json, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="DeepTrust AI", layout="wide")

# ─── UI ───
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #020617, #0f172a, #1e293b, #020617);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    margin: 10px 0;
    backdrop-filter: blur(12px);
}
.glow {
    font-size: 2rem;
    font-weight: 900;
    text-align: center;
    color: #60a5fa;
}
</style>
""", unsafe_allow_html=True)

# ─── AUTH ───
USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    return json.load(open(USER_FILE))

def save_users(u):
    json.dump(u, open(USER_FILE,"w"))

def hash_pass(p):
    return hashlib.sha256(p.encode()).hexdigest()

users = load_users()

if "logged" not in st.session_state:
    st.session_state.logged = False

if "history" not in st.session_state:
    st.session_state.history = []

def login():
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in users and users[u] == hash_pass(p):
            st.session_state.logged = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

def signup():
    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")
    if st.button("Signup"):
        users[u] = hash_pass(p)
        save_users(users)
        st.success("Account created")

if not st.session_state.logged:
    st.title("🛡️ DeepTrust AI")
    t1,t2 = st.tabs(["Login","Signup"])
    with t1: login()
    with t2: signup()
    st.stop()

# ─── SIDEBAR ───
with st.sidebar:
    st.write(f"👤 {st.session_state.user}")
    if st.button("Logout"):
        st.session_state.logged=False
        st.rerun()

mode = st.sidebar.radio("Mode", ["Upload","Compare","Video","Dashboard"])

st.title("🛡️ DeepTrust AI")

# ─── FEATURE EXTRACTION ───
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    texture = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = np.std(gray)
    edges = np.mean(cv2.Canny(gray,100,200))
    brightness = np.mean(gray)
    contrast = gray.std()

    return [texture, noise, edges, brightness, contrast]

# ─── GENERATE SYNTHETIC TRAINING DATA ───
def generate_data():
    real = []
    fake = []

    # Simulated real images (lower artifacts)
    for _ in range(200):
        real.append([
            np.random.uniform(80,140),
            np.random.uniform(10,30),
            np.random.uniform(20,50),
            np.random.uniform(100,140),
            np.random.uniform(30,50)
        ])

    # Simulated fake images (higher artifacts)
    for _ in range(200):
        fake.append([
            np.random.uniform(180,300),
            np.random.uniform(40,80),
            np.random.uniform(60,120),
            np.random.uniform(140,180),
            np.random.uniform(60,90)
        ])

    X = np.array(real + fake)
    y = np.array([0]*len(real) + [1]*len(fake))

    return X, y

# ─── TRAIN MODEL ───
@st.cache_resource
def train_model():
    X, y = generate_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model()

# ─── DETECTOR ───
class Detector:
    def analyze(self, img):
        features = extract_features(img)
        features_scaled = scaler.transform([features])

        prob = model.predict_proba(features_scaled)[0]

        real_prob = prob[0]
        fake_prob = prob[1]

        if fake_prob > real_prob:
            return int(fake_prob*100), "Fake 🚨"
        else:
            return int(real_prob*100), "Real ✅"

detector = Detector()

# ─── HEATMAP ───
def heatmap(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    heat = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    return cv2.addWeighted(img,0.6,heat,0.4,0)

# ─── IMAGE MODE ───
if mode=="Upload":
    file = st.file_uploader("Upload Image", type=["jpg","png"])

    if file:
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
        st.image(file)

        if st.button("Analyze"):
            s,v = detector.analyze(img)
            st.session_state.history.append(s)

            st.markdown(f"<div class='card'><div class='glow'>{v} ({s}%)</div></div>", unsafe_allow_html=True)
            st.image(heatmap(img))

# ─── COMPARE ───
elif mode=="Compare":
    c1,c2 = st.columns(2)
    f1 = c1.file_uploader("Image1", key="1")
    f2 = c2.file_uploader("Image2", key="2")

    if f1 and f2:
        img1 = cv2.cvtColor(np.array(Image.open(f1)), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(Image.open(f2)), cv2.COLOR_RGB2BGR)

        c1.image(f1)
        c2.image(f2)

        if st.button("Compare"):
            s1,v1 = detector.analyze(img1)
            s2,v2 = detector.analyze(img2)

            c1.markdown(f"<div class='card'>{v1} ({s1}%)</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='card'>{v2} ({s2}%)</div>", unsafe_allow_html=True)

# ─── VIDEO MODE ───
elif mode=="Video":
    video = st.file_uploader("Upload Video", type=["mp4"])

    if video:
        with open("temp.mp4","wb") as f:
            f.write(video.read())

        cap = cv2.VideoCapture("temp.mp4")
        scores=[]
        frames=0

        while cap.isOpened():
            ret,frame=cap.read()
            if not ret: break

            if frames%8==0:
                s,_=detector.analyze(frame)
                scores.append(s)

            frames+=1

        cap.release()

        if scores:
            avg=int(np.mean(scores))

            if avg < 50:
                result="Fake 🚨"
            else:
                result="Real ✅"

            st.markdown(f"<div class='card'><div class='glow'>{result} ({avg}%)</div></div>", unsafe_allow_html=True)
            st.line_chart(scores)

# ─── DASHBOARD ───
elif mode=="Dashboard":
    data = st.session_state.history

    if data:
        st.line_chart(data)

        real=sum(1 for x in data if x>=50)
        fake=sum(1 for x in data if x<50)

        st.bar_chart({"Real":real,"Fake":fake})
    else:
        st.info("No data yet")

st.markdown("---")
st.caption("🚀 DeepTrust AI | ML Powered Detection")
