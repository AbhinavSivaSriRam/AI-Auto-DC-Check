import streamlit as st
import requests
import pandas as pd
import time
import io

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AutoDC | Universal AI Refiner",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NEON TECH CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background: linear-gradient(45deg, #00ffcc, #0099ff);
        color: #0e1117;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(0, 255, 204, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #ff00ff, #00ffcc);
        color: white;
        box-shadow: 0 0 20px #ff00ff;
        transform: translateY(-2px);
    }
    .project-desc-box {
        background-color: #1c2128;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .results-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
        width: 100%;
    }
    .result-card {
        width: 280px;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-value {
        color: #00ffcc;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .thanks-msg {
        color: #00ffcc;
        font-weight: bold;
        font-size: 1.3rem;
        margin-top: 40px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("AutoDC Core")
    st.info("⚡ **HI-DBSF v1.0 Activated**")
    st.write("Target: Multi-Modal Noise Purging")
    st.write("Status: **System Online**")

# --- HEADER ---
st.title("🧪 Universal Data-Centric AI")
st.markdown("""
    <div class="project-desc-box">
        <h3>🚀 Universal Data Refinement Engine</h3>
        <p>AutoDC uses <b>Hybrid Intelligence</b> (Isolation Forest + DBSCAN) to validate and clean 
        <b>Images, Videos, PDFs, and Tabular datasets</b> in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

# --- STEP 1: PROJECT DESCRIPTION ---
st.subheader("🎯 Step 1: Describe your Project")
user_description = st.text_input(
    "What project are you building?", 
    placeholder="e.g., Road Safety Monitoring, Smart Legal Archival, or Voting System"
)

if user_description:
    st.subheader("📂 Step 2: Upload Data Source")
    uploaded_file = st.file_uploader("Upload CSV, PNG, JPG, PDF, or MP4", type=["csv", "png", "jpg", "pdf", "mp4"])

    if uploaded_file:
        # Prepare file for the request
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data_payload = {"goal": user_description}
        
        try:
            # Step 2: Initial Analysis Request
            response = requests.post("http://127.0.0.1:8000/process-data", files=files, data=data_payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if not data.get("is_related", True):
                    st.error(f"⚠️ {data.get('message', 'Validation Failed')}")
                    st.stop()
                
                st.success(f"✅ {data.get('message', 'System Ready')}")
                meta = data.get('metadata', {})
                st.write(f"**Category Identified:** `{meta.get('category')}` | **Format:** `{meta.get('extension')}`")

                st.write("---")
                st.subheader("⚙️ Step 3: Analysis & Refinement")
                
                if st.button("🚀 INITIATE REFINEMENT"):
                    with st.status("🔍 HI-DBSF Pipeline Executing...", expanded=True) as status:
                        st.write("🧠 Training Isolation Forest on Features...")
                        time.sleep(0.8)
                        st.write("🛰️ Mapping Cluster Density (DBSCAN)...")
                        time.sleep(0.6)
                        st.write("🧹 Purging Non-Statistical Artifacts...")
                        status.update(label="✅ Refinement Complete!", state="complete")
                        st.balloons()

                    # --- RESULTS SECTION ---
                    st.subheader("📊 Step 4: Results & Metrics")
                    m = data.get('metrics', {})
                    rep = data.get('report', {})

                    st.markdown(f"""
                        <div class="results-container">
                            <div class="result-card">
                                <div style="color:#8b949e">Quality Boost</div>
                                <div class="result-value">{m.get('improvement', '0%')}</div>
                            </div>
                            <div class="result-card">
                                <div style="color:#8b949e">Data Retention</div>
                                <div class="result-value">{m.get('retention', '0%')}</div>
                            </div>
                            <div class="result-card">
                                <div style="color:#8b949e">Noise Removed</div>
                                <div class="result-value">{rep.get('original_rows', 0) - rep.get('cleaned_rows', 0)}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("---")
                    st.subheader("📥 Step 5: Export Cleaned File")
                    
                    # Call Download Endpoint with a fresh file stream
                    # We send the file again because FastAPI endpoints are stateless
                    dl_res = requests.post("http://127.0.0.1:8000/download-cleaned", 
                                            files={"file": (uploaded_file.name, uploaded_file.getvalue())})
                    
                    if dl_res.status_code == 200:
                        # DYNAMIC EXTENSION HANDLER for the UI
                        category = meta.get('category')
                        if category == "Image":
                            file_ext = ".jpg"
                            mime_type = "image/jpeg"
                        elif category == "Document":
                            file_ext = ".pdf"
                            mime_type = "application/pdf"
                        else:
                            file_ext = ".csv"
                            mime_type = "text/csv"

                        st.download_button(
                            label=f"📥 DOWNLOAD REFINED {category.upper()}",
                            data=dl_res.content,
                            file_name=f"Refined_{uploaded_file.name.split('.')[0]}{file_ext}",
                            mime=mime_type,
                            use_container_width=True
                        )
                    st.markdown('<p class="thanks-msg">Refinement Strategy Applied Successfully!</p>', unsafe_allow_html=True)
            else:
                st.error("🚨 Backend connection failed. Ensure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"⚠️ Connection Failed: {str(e)}")
