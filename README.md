# AI-Auto-DC-Check

An automated Data Cleaning (DC) and refinement pipeline designed to reduce noise, fix anomalies, and format discrepancies in uploaded datasets before they hit Machine Learning training pipelines. 

By eliminating data noise early, **AI-Auto-DC-Check** significantly improves predictive model accuracy and reduces training errors.

---

## 🚀 Features

* **Universal File Detection & Conversion:** Automatically identifies uploaded file types and seamlessly handles multi-format conversions.
* **AI-Powered Data Engine:** Detects structural anomalies, missing values, and high-frequency noise within datasets.
* **Predictive Evaluation Engine:** Scores datasets to estimate the impact of cleaning on overall training accuracy.
* **Dual Interface Application:** * **Backend:** High-performance REST API powered by **FastAPI**.
    * **Frontend (AutoDC Universal AI Refiner):** An intuitive web UI built with **Streamlit**.

---

## 📂 Project Structure

```text
├── main.py          # FastAPI Application (Backend Server)
├── frontend.py      # Streamlit Web Interface (UI Client)
├── processor.py     # AIDataEngine (Preprocessing & Anomaly Detection)
├── converter.py     # UniversalConverter (Multi-format File Handler)
├── detector.py      # FileDetector (Format Verification)
├── evaluator.py     # EvaluationEngine (Predictive Accuracy Scoring)
└── README.md        # Project Documentation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required libraries
pip install fastapi uvicorn streamlit pandas numpy scikit-learn

Step 1: Start the FastAPI Backend
uvicorn main:app --reload --port 8000

Step 2: Start the Streamlit Frontend
streamlit run frontend.py
