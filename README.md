🚢 Ship Power & Weight Prediction

A machine learning web app that predicts Installed Power (kW) and Steel Weight (tonnes) of ships based on design parameters — built for naval engineers, ship designers, and estimators.

🔴 Live Demo: https://ship-power-weight-prediction-pramodakibhad.streamlit.app/

📌 Project Overview
This project uses pre-trained Random Forest pipelines to predict ship power and weight from basic hull dimensions and speed. It is designed to assist early-stage ship design where quick, data-driven estimates are valuable.

✨ Key Features

🔢 Single ship prediction — interactive input form with ship type presets
📂 Batch prediction — upload any CSV (column names are auto-detected)
📐 Derived ratios — L/B, B/D, Speed-Length ratio computed automatically
🔍 SHAP analysis — feature contribution visualization for explainability
✅ Sanity checks — flags unusual inputs before prediction
🔒 Engineer validation mode — manual cross-check panel


🧠 Model Details
TargetAlgorithmNotesInstalled Power (kW)Random Forest RegressorTrained on multi-type vessel datasetSteel Weight (tonnes)Random Forest RegressorTrained on multi-type vessel dataset
Input features: LOA, Breadth, Depth, Draft, Service Speed, Year Built, Ship Type
Training range:

LOA: 25 – 400 m
Speed: 10 – 25 knots
Ship types: OSV, Tug, Bulk Carrier, Container


📁 Folder Structure
SHIP_PW/
│
├─ saved_models/
│   ├─ power_pipeline.pkl
│   └─ weight_pipeline.pkl
│
├─ app.py                  # Main Streamlit app
├─ model_rebuild.ipynb     # Notebook for rebuilding models
├─ requirements.txt        # Python dependencies
└─ .gitignore

🚀 Usage
1. Clone & Setup
bashgit clone <repo_url>
cd SHIP_PW

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
2. Run the App
bashstreamlit run app.py
3. Batch CSV Upload
Upload any CSV with ship parameters — the app auto-detects column names like:
length, Length_m, LOA, loa_m, beam, Beam_m, draft, Draft_m etc.

⚠️ Assumptions & Limitations

Predictions assume standard steel hull construction
Accuracy is highest within the training range (LOA: 25–400 m, Speed: 10–25 kts)
Environmental factors (sea state, wind, currents) are not included
Specialized vessels (submarines, yachts) may have reduced accuracy
Missing secondary parameters are filled with sensible defaults


🛠️ Dependencies
streamlit==1.55.0
pandas==2.3.3
numpy==1.26.4
scikit-learn==1.3.2
joblib==1.5.3
matplotlib==3.10.8
shap==0.44.0

📄 License
MIT License — free to reuse, modify, or integrate into your workflow.
