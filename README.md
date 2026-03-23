Ship Power & Weight Prediction

Project Overview

This project predicts the Installed Power (kW) and Steel Weight (tons) of ships based on their design parameters. It leverages pre-trained ML pipelines for fast, accurate, and interactive predictions, suitable for ship designers, naval engineers, and estimators.

Key Features:

1.Single ship interactive prediction via Streamlit
2.Batch predictions via CSV upload
3.Derived ratios (L/B, B/D, Speed-Length) for quick reference
4.SHAP-based feature contribution analysis
5.Engineer validation mode for manual cross-checks
6.Assumptions & Limitations panel for transparency

Folder Structure:
SHIP_PW/
│
├─ saved_models/          # Pre-trained model pipelines & feature columns
│   ├─ power_pipeline.pkl
│   ├─ weight_pipeline.pkl
│   └─ feature_columns.pkl
│
├─ app.py                 # Main Streamlit app
├─ model_rebuild.ipynb    # Reference notebook for rebuilding models
├─ requirements.txt       # Python dependencies
└─ .gitignore             # Folders/files to exclude from git

Usage:
1. Setup Environment
# Clone the repo
git clone <repo_url>
cd SHIP_PW

# Create virtual environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
2. Run Streamlit App
streamlit run app.py

Single Ship: Enter ship parameters and select ship type for interactive prediction.

Batch Mode: Upload CSV files with ship parameters for bulk predictions.

Assumptions & Limitations:
1.Predictions assume standard steel hull construction.
2.Accuracy is highest for ships within the training range (LOA: 25–400 m, Speed: 10–25 kts).
3.Environmental factors (sea state, wind, currents) are not included.
4.Specialized vessels (submarines, yachts) may have reduced accuracy.
5.Default values are applied for secondary parameters if missing.

Dependencies:
1.Python 3.10+
2.pandas, joblib, shap, matplotlib, streamlit
3.Install via: pip install -r requirements.txt

Optional Features:
1.Engineer validation mode to compare AI predictions with manual estimates.
2.SHAP visualizations to analyze feature contributions.
3.Feedback panel to log UX insights for continuous improvement.

License:
MIT License — feel free to reuse, modify, or integrate into your workflow.
