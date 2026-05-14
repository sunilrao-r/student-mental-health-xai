# Explainable AI–Based Analysis of Factors Influencing Student Mental Health - Mini Project

**Team.no:** 
**Team:** Sunil Rao R | Kamalesh V  
**Reg.no:** 3592310051 | 3592310019
**Objective:** Predict depression risk using student data with Explainable AI (SHAP)
**Keywords:** Explainable AI, Student Mental Health, Data Analysis, SHAP, Interpretable Machine Learning

### How to run
1. `pip install -r requirements.txt`
2. `python main.py`
3. `streamlit run src/app.py`

### Files
- `src/preprocess.py` → data cleaning
- `src/model.py` → Random Forest + evaluation
- `src/xai.py` → SHAP explanations
- `src/app.py` → Interactive web app
- `main.py` → runs everything

Plots are saved in `plots/` folder.