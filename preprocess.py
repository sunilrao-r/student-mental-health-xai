import numpy as np
import pandas as pd

SLEEP_MAP = {"Less than 5 hours": 1, "5-6 hours": 2, "7-8 hours": 3, "More than 8 hours": 4, "Others": 2}
DIET_MAP = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2, "Others": 1}
YES_NO = {"Yes": 1, "No": 0}
GENDER_MAP = {"Male": 0, "Female": 1}

def load_data(filepath="data/Student_Depression_Dataset.csv"):
    df = pd.read_csv(filepath)
    df = df[df["Profession"] == "Student"].copy().reset_index(drop=True)
    df = df.drop(columns=["id", "City", "Profession", "Work Pressure", "Job Satisfaction"])
    
    def degree_level(d):
        d = str(d).strip()
        if d == "Class 12": return 0
        if d.startswith(("B.", "BA", "BSc", "BCA", "BBA", "BHM", "BFA", "BDS", "BPT")): return 1
        if d.startswith(("M.", "MA", "MSc", "MCA", "MBA", "MHM", "PhD")): return 2
        return 1
    df["degree_level"] = df["Degree"].apply(degree_level)
    df = df.drop(columns=["Degree"])
    
    df["sleep_encoded"] = df["Sleep Duration"].map(SLEEP_MAP)
    df["diet_encoded"] = df["Dietary Habits"].map(DIET_MAP)
    df["suicidal_encoded"] = df["Have you ever had suicidal thoughts ?"].map(YES_NO)
    df["family_history"] = df["Family History of Mental Illness"].map(YES_NO)
    df["gender_encoded"] = df["Gender"].map(GENDER_MAP)
    
    df = df.drop(columns=["Sleep Duration", "Dietary Habits",
                          "Have you ever had suicidal thoughts ?",
                          "Family History of Mental Illness", "Gender"])
    
    df["Financial Stress"] = df["Financial Stress"].fillna(df["Financial Stress"].median())
    
    df = df.rename(columns={
        "Age": "age", "Academic Pressure": "academic_pressure", "CGPA": "cgpa",
        "Study Satisfaction": "study_satisfaction", "Work/Study Hours": "study_hours_per_day",
        "Financial Stress": "financial_stress", "Depression": "depression"
    })
    
    feature_cols = ["academic_pressure", "financial_stress", "study_satisfaction",
                    "sleep_encoded", "suicidal_encoded", "family_history",
                    "study_hours_per_day", "cgpa", "age", "diet_encoded",
                    "degree_level", "gender_encoded"]
    
    feature_labels = ["Academic Pressure", "Financial Stress", "Study Satisfaction",
                      "Sleep Duration", "Suicidal Thoughts", "Family History",
                      "Study Hours/Day", "CGPA", "Age", "Dietary Habits",
                      "Degree Level", "Gender"]
    
    df_clean = df[feature_cols + ["depression"]].dropna().reset_index(drop=True)
    
    X = df_clean[feature_cols].values.astype(np.float64)
    y = df_clean["depression"].values.astype(int)
    
    print(f"✅ Data ready: {len(df_clean)} students, {X.shape[1]} features")
    return X, y, feature_labels, df_clean
