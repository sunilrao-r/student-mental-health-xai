import os
os.makedirs("plots", exist_ok=True)

print("=== Student Depression Mini Project ===")
from preprocess import load_data
from model import train_and_evaluate
from xai import explain_model

X, y, labels, _ = load_data()
model, X_test, _, _, _ = train_and_evaluate(X, y)
explain_model(model, X_test, labels)

print("\n✅ All done!")
print("   → Run the app: streamlit run app.py")