import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10,
        min_samples_leaf=5, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    acc = model.score(X_test, y_test)
    
    print(f"Model Accuracy: {acc:.3f} | ROC-AUC: {auc:.4f}")
    
    # Confusion Matrix
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Depression", "Depressed"]).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(7,5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("plots/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return model, X_test, y_test, y_prob, auc