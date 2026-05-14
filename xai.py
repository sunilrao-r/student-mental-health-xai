import shap
import matplotlib.pyplot as plt
import numpy as np

def explain_model(model, X_test, feature_labels):
    # Background sample for SHAP
    background = X_test[np.random.choice(len(X_test), 300, replace=False)]
    
    explainer = shap.TreeExplainer(model, background)
    
    # IMPORTANT: check_additivity=False fixes the common error with class_weight="balanced"
    shap_values = explainer(X_test[:500], check_additivity=False)
    
    # Global Beeswarm Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values.values[:, :, 1], 
                      X_test[:500], 
                      feature_names=feature_labels, 
                      show=False)
    plt.title("SHAP Global Feature Importance (Beeswarm)")
    plt.tight_layout()
    plt.savefig("plots/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Waterfall Plot for one high-risk student
    dep_idx = np.argmax(shap_values.values[:, :, 1].sum(axis=1))
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(
        values=shap_values.values[dep_idx, :, 1],
        base_values=shap_values.base_values[dep_idx, 1],
        data=X_test[dep_idx],
        feature_names=feature_labels
    ), max_display=12, show=False)
    plt.title("SHAP Explanation - One Depressed Student")
    plt.tight_layout()
    plt.savefig("plots/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("✅ SHAP plots saved successfully!")
    return shap_values