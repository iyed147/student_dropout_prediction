import matplotlib
matplotlib.use("Agg")   # pas de fenêtre GUI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
)


class ModelEvaluator:
    """Évaluation complète des modèles."""

    def __init__(self):
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")
        os.makedirs("reports", exist_ok=True)

    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, y_true, y_pred, model_name=""):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Sécurisé","À risque"],
                    yticklabels=["Sécurisé","À risque"])
        plt.title(f"Matrice de Confusion — {model_name}")
        plt.ylabel("Vrai")
        plt.xlabel("Prédit")
        plt.tight_layout()
        plt.savefig(f"reports/confusion_matrix_{model_name}.png", dpi=300)
        plt.close()
        return cm

    # ------------------------------------------------------------------
    def plot_roc_curve(self, y_true, y_pred_proba, model_name=""):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC (AUC = {roc_auc:.3f})")
        plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Courbe ROC — {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"reports/roc_curve_{model_name}.png", dpi=300)
        plt.close()
        return roc_auc

    # ------------------------------------------------------------------
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name=""):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="green", lw=2,
                 label=f"PR (AUC = {pr_auc:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Courbe Precision-Recall — {model_name}")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"reports/pr_curve_{model_name}.png", dpi=300)
        plt.close()
        return pr_auc

    # ------------------------------------------------------------------
    def plot_feature_importance(self, model, feature_names, top_n=20):
        # Cherche feature_importances_ dans le pipeline ou directement
        clf = None
        if hasattr(model, "feature_importances_"):
            clf = model
        elif hasattr(model, "named_steps") and hasattr(
            model.named_steps.get("classifier"), "feature_importances_"
        ):
            clf = model.named_steps["classifier"]

        if clf is None:
            print("⚠️  Modèle sans feature_importances_ — graphique ignoré")
            return None

        importances = clf.feature_importances_

        # Alignement taille (pipeline peut réduire les features)
        n = min(len(feature_names), len(importances))
        fi = pd.DataFrame({
            "feature": list(feature_names)[:n],
            "importance": importances[:n],
        }).sort_values("importance", ascending=False).head(top_n)

        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(fi)), fi["importance"].values)
        plt.yticks(range(len(fi)), fi["feature"].values)
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} features importantes")
        plt.gca().invert_yaxis()
        for bar in bars:
            w = bar.get_width()
            plt.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
                     f"{w:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig("reports/feature_importance.png", dpi=300)
        plt.close()
        return fi

    # ------------------------------------------------------------------
    def comprehensive_evaluation(self, y_true, y_pred, y_pred_proba, model_name=""):
        print(f"\n📊 ÉVALUATION COMPLÈTE — {model_name}")
        print("=" * 50)

        print("\n📋 Rapport de classification:")
        print(classification_report(y_true, y_pred,
                                    target_names=["Sécurisé","À risque"]))

        cm = self.plot_confusion_matrix(y_true, y_pred, model_name)
        tn, fp, fn, tp = cm.ravel()

        print("🎯 Matrice de confusion:")
        print(f"  Vrais Positifs (TP) : {tp}")
        print(f"  Faux Négatifs  (FN) : {fn}  ← CRITIQUE")
        print(f"  Faux Positifs  (FP) : {fp}")
        print(f"  Vrais Négatifs (TN) : {tn}")

        sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr_rate     = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"\n📈 Métriques:")
        print(f"  Sensitivité (Recall) : {sensitivity:.3f}")
        print(f"  Spécificité          : {specificity:.3f}")
        print(f"  Taux Faux Positifs   : {fpr_rate:.3f}")

        roc_auc = self.plot_roc_curve(y_true, y_pred_proba, model_name)
        pr_auc  = self.plot_precision_recall_curve(y_true, y_pred_proba, model_name)

        print(f"\n📊 Courbes:")
        print(f"  AUC ROC              : {roc_auc:.3f}")
        print(f"  AUC Precision-Recall : {pr_auc:.3f}")

        return {
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }