import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend non-interactif (pas de fenêtre GUI)
import matplotlib.pyplot as plt
import os


class TargetCreator:
    """Crée la variable cible 'dropout_risk'."""

    def create_dropout_target(self, data):
        print("\n🎯 ÉTAPE : Création de la variable cible 'dropout_risk'")
        print("=" * 50)

        required = ["G1","G2","absences","failures","studytime",
                    "famsup","famrel","health","Dalc","Walc"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        # Scores de risque pondérés
        low_grades          = ((data["G1"] < 10) & (data["G2"] < 10)).astype(int) * 3
        high_absences       = (data["absences"] > 15).astype(int) * 2
        has_failures        = (data["failures"] > 0).astype(int) * 2
        low_studytime       = (data["studytime"] < 2).astype(int) * 1
        low_family_support  = ((data["famsup"] == "no") & (data["famrel"] < 3)).astype(int) * 1
        health_issues       = (data["health"] < 3).astype(int) * 0.5
        high_alcohol        = ((data["Dalc"] >= 3) | (data["Walc"] >= 4)).astype(int) * 0.5

        risk_score = (
            low_grades + high_absences + has_failures
            + low_studytime + low_family_support
            + health_issues + high_alcohol
        )

        print(f"  Min    : {risk_score.min()}")
        print(f"  Max    : {risk_score.max()}")
        print(f"  Moyenne: {risk_score.mean():.2f}")
        print(f"  Médiane: {risk_score.median():.2f}")

        self._plot_risk_distribution(risk_score)

        threshold = np.percentile(risk_score, 75)
        print(f"\n🔍 Seuil (75ème percentile) : {threshold:.2f}")

        data = data.copy()
        data["dropout_risk"] = (risk_score >= threshold).astype(int)

        dist = data["dropout_risk"].value_counts()
        total = len(data)
        print(f"\n📊 Distribution de la variable cible:")
        print(f"  À risque (1)  : {dist.get(1,0)}  ({dist.get(1,0)/total*100:.1f}%)")
        print(f"  Sécurisé (0)  : {dist.get(0,0)}  ({dist.get(0,0)/total*100:.1f}%)")
        if 1 in dist:
            print(f"  Ratio         : {dist[0]/dist[1]:.1f}:1")

        self._validate_target(data)
        return data

    # ------------------------------------------------------------------
    def _plot_risk_distribution(self, risk_score):
        os.makedirs("reports", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.hist(risk_score, bins=30, alpha=0.7, edgecolor="black")
        plt.axvline(
            x=np.percentile(risk_score, 75),
            linestyle="--", linewidth=2,
            label="Seuil (75ème percentile)",
        )
        plt.title("Distribution des Scores de Risque d'Abandon", fontsize=14)
        plt.xlabel("Score de Risque")
        plt.ylabel("Nombre d'Étudiants")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/risk_score_distribution.png", dpi=300)
        plt.close()
        print("  📈 reports/risk_score_distribution.png sauvegardé")

    # ------------------------------------------------------------------
    def _validate_target(self, data):
        print("\n🔍 Validation:")

        if data["dropout_risk"].nunique() > 1:
            corr_abs = np.corrcoef(data["absences"], data["dropout_risk"])[0, 1]
            corr_g   = np.corrcoef((data["G1"] + data["G2"]) / 2, data["dropout_risk"])[0, 1]
            print(f"  Corrélation absences      : {corr_abs:.3f}")
            print(f"  Corrélation notes moyennes: {corr_g:.3f}")
        else:
            print("  ⚠️  Classe unique — corrélations non calculables")

        criteria = {
            "Notes < 10":          (data["G1"] < 10) & (data["G2"] < 10),
            "Absences > 15":        data["absences"] > 15,
            "Échecs > 0":           data["failures"] > 0,
            "Temps étude < 2":      data["studytime"] < 2,
            "Famsup=no & Famrel<3": (data["famsup"] == "no") & (data["famrel"] < 3),
        }
        print("\n  Taux de risque par critère:")
        for name, cond in criteria.items():
            if cond.any():
                rate = data.loc[cond, "dropout_risk"].mean() * 100
                print(f"    {name}: {rate:.1f}%")