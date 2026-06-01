import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# =============================================================================
# PYTHONPATH — doit être AVANT joblib.load pour que les classes src/ soient
# trouvées lors de la désérialisation du pickle
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "app" / "templates"),
    static_folder=str(STATIC_DIR),
)

# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle (pipeline complet) chargé avec succès")
except Exception as e:
    print(f"⚠️ Erreur chargement modèle: {e}")
    model = None


# =============================================================================
# LOGIQUE MÉTIER
# =============================================================================
class StudentPredictor:
    def __init__(self):
        self.default_values = {
            "failures": 0,
            "studytime": 2,
            "absences": 0,
            "G1": 10,
            "G2": 10,
            "G3": 10,
            "Medu": 2,
            "Fedu": 2,
            "traveltime": 1,
            "famsup": "yes",
            "schoolsup": "no",
            "paid": "no",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
            "famsize": "GT3",
            "Pstatus": "T",
            "Mjob": "other",
            "Fjob": "other",
            "reason": "course",
            "guardian": "mother",
            "course": "general",
            "famrel": 4,
            "freetime": 3,
            "goout": 3,
            "Dalc": 1,
            "Walc": 1,
            "health": 4,
            "address": "U",
            "sex": "F",
            "school": "GP",
            "age": 17,
        }

    def predict(self, input_data):
        if model is None:
            return {
                "success": False,
                "error": (
                    "Modèle non chargé. Lancez d'abord main.py depuis la "
                    "racine du projet pour entraîner et sauvegarder le modèle."
                ),
            }

        try:
            merged = {**self.default_values, **input_data}
            df = pd.DataFrame([merged])

            prediction = int(model.predict(df)[0])
            probability = float(model.predict_proba(df)[0][1])

            # ----------------------------------------------------------------
            # Correction métier : si les signaux d'alerte sont forts,
            # on force la probabilité affichée à un minimum cohérent
            # indépendamment de ce que le modèle renvoie.
            # Le modèle XGBoost sous-pondère les absences face aux notes.
            # ----------------------------------------------------------------
            absences  = input_data.get("absences", 0)
            g1        = input_data.get("G1", 20)
            g2        = input_data.get("G2", 20)
            failures  = input_data.get("failures", 0)
            studytime = input_data.get("studytime", 4)

            risk_floor = probability

            # Absences très élevées seules → plancher à 45 %
            if absences > 25:
                risk_floor = max(risk_floor, 0.45)
            # Absences élevées seules → plancher à 35 %
            elif absences > 15:
                risk_floor = max(risk_floor, 0.35)

            # Combinaison absences + notes sous la moyenne → plancher à 55 %
            if absences > 15 and (g1 < 12 or g2 < 12):
                risk_floor = max(risk_floor, 0.55)

            # Absences très élevées + notes quelconques → plancher à 60 %
            if absences > 25 and (g1 < 14 or g2 < 14):
                risk_floor = max(risk_floor, 0.60)

            # Échecs multiples → plancher à 50 %
            if failures >= 2:
                risk_floor = max(risk_floor, 0.50)

            # Si le plancher a été relevé, on reclasse en "à risque"
            if risk_floor > 0.50:
                prediction = 1

            probability = risk_floor
            # ----------------------------------------------------------------

            factors         = self._analyze_factors(input_data)
            recommendations = self._get_recommendations(prediction, probability, factors)

            return {
                "success": True,
                "prediction": prediction,
                "risk_percentage": round(probability * 100, 1),
                "factors": factors,
                "recommendations": recommendations,
                "model_loaded": True,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    def _analyze_factors(self, d):
        factors = []

        absences  = d.get("absences", 0)
        g1        = d.get("G1", 20)
        g2        = d.get("G2", 20)
        failures  = d.get("failures", 0)
        studytime = d.get("studytime", 4)

        # --- Absences (seuils gradués) ---
        if absences > 25:
            factors.append(f"Absences très élevées ({absences} jours) — signal critique")
        elif absences > 15:
            factors.append(f"Absences élevées ({absences} jours) — surveillance recommandée")
        elif absences > 10:
            factors.append(f"Absences notables ({absences} jours)")

        # --- Notes (seuil 12/20 = sous la moyenne réelle du dataset) ---
        if g1 < 8 or g2 < 8:
            factors.append(f"Notes très faibles (G1 : {g1}, G2 : {g2}) — risque critique")
        elif g1 < 10 or g2 < 10:
            factors.append(f"Notes faibles (G1 : {g1}, G2 : {g2})")
        elif g1 < 12 or g2 < 12:
            factors.append(f"Notes en dessous de la moyenne (G1 : {g1}, G2 : {g2})")

        # --- Combinaison absences + notes ---
        if absences > 10 and (g1 < 13 or g2 < 13):
            factors.append("Combinaison absences + notes insuffisantes — intervention conseillée")

        # --- Échecs ---
        if failures >= 2:
            factors.append(f"Plusieurs échecs précédents ({failures}) — facteur aggravant")
        elif failures == 1:
            factors.append("Un échec précédent")

        # --- Temps d'étude ---
        if studytime < 2:
            factors.append("Temps d'étude insuffisant (< 2h/semaine)")

        # --- Support ---
        no_famsup    = d.get("famsup", "yes") == "no"
        no_schoolsup = d.get("schoolsup", "no") == "no"
        if no_famsup and no_schoolsup:
            factors.append("Aucun support scolaire ni familial")
        elif no_famsup:
            factors.append("Manque de support familial")

        if not factors:
            factors.append("Aucun facteur de risque majeur identifié")

        return factors

    def _get_recommendations(self, prediction, probability, factors):
        recommendations = []

        critical = any(
            kw in f
            for f in factors
            for kw in ["critique", "très élevées", "Combinaison", "aggravant"]
        )

        if critical or probability > 0.55:
            recommendations.append("Entretien individuel urgent avec un conseiller pédagogique")
            recommendations.append("Mise en place immédiate d'un plan de suivi personnalisé")
            if any("Absences" in f or "absences" in f for f in factors):
                recommendations.append("Suivi renforcé des présences — contact famille recommandé")
            if any("Notes" in f for f in factors):
                recommendations.append("Tutorat académique conseillé dès la semaine prochaine")
            if any("Temps" in f for f in factors):
                recommendations.append("Accompagnement à la méthode et organisation du travail")
            if any("support" in f.lower() for f in factors):
                recommendations.append("Orientation vers le service social de l'établissement")

        elif prediction == 1 or probability > 0.40:
            recommendations.append("Consultation avec un conseiller pédagogique")
            recommendations.append("Mise en place d'un plan de suivi personnalisé")
            if any("Absences" in f or "absences" in f for f in factors):
                recommendations.append("Suivi renforcé des présences")
            if any("Notes" in f for f in factors):
                recommendations.append("Tutorat académique conseillé")
            if any("Temps" in f for f in factors):
                recommendations.append("Accompagnement à la méthode de travail")

        else:
            recommendations.append("Situation stable — continuer le suivi régulier")
            recommendations.append("Encourager la participation aux activités")

        return recommendations


predictor = StudentPredictor()


# =============================================================================
# ROUTES
# =============================================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"success": False, "error": "JSON invalide"}), 400
    return jsonify(predictor.predict(data))


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
    })


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("🚀 Application de prédiction d'abandon étudiant")
    print(f"📂 Racine projet : {BASE_DIR}")
    print(f"🤖 Modèle        : {MODEL_PATH}")
    print(f"🌐 URL           : http://localhost:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)