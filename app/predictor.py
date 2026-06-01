import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any

class StudentDropoutPredictor:
    """
    Classe pour faire des prédictions avec le modèle entraîné
    """
    
    def __init__(self, model_path: str = 'models/best_model.pkl',
                 preprocessor_path: str = 'models/preprocessor.pkl'):
        """
        Initialise le prédicteur avec le modèle et préprocesseur
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
            preprocessor_path: Chemin vers le préprocesseur sauvegardé
        """
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            self.loaded = True
        except Exception as e:
            print(f"⚠️ Erreur chargement: {e}")
            self.model = None
            self.preprocessor = None
            self.loaded = False
    
    def prepare_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prépare les données d'entrée pour le modèle
        
        Args:
            input_data: Dictionnaire avec les données de l'étudiant
            
        Returns:
            DataFrame prêt pour la prédiction
        """
        # Mapping des features attendues
        expected_features = [
            'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
            'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
            'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
            'paid', 'activities', 'nursery', 'higher', 'internet',
            'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
            'health', 'absences', 'G1', 'G2'
        ]
        
        # Créer un DataFrame avec toutes les features
        df = pd.DataFrame([input_data])
        
        # S'assurer que toutes les colonnes sont présentes
        for feature in expected_features:
            if feature not in df.columns:
                # Valeurs par défaut selon le type
                if feature in ['G1', 'G2', 'age', 'absences', 'failures']:
                    df[feature] = 0
                elif feature in ['Medu', 'Fedu', 'traveltime', 'studytime', 
                                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']:
                    df[feature] = 3  # Valeur moyenne
                else:
                    df[feature] = 'unknown'
        
        # Réorganiser les colonnes
        df = df[expected_features]
        
        return df
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fait une prédiction pour un étudiant
        
        Args:
            input_data: Dictionnaire avec les données de l'étudiant
            
        Returns:
            Dictionnaire avec les résultats de prédiction
        """
        try:
            if not self.loaded:
                return self._demo_prediction(input_data)
            
            # Préparer les données
            df = self.prepare_input(input_data)
            
            # Appliquer le préprocesseur
            X_processed = self.preprocessor.transform(df)
            
            # Faire la prédiction
            prediction = self.model.predict(X_processed)[0]
            probability = self.model.predict_proba(X_processed)[0][1]
            
            # Analyser les facteurs
            factors = self._analyze_factors(input_data)
            
            # Générer des recommandations
            recommendations = self._generate_recommendations(prediction, probability, factors)
            
            # Préparer la réponse
            return {
                'success': True,
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': "ÉLEVÉ" if prediction == 1 else "FAIBLE",
                'risk_percentage': round(probability * 100, 1),
                'factors': factors,
                'recommendations': recommendations,
                'model_loaded': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_loaded': self.loaded
            }
    
    def _demo_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prédiction de démonstration si modèle non chargé"""
        absences = input_data.get('absences', 0)
        g1 = input_data.get('G1', 10)
        g2 = input_data.get('G2', 10)
        
        # Logique simple de démo
        risk_score = (
            (absences > 10) * 0.4 +
            (g1 < 10) * 0.3 +
            (g2 < 10) * 0.3
        )
        
        probability = min(risk_score + 0.1, 0.95)  # Ajouter un peu de bruit
        prediction = 1 if probability > 0.5 else 0
        
        factors = self._analyze_factors(input_data)
        recommendations = self._generate_recommendations(prediction, probability, factors)
        
        return {
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'risk_level': "ÉLEVÉ" if prediction == 1 else "FAIBLE",
            'risk_percentage': round(probability * 100, 1),
            'factors': factors,
            'recommendations': recommendations,
            'model_loaded': False,
            'note': "Mode démonstration - modèle non chargé"
        }
    
    def _analyze_factors(self, input_data: Dict[str, Any]) -> List[str]:
        """Analyse les facteurs contribuant au risque"""
        factors = []
        
        # Critères de risque
        if input_data.get('absences', 0) > 10:
            factors.append(f"Absences élevées ({input_data.get('absences', 0)} jours)")
        
        if input_data.get('G1', 20) < 10:
            factors.append(f"Note G1 faible: {input_data.get('G1', 'N/A')}/20")
        
        if input_data.get('G2', 20) < 10:
            factors.append(f"Note G2 faible: {input_data.get('G2', 'N/A')}/20")
        
        if input_data.get('failures', 0) > 0:
            factors.append(f"Échecs précédents: {input_data.get('failures', 0)}")
        
        if input_data.get('studytime', 4) < 2:
            factors.append(f"Temps d'étude insuffisant: {input_data.get('studytime', 1)}/4")
        
        if input_data.get('famsup', 'no') == 'no':
            factors.append("Support familial limité")
        
        if input_data.get('famrel', 3) < 3:
            factors.append(f"Relations familiales difficiles: {input_data.get('famrel', 3)}/5")
        
        if len(factors) == 0:
            factors.append("Aucun facteur de risque majeur identifié")
        
        return factors
    
    def _generate_recommendations(self, prediction: int, 
                                 probability: float, 
                                 factors: List[str]) -> List[str]:
        """Génère des recommandations personnalisées"""
        recommendations = []
        
        if prediction == 1 or probability > 0.6:
            recommendations.append("⚠️ Intervention recommandée - Consulter un conseiller pédagogique")
            
            if any("Absences" in factor for factor in factors):
                recommendations.append("📅 Mettre en place un suivi des présences")
            
            if any("Note" in factor for factor in factors):
                recommendations.append("📚 Proposer un soutien académique/tutorat")
            
            if any("Support familial" in factor for factor in factors):
                recommendations.append("👨‍👩‍👧‍👦 Organiser une rencontre avec la famille")
            
            if any("Temps d'étude" in factor for factor in factors):
                recommendations.append("⏰ Établir un planning de travail")
            
            recommendations.append("🎯 Planifier un suivi régulier toutes les 2 semaines")
        else:
            recommendations.append("✅ Situation stable - Continuer le suivi habituel")
            recommendations.append("📊 Maintenir l'engagement académique")
            recommendations.append("🌟 Encourager la participation aux activités")
        
        return recommendations
    
    def batch_predict(self, students_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prédictions pour plusieurs étudiants
        
        Args:
            students_data: Liste de dictionnaires avec données étudiants
            
        Returns:
            Liste de prédictions
        """
        results = []
        for student_data in students_data:
            result = self.predict(student_data)
            results.append(result)
        
        return results

# Instance globale pour l'application
predictor = StudentDropoutPredictor()