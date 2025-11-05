import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging


logger = logging.getLogger(__name__)


try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.debug("SHAP not available. Using feature importance instead.")

class CricketPredictor:
    
    
    def _init_(self, model_path: str = None):
        self.model = None
        self.model_info = None
        self.explainer = None
        
        if model_path is None:
            # Default path
            base_dir = Path(_file_).parent.parent.parent
            model_path = base_dir / "models" / "cricket_model.pkl"
        
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.debug(f"Model loaded from: {self.model_path}")
                
               
                info_path = str(self.model_path).replace("cricket_model.pkl", "model_info.pkl")
                if os.path.exists(info_path):
                    self.model_info = joblib.load(info_path)
                    logger.debug(f"Model info loaded from: {info_path}")
                
               
                self._initialize_explainer()
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                logger.debug("Using mock predictions. Train the model first using model_trainer.py")
        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            logger.debug("Using mock predictions")
    
    def _initialize_explainer(self):
       
        if not SHAP_AVAILABLE:
            logger.debug("SHAP not available. Will use feature importance instead.")
            self.explainer = None
            return
            
        try:
           
            classifier = self.model.named_steps['classifier']
            
            self.explainer = shap.TreeExplainer(classifier)
            logger.debug("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def predict(self, input_data: Dict) -> Tuple[str, float, List[Dict]]:
        
        if self.model is None:
            return self._mock_prediction(input_data)
        
        try:
            
            df = self._prepare_input(input_data)
            
            
            probabilities = self.model.predict_proba(df)[0]
            prediction = self.model.predict(df)
            
            
            logger.debug(f"probabilities array: {probabilities}")
            logger.debug(f"prediction: {prediction}")
            
           
            batting_team_win_probability = probabilities[1]  # Class 1 = batting team wins
            
            
            predicted_class_idx = int(prediction[0]) if hasattr(prediction, '_len_') else int(prediction)
            
            logger.debug(f"predicted_class_idx: {predicted_class_idx}")
            logger.debug(f"batting_team_win_prob: {batting_team_win_probability}")
            
           
            if batting_team_win_probability > 0.5:
                winner = input_data.get('batting_team')
            else:
                winner = input_data.get('bowling_team')
            
            logger.debug(f"winner: {winner}")
            
            
            shap_values = self._get_shap_explanation(df)
            
            
            return winner, float(batting_team_win_probability), shap_values
            
        except Exception as e:
            logger.exception(f"Error during prediction: {e}")
            return self._mock_prediction(input_data)
    
    def _prepare_input(self, input_data: Dict) -> pd.DataFrame:
        
        data = {
            'batting_team': input_data.get('batting_team', input_data.get('team1')),
            'bowling_team': input_data.get('bowling_team', input_data.get('team2')),
            'venue': input_data.get('venue'),
            'toss_winner': input_data.get('toss_winner', input_data.get('team1')),
            'toss_decision': input_data.get('toss_decision', 'bat'),
            'runs_required': input_data.get('runs_required', 150),
            'balls_remaining': input_data.get('balls_remaining', 120),
            'wickets_in_hand': input_data.get('wickets_in_hand', 10),
            'target_match': input_data.get('target_match', 250),
            'current_run_rate': input_data.get('current_run_rate', 6.0),
            'required_run_rate': input_data.get('required_run_rate', 7.5)
        }
        
        return pd.DataFrame([data])
    
    def _get_shap_explanation(self, df: pd.DataFrame) -> List[Dict]:
       
        if self.explainer is None:
            return self._get_feature_importance_explanation(df)
        
        try:
           
            preprocessor = self.model.named_steps['preprocessor']
            X_transformed = preprocessor.transform(df)
            
            
            shap_values_raw = self.explainer.shap_values(X_transformed)
            
            
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[1]
            
            
            feature_names = self._get_feature_names()
            
           
            shap_list = []
            for idx, value in enumerate(shap_values_raw[0]):
                if abs(value) > 0.01: 
                    shap_list.append({
                        'feature': feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                        'value': float(value),
                        'impact': 'positive' if value > 0 else 'negative' if value < 0 else 'neutral'
                    })
            
           
            shap_list = sorted(shap_list, key=lambda x: abs(x['value']), reverse=True)[:10]
            
            return shap_list
            
        except Exception as e:
            logger.exception(f"Error generating SHAP values: {e}")
            return self._get_feature_importance_explanation(df)
    
    def _get_feature_importance_explanation(self, df: pd.DataFrame) -> List[Dict]:
       
        try:
            classifier = self.model.named_steps['classifier']
            feature_names = self._get_feature_names()
            
           
            importances = classifier.feature_importances_
            
            
            input_data = df.iloc[0]
            
            
            importance_list = []
            
           
            numerical_features = {
                'runs_required': input_data.get('runs_required', 0),
                'wickets_in_hand': input_data.get('wickets_in_hand', 0),
                'balls_remaining': input_data.get('balls_remaining', 0),
                'required_run_rate': input_data.get('required_run_rate', 0),
                'current_run_rate': input_data.get('current_run_rate', 0),
            }
            
            
            for feature, value in numerical_features.items():
                if feature in feature_names:
                    idx = feature_names.index(feature)
                    base_importance = importances[idx] if idx < len(importances) else 0.1
                    
                    
                    impact_direction = 'positive'
                    impact_value = base_importance
                    
                    if feature == 'runs_required':
                        
                        impact_direction = 'negative' if value > 100 else 'positive'
                        impact_value = base_importance * (value / 200)  # Scale by value
                    elif feature == 'wickets_in_hand':
                       
                        impact_direction = 'positive'
                        impact_value = base_importance * (value / 10)
                    elif feature == 'required_run_rate':
                        
                        impact_direction = 'negative' if value > 8 else 'positive'
                        impact_value = base_importance * min(value / 10, 1.0)
                    elif feature == 'current_run_rate':
                       
                        impact_direction = 'positive'
                        impact_value = base_importance * min(value / 10, 1.0)
                    elif feature == 'balls_remaining':
                        
                        impact_direction = 'positive'
                        impact_value = base_importance * (value / 120)
                    
                    importance_list.append({
                        'feature': self._clean_feature_name(feature),
                        'value': float(impact_value),
                        'impact': impact_direction
                    })
            
            
            importance_list = sorted(importance_list, key=lambda x: abs(x['value']), reverse=True)[:5]
            
            return importance_list
            
        except Exception as e:
            logger.exception(f"Error generating feature importance: {e}")
            return self._default_shap_values()
    
    def _clean_feature_name(self, name: str) -> str:
        
       
        if '_' in name:
            parts = name.split('_', 1)
            if parts[0] in ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']:
                return f"{parts[0].replace('_', ' ').title()}: {parts[1]}"
        
       
        name = name.replace('_', ' ').title()
        return name
    
    def _get_feature_names(self) -> List[str]:
        
        try:
            preprocessor = self.model.named_steps['preprocessor']
            
           
            num_features = self.model_info['numerical_features']
            
            
            cat_transformer = preprocessor.named_transformers_['cat']
            onehot = cat_transformer.named_steps['onehot']
            cat_features = onehot.get_feature_names_out(self.model_info['categorical_features'])
            
           
            all_features = list(num_features) + list(cat_features)
            return all_features
            
        except Exception as e:
            logger.exception(f"Error getting feature names: {e}")
            return []
    
    def _default_shap_values(self) -> List[Dict]:
        """Return default SHAP values when actual calculation fails"""
        return [
            {'feature': 'runs_required', 'value': 0.15, 'impact': 'positive'},
            {'feature': 'wickets_in_hand', 'value': 0.12, 'impact': 'positive'},
            {'feature': 'required_run_rate', 'value': -0.10, 'impact': 'negative'},
            {'feature': 'balls_remaining', 'value': 0.08, 'impact': 'positive'},
            {'feature': 'current_run_rate', 'value': 0.06, 'impact': 'positive'},
        ]
    
    def _mock_prediction(self, input_data: Dict) -> Tuple[str, float, List[Dict]]:
        """Mock prediction when model is not available"""
        import random
        probability = random.uniform(0.55, 0.85)
        winner = input_data.get('team1', input_data.get('batting_team', 'Team 1'))
        shap_values = self._default_shap_values()
        return winner, probability, shap_values