import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class CricketModelTrainer:
   
    
    def __init__(self):
        self.categorical_features = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']
        self.numerical_features = ['runs_required', 'balls_remaining', 'wickets_in_hand', 
                                   'target_match', 'current_run_rate', 'required_run_rate']
        self.target = 'win'
        self.model = None
        self.preprocessor = None
        
    def create_model_pipeline(self):
       
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
       
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
       
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
       
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
        
        return self.model
    
    def train(self, data_path: str):
       
        try:
           
            df = pd.read_csv(data_path)
            print(f"Loaded data with shape: {df.shape}")
            
           
            self.create_model_pipeline()
            
           
            X = df[self.categorical_features + self.numerical_features]
            y = df[self.target]
            
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
           
            print("Training model...")
            self.model.fit(X_train, y_train)
            print("Model training complete.")
            
           
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nAccuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            return self.model, accuracy
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise
    
    def save_model(self, model_dir: str = "models"):
       
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = os.path.join(model_dir, "cricket_model.pkl")
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        
        info = {
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'target': self.target
        }
        info_path = os.path.join(model_dir, "model_info.pkl")
        joblib.dump(info, info_path)
        print(f"Model info saved to: {info_path}")
        
        return model_path

if __name__ == "__main__":
   
    trainer = CricketModelTrainer()
    
    data_path = "../cricket_features.csv"
    
    if os.path.exists(data_path):
        trainer.train(data_path)
        trainer.save_model()
    else:
        print(f"Data file not found: {data_path}")
        print("Please provide the path to cricket_features.csv")
