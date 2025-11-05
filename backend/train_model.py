"""
Script to train the cricket prediction model
Run this script after installing dependencies to train and save the model
"""
import sys
import os
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from app.ml.model_trainer import CricketModelTrainer

def main():
    print("=" * 60)
    print("Cricket Match Prediction Model Training")
    print("=" * 60)
    
   
    trainer = CricketModelTrainer()
    
   
    data_path = Path(__file__).parent.parent / "cricket_features.csv"
    
    if not data_path.exists():
        print(f"\n❌ Error: Data file not found at {data_path}")
        print("\nPlease ensure cricket_features.csv is in the project root directory.")
        print("Expected location: ", data_path)
        return
    
    print(f"\n✓ Found data file: {data_path}")
    
    try:
       
        model, accuracy = trainer.train(str(data_path))
        
        models_dir = Path(__file__).parent / "models"
        model_path = trainer.save_model(str(models_dir))
        
        print("\n" + "=" * 60)
        print("✓ Training Complete!")
        print("=" * 60)
        print(f"Model Accuracy: {accuracy:.2%}")
        print(f"Model saved to: {model_path}")
        print("\nYou can now start the FastAPI server and make predictions!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
