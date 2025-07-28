"""
Prediction Script for Laptop Price Prediction
Allows users to make price predictions using the trained model
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor

def load_model(model_name: str = "random_forest") -> tuple:
    """
    Load the trained model and preprocessor
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        tuple: (model, preprocessor)
    """
    try:
        # Load model
        model_path = os.path.join("models", f"{model_name}.joblib")
        model = joblib.load(model_path)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        return model, preprocessor
        
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        print("Please run the training script first: python main.py")
        return None, None

def predict_price(model, preprocessor, laptop_specs: dict) -> float:
    """
    Predict laptop price based on specifications
    
    Args:
        model: Trained model
        preprocessor: DataPreprocessor instance
        laptop_specs (dict): Dictionary with laptop specifications
        
    Returns:
        float: Predicted price
    """
    # Create DataFrame from specifications
    sample_df = pd.DataFrame([laptop_specs])
    
    # Apply preprocessing
    sample_processed, _ = preprocessor.create_preprocessing_pipeline(sample_df)
    
    # Make prediction
    prediction = model.predict(sample_processed)[0]
    
    return prediction

def get_user_input() -> dict:
    """
    Get laptop specifications from user input
    
    Returns:
        dict: Laptop specifications
    """
    print("\nEnter Laptop Specifications:")
    print("-" * 30)
    
    specs = {}
    
    # Brand
    specs['Brand'] = input("Brand (e.g., Dell, HP, Lenovo): ").strip()
    
    # Processor
    specs['Processor'] = input("Processor (e.g., Intel Core i5, AMD Ryzen 7): ").strip()
    
    # RAM
    specs['RAM'] = input("RAM (e.g., 8 GB, 16 GB): ").strip()
    
    # Storage
    specs['Storage'] = input("Storage (e.g., 512 GB SSD, 1 TB HDD): ").strip()
    
    # GPU
    specs['GPU'] = input("GPU (e.g., Integrated, NVIDIA GTX 1650): ").strip()
    
    # Screen Size
    specs['Screen_Size'] = input("Screen Size (e.g., 15.6 inches): ").strip()
    
    # Weight
    specs['Weight'] = input("Weight (e.g., 2.1 kg): ").strip()
    
    return specs

def main():
    """Main function for price prediction"""
    
    print("=" * 50)
    print("LAPTOP PRICE PREDICTION")
    print("=" * 50)
    
    # Load model
    print("Loading trained model...")
    model, preprocessor = load_model()
    
    if model is None:
        return
    
    print("✓ Model loaded successfully!")
    
    while True:
        print("\n" + "=" * 50)
        print("PRICE PREDICTION")
        print("=" * 50)
        
        # Get user input
        laptop_specs = get_user_input()
        
        # Validate input
        if not all(laptop_specs.values()):
            print("❌ Please provide all specifications!")
            continue
        
        try:
            # Make prediction
            predicted_price = predict_price(model, preprocessor, laptop_specs)
            
            print("\n" + "=" * 50)
            print("PREDICTION RESULTS")
            print("=" * 50)
            
            print("Laptop Specifications:")
            for key, value in laptop_specs.items():
                print(f"  {key}: {value}")
            
            print(f"\n🏆 Predicted Price: ${predicted_price:,.2f}")
            
            # Provide price range
            lower_bound = predicted_price * 0.9
            upper_bound = predicted_price * 1.1
            print(f"📊 Estimated Price Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            print("Please check your input specifications and try again.")
        
        # Ask if user wants to continue
        continue_prediction = input("\nMake another prediction? (y/n): ").strip().lower()
        if continue_prediction != 'y':
            break
    
    print("\nThank you for using the Laptop Price Predictor!")
    print("=" * 50)

if __name__ == "__main__":
    main() 