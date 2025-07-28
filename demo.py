"""
Comprehensive Demo for Laptop Recommender System
Showcases all features including ML models and chatbot interface
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)

def print_section(title):
    """Print a formatted section"""
    print(f"\n📋 {title}")
    print("-" * 40)

def demo_ml_pipeline():
    """Demo the ML pipeline"""
    print_header("MACHINE LEARNING PIPELINE DEMO")
    
    try:
        from data_loader import DataLoader
        from data_preprocessing import DataPreprocessor
        from model_training import ModelTrainer
        from model_evaluation import ModelEvaluator
        
        print_section("1. Loading Data")
        loader = DataLoader()
        data = loader.load_data()
        print(f"✅ Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
        
        print_section("2. Data Preprocessing")
        preprocessor = DataPreprocessor()
        target_col = loader.get_target_variable()
        X, y = loader.split_features_target(target_col)
        X_processed, feature_names = preprocessor.create_preprocessing_pipeline(X)
        print(f"✅ Data preprocessed: {len(feature_names)} features prepared")
        
        print_section("3. Model Training")
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_processed, y)
        print("✅ Models trained successfully")
        
        print_section("4. Model Evaluation")
        evaluator = ModelEvaluator()
        report = evaluator.create_evaluation_report(results)
        print("✅ Evaluation completed")
        
        # Show best model
        best_model_name, best_model = trainer.get_best_model()
        print(f"🏆 Best Model: {best_model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in ML pipeline: {e}")
        return False

def demo_chatbot():
    """Demo the chatbot interface"""
    print_header("CHATBOT INTERFACE DEMO")
    
    try:
        from chatbot_interface import LaptopRecommenderChatbot
        
        print_section("Initializing Chatbot")
        chatbot = LaptopRecommenderChatbot()
        print("✅ Chatbot initialized successfully")
        
        print_section("Sample Conversation")
        sample_conversations = [
            "Hi, I need a laptop for gaming",
            "My budget is around $1000",
            "I prefer Dell or HP",
            "I need something portable for travel"
        ]
        
        for message in sample_conversations:
            print(f"\nUser: {message}")
            response = chatbot.get_response(message)
            print(f"Bot: {response}")
            time.sleep(1)
        
        print_section("Conversation Summary")
        summary = chatbot.get_conversation_summary()
        print(f"Total messages: {summary['total_messages']}")
        print(f"User preferences: {summary['user_preferences']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in chatbot demo: {e}")
        return False

def demo_prediction():
    """Demo the prediction functionality"""
    print_header("PREDICTION DEMO")
    
    try:
        from predict import load_model, predict_price
        from data_preprocessing import DataPreprocessor
        
        print_section("Loading Trained Model")
        model, preprocessor = load_model()
        if model is None:
            print("⚠️  No trained model found. Run 'python main.py' first.")
            return False
        
        print("✅ Model loaded successfully")
        
        print_section("Sample Predictions")
        sample_laptops = [
            {
                'Brand': 'Dell',
                'Processor': 'Intel Core i5',
                'RAM': '8 GB',
                'Storage': '512 GB SSD',
                'GPU': 'Integrated',
                'Screen_Size': '15.6 inches',
                'Weight': '2.1 kg'
            },
            {
                'Brand': 'HP',
                'Processor': 'AMD Ryzen 7',
                'RAM': '16 GB',
                'Storage': '1 TB SSD',
                'GPU': 'NVIDIA GTX 1650',
                'Screen_Size': '14 inches',
                'Weight': '1.8 kg'
            }
        ]
        
        for i, laptop in enumerate(sample_laptops, 1):
            try:
                predicted_price = predict_price(model, preprocessor, laptop)
                print(f"\nLaptop {i}:")
                for key, value in laptop.items():
                    print(f"  {key}: {value}")
                print(f"  Predicted Price: ${predicted_price:,.2f}")
            except Exception as e:
                print(f"  Error predicting price: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction demo: {e}")
        return False

def show_available_features():
    """Show all available features"""
    print_header("AVAILABLE FEATURES")
    
    features = [
        "📊 Data Loading and Exploration",
        "🔧 Data Preprocessing and Feature Engineering",
        "🤖 Multiple ML Models (Linear, Random Forest, Gradient Boosting, etc.)",
        "📈 Model Evaluation and Comparison",
        "🎨 Visualization and Results Analysis",
        "💬 Conversational AI Chatbot Interface",
        "🌐 Web-based Chatbot with Modern UI",
        "💻 Command-line Chatbot for Easy Testing",
        "🎯 Interactive Price Predictions",
        "📋 Personalized Laptop Recommendations"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n📁 Project Structure:")
    structure = [
        "├── main.py (ML pipeline)",
        "├── predict.py (price predictions)",
        "├── chatbot.py (command-line chatbot)",
        "├── web_chatbot.py (web interface)",
        "├── src/ (ML modules)",
        "├── data/ (dataset)",
        "├── models/ (trained models)",
        "└── results/ (evaluation results)"
    ]
    
    for item in structure:
        print(f"  {item}")

def main():
    """Main demo function"""
    print_header("LAPTOP RECOMMENDER SYSTEM - COMPREHENSIVE DEMO")
    
    print("This demo showcases all features of the laptop recommender system.")
    print("Choose an option:")
    print("1. Show available features")
    print("2. Demo ML pipeline")
    print("3. Demo chatbot interface")
    print("4. Demo prediction functionality")
    print("5. Run all demos")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                show_available_features()
                
            elif choice == '2':
                demo_ml_pipeline()
                
            elif choice == '3':
                demo_chatbot()
                
            elif choice == '4':
                demo_prediction()
                
            elif choice == '5':
                print_header("RUNNING ALL DEMOS")
                show_available_features()
                demo_ml_pipeline()
                demo_chatbot()
                demo_prediction()
                print_header("ALL DEMOS COMPLETED")
                
            elif choice == '6':
                print("👋 Thanks for trying the demo!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 