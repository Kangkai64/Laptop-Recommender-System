"""
Chatbot Interface for Laptop Recommender System
Provides conversational AI interface for laptop recommendations
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
import json
import os
from datetime import datetime

# Import our custom modules
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaptopRecommenderChatbot:
    """Chatbot interface for laptop recommendations"""
    
    def __init__(self, data_path: str = "data/Cleaned_Laptop_data.csv"):
        """
        Initialize the chatbot
        
        Args:
            data_path (str): Path to the dataset
        """
        self.data_path = data_path
        self.data = None
        self.preprocessor = None
        self.trainer = None
        self.user_preferences = {}
        self.conversation_history = []
        self.recommendation_history = []
        
        # Load data and models
        self._initialize_system()
        
        # Define conversation patterns
        self.greeting_patterns = [
            r'\b(hi|hello|hey|greetings)\b',
            r'\b(start|begin|recommend)\b',
            r'\b(laptop|computer|pc)\b'
        ]
        
        self.preference_patterns = {
            'budget': [
                r'\b(budget|price|cost|money|cheap|expensive)\b',
                r'\$(\d+)',
                r'\b(\d+)\s*(dollars?|usd)\b'
            ],
            'usage': [
                r'\b(gaming|game|play)\b',
                r'\b(work|business|office|professional)\b',
                r'\b(student|study|school|college|university)\b',
                r'\b(creative|design|video|photo|editing)\b',
                r'\b(casual|basic|simple|everyday)\b'
            ],
            'brand': [
                r'\b(dell|hp|lenovo|asus|acer|apple|macbook)\b',
                r'\b(msi|razer|alienware|gigabyte)\b'
            ],
            'performance': [
                r'\b(fast|powerful|high|performance|speed)\b',
                r'\b(slow|basic|entry|low)\b'
            ],
            'portability': [
                r'\b(light|portable|travel|mobile)\b',
                r'\b(heavy|desktop|stationary)\b'
            ]
        }
        
        # Define response templates
        self.responses = {
            'greeting': [
                "Hello! I'm your laptop recommendation assistant. I can help you find the perfect laptop based on your needs. What are you looking for?",
                "Hi there! I'm here to help you choose the right laptop. Tell me about your requirements!",
                "Welcome! I'm your laptop expert. What kind of laptop are you interested in?"
            ],
            'budget_question': [
                "What's your budget range for the laptop? (e.g., $500-1000, under $800, etc.)",
                "How much are you willing to spend on a laptop?",
                "What's your price range for this purchase?"
            ],
            'usage_question': [
                "What will you mainly use the laptop for? (gaming, work, study, creative work, etc.)",
                "How do you plan to use this laptop?",
                "What's the primary purpose of this laptop?"
            ],
            'brand_question': [
                "Do you have any brand preferences? (Dell, HP, Lenovo, ASUS, etc.)",
                "Are there any specific brands you prefer or want to avoid?",
                "Any particular laptop brands you're interested in?"
            ],
            'clarification': [
                "Could you tell me more about that?",
                "I need a bit more information to give you better recommendations.",
                "Can you elaborate on your requirements?"
            ],
            'recommendation': [
                "Based on your preferences, here are some great options:",
                "I found these laptops that match your needs:",
                "Here are my top recommendations for you:"
            ]
        }
    
    def _initialize_system(self):
        """Initialize the data and models"""
        try:
            # Load data
            loader = DataLoader(self.data_path)
            self.data = loader.load_data()
            logger.info("Dataset loaded successfully")
            
            # Initialize preprocessor
            self.preprocessor = DataPreprocessor()
            
            # Initialize trainer (for price predictions)
            self.trainer = ModelTrainer()
            
            logger.info("Chatbot system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            print("Warning: Could not load dataset. Some features may be limited.")
    
    def get_response(self, user_input: str) -> str:
        """
        Generate response based on user input
        
        Args:
            user_input (str): User's message
            
        Returns:
            str: Bot's response
        """
        # Store conversation
        self.conversation_history.append({
            'user': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Clean and analyze input
        cleaned_input = user_input.lower().strip()
        
        # Check for greetings
        if self._is_greeting(cleaned_input):
            return self._get_random_response('greeting')
        
        # Extract preferences
        preferences = self._extract_preferences(cleaned_input)
        self.user_preferences.update(preferences)
        
        # Generate appropriate response
        response = self._generate_response(cleaned_input, preferences)
        
        # Store bot response
        self.conversation_history.append({
            'bot': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _is_greeting(self, text: str) -> bool:
        """Check if input is a greeting"""
        for pattern in self.greeting_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _extract_preferences(self, text: str) -> Dict[str, Any]:
        """Extract user preferences from text"""
        preferences = {}
        
        # Extract budget
        budget_match = re.search(r'\$(\d+)', text)
        if budget_match:
            preferences['budget'] = int(budget_match.group(1))
        
        # Extract usage patterns
        for usage_type, patterns in self.preference_patterns['usage'].items():
            for pattern in patterns:
                if re.search(pattern, text):
                    preferences['usage'] = usage_type
                    break
        
        # Extract brand preferences
        for pattern in self.preference_patterns['brand']:
            match = re.search(pattern, text)
            if match:
                preferences['brand'] = match.group(0)
                break
        
        # Extract performance requirements
        if re.search(r'\b(fast|powerful|high|performance)\b', text):
            preferences['performance'] = 'high'
        elif re.search(r'\b(slow|basic|entry|low)\b', text):
            preferences['performance'] = 'low'
        
        # Extract portability requirements
        if re.search(r'\b(light|portable|travel|mobile)\b', text):
            preferences['portability'] = 'high'
        elif re.search(r'\b(heavy|desktop|stationary)\b', text):
            preferences['portability'] = 'low'
        
        return preferences
    
    def _generate_response(self, text: str, preferences: Dict[str, Any]) -> str:
        """Generate appropriate response based on input and preferences"""
        
        # If we have enough information, provide recommendations
        if len(self.user_preferences) >= 2:
            recommendations = self._get_recommendations()
            if recommendations:
                return self._format_recommendations(recommendations)
        
        # Ask for missing information
        missing_info = self._get_missing_information()
        if missing_info:
            return self._get_question_response(missing_info)
        
        # Default response
        return self._get_random_response('clarification')
    
    def _get_missing_information(self) -> Optional[str]:
        """Determine what information is missing"""
        if 'budget' not in self.user_preferences:
            return 'budget'
        elif 'usage' not in self.user_preferences:
            return 'usage'
        elif 'brand' not in self.user_preferences:
            return 'brand'
        return None
    
    def _get_question_response(self, info_type: str) -> str:
        """Get response asking for specific information"""
        if info_type == 'budget':
            return self._get_random_response('budget_question')
        elif info_type == 'usage':
            return self._get_random_response('usage_question')
        elif info_type == 'brand':
            return self._get_random_response('brand_question')
        return self._get_random_response('clarification')
    
    def _get_recommendations(self) -> List[Dict[str, Any]]:
        """Get laptop recommendations based on user preferences"""
        if self.data is None:
            return []
        
        # Filter data based on preferences
        filtered_data = self.data.copy()
        
        # Apply filters
        if 'brand' in self.user_preferences:
            brand = self.user_preferences['brand']
            filtered_data = filtered_data[
                filtered_data['Brand'].str.contains(brand, case=False, na=False)
            ]
        
        if 'budget' in self.user_preferences:
            budget = self.user_preferences['budget']
            # Assuming 'Price' is the target column
            if 'Price' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Price'] <= budget * 1.1]
        
        # Apply usage-based filtering
        if 'usage' in self.user_preferences:
            usage = self.user_preferences['usage']
            if usage == 'gaming':
                # Filter for gaming laptops (high-end GPUs, good processors)
                filtered_data = filtered_data[
                    filtered_data['GPU'].str.contains('RTX|GTX|Gaming', case=False, na=False)
                ]
            elif usage == 'work':
                # Filter for business laptops
                filtered_data = filtered_data[
                    filtered_data['Brand'].str.contains('Dell|HP|Lenovo', case=False, na=False)
                ]
            elif usage == 'student':
                # Filter for affordable laptops
                if 'Price' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['Price'] <= 1000]
        
        # Sort by price and return top recommendations
        if 'Price' in filtered_data.columns:
            filtered_data = filtered_data.sort_values('Price')
        
        # Return top 3 recommendations
        recommendations = []
        for _, row in filtered_data.head(3).iterrows():
            rec = {
                'Brand': row.get('Brand', 'Unknown'),
                'Model': row.get('Model', 'Unknown'),
                'Processor': row.get('Processor', 'Unknown'),
                'RAM': row.get('RAM', 'Unknown'),
                'Storage': row.get('Storage', 'Unknown'),
                'Price': row.get('Price', 'Unknown'),
                'GPU': row.get('GPU', 'Unknown'),
                'Screen_Size': row.get('Screen_Size', 'Unknown')
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _format_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format recommendations into a readable response"""
        if not recommendations:
            return "I couldn't find laptops matching your exact requirements. Could you try adjusting your preferences?"
        
        response = self._get_random_response('recommendation') + "\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            response += f"**{i}. {rec['Brand']} {rec['Model']}**\n"
            response += f"   • Processor: {rec['Processor']}\n"
            response += f"   • RAM: {rec['RAM']}\n"
            response += f"   • Storage: {rec['Storage']}\n"
            response += f"   • GPU: {rec['GPU']}\n"
            response += f"   • Screen: {rec['Screen_Size']}\n"
            response += f"   • Price: ${rec['Price']:,.2f}\n\n"
        
        response += "Would you like me to explain any of these options or help you refine your search?"
        
        # Store recommendations
        self.recommendation_history.append({
            'recommendations': recommendations,
            'preferences': self.user_preferences.copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _get_random_response(self, response_type: str) -> str:
        """Get a random response from the specified type"""
        responses = self.responses.get(response_type, ["I'm not sure how to respond to that."])
        return np.random.choice(responses)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation"""
        return {
            'conversation_history': self.conversation_history,
            'user_preferences': self.user_preferences,
            'recommendation_history': self.recommendation_history,
            'total_messages': len(self.conversation_history)
        }
    
    def reset_conversation(self):
        """Reset the conversation state"""
        self.user_preferences = {}
        self.conversation_history = []
        self.recommendation_history = []
        print("Conversation reset. How can I help you today?")

def main():
    """Main function to run the chatbot"""
    print("=" * 60)
    print("LAPTOP RECOMMENDER CHATBOT")
    print("=" * 60)
    print("Type 'quit' to exit, 'reset' to start over")
    print()
    
    # Initialize chatbot
    chatbot = LaptopRecommenderChatbot()
    
    print("Bot: Hello! I'm your laptop recommendation assistant. How can I help you today?")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Thank you for using the laptop recommender! Goodbye!")
                break
            
            elif user_input.lower() == 'reset':
                chatbot.reset_conversation()
                continue
            
            elif not user_input:
                continue
            
            # Get bot response
            response = chatbot.get_response(user_input)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye!")
            break
        except Exception as e:
            print(f"\nBot: Sorry, I encountered an error: {str(e)}")
            print("Bot: Please try again or type 'reset' to start over.")

if __name__ == "__main__":
    main() 