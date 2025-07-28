"""
Simple Command-line Chatbot for Laptop Recommendations
Easy-to-use interface for the laptop recommender system
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot_interface import LaptopRecommenderChatbot

def main():
    """Main function for command-line chatbot"""
    print("=" * 60)
    print("💻 LAPTOP RECOMMENDER CHATBOT")
    print("=" * 60)
    print("I can help you find the perfect laptop based on your needs!")
    print("Commands:")
    print("  - Type 'quit' to exit")
    print("  - Type 'reset' to start over")
    print("  - Type 'help' for assistance")
    print("  - Type 'web' to start web interface")
    print()
    
    # Initialize chatbot
    try:
        chatbot = LaptopRecommenderChatbot()
        print("✅ Chatbot initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print("Make sure the dataset is available in the data/ directory.")
        return
    
    print("\nBot: Hello! I'm your laptop recommendation assistant. How can I help you today?")
    print("Bot: You can tell me about your budget, usage needs, brand preferences, and more!")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Thank you for using the laptop recommender! Goodbye! 👋")
                break
            
            elif user_input.lower() == 'reset':
                chatbot.reset_conversation()
                print("Bot: Conversation reset. How can I help you today?")
                continue
            
            elif user_input.lower() == 'help':
                print("\nBot: Here's how I can help you:")
                print("• Tell me your budget (e.g., 'I have $800 to spend')")
                print("• Describe your usage (e.g., 'I need it for gaming' or 'for work')")
                print("• Mention brand preferences (e.g., 'I prefer Dell' or 'no Apple')")
                print("• Ask about performance needs (e.g., 'I need something fast')")
                print("• Specify portability requirements (e.g., 'I travel a lot')")
                print("\nI'll ask questions to understand your needs better!")
                continue
            
            elif user_input.lower() == 'web':
                print("Bot: Starting web interface...")
                print("Opening browser to: http://localhost:5000")
                try:
                    import webbrowser
                    webbrowser.open('http://localhost:5000')
                except:
                    pass
                
                print("Bot: If the web interface doesn't start automatically, run:")
                print("python web_chatbot.py")
                continue
            
            elif not user_input:
                continue
            
            # Get bot response
            print("Bot: ", end="")
            response = chatbot.get_response(user_input)
            print(response)
            
            # Show current preferences if available
            summary = chatbot.get_conversation_summary()
            if summary['user_preferences']:
                print(f"\n📋 Current preferences: {summary['user_preferences']}")
            
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nBot: Sorry, I encountered an error: {str(e)}")
            print("Bot: Please try again or type 'reset' to start over.")

if __name__ == "__main__":
    main() 