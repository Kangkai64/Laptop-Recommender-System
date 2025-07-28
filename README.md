# Laptop Price Prediction AI Model

This project uses Machine Learning to predict laptop prices based on various features using the dataset from Kaggle.

## Dataset
- Source: [Latest Laptop Price List](https://www.kaggle.com/datasets/kuchhbhi/latest-laptop-price-list/data?select=Cleaned_Laptop_data.csv)
- File: `Cleaned_Laptop_data.csv`

## Project Structure
```
AI Assignment/
├── requirements.txt          # Python dependencies
├── README.md               # Project documentation
├── data/                   # Data directory
│   └── Cleaned_Laptop_data.csv
├── src/                    # Source code
│   ├── data_loader.py     # Data loading utilities
│   ├── data_preprocessing.py # Data preprocessing functions
│   ├── model_training.py  # Model training script
│   └── model_evaluation.py # Model evaluation utilities
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks
└── results/               # Results and visualizations
```

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   - Download `Cleaned_Laptop_data.csv` from the Kaggle link
   - Place it in the `data/` directory

3. **Run the Model Training**
   ```bash
   python src/model_training.py
   ```

## Features
- Data preprocessing and feature engineering
- Multiple ML models (Random Forest, Linear Regression, etc.)
- Model evaluation and comparison
- Visualization of results
- Price prediction functionality
- **Conversational AI chatbot interface** for interactive recommendations
- **Web-based chatbot** with modern UI
- **Command-line chatbot** for easy testing

## Usage

### Basic ML Pipeline
1. Ensure the dataset is in the `data/` folder
2. Run the training script: `python main.py`
3. Check results in the `results/` folder
4. Use the trained model for predictions: `python predict.py`

### Chatbot Interface
1. **Command-line chatbot**: `python chatbot.py`
2. **Web-based chatbot**: `python web_chatbot.py` (then open http://localhost:5000)

### Interactive Recommendations
The chatbot can understand natural language and provide personalized recommendations based on:
- Budget constraints
- Usage requirements (gaming, work, study, etc.)
- Brand preferences
- Performance needs
- Portability requirements 

## Project Overview

Problem Background
The technology and the specifications of the laptop industry is evolving rapidly. Laptops are expensive. Thus, users need to choose wisely, compare and contrast before buying a laptop.  Users who have no technological background find it difficult to choose a suitable and affordable laptop that fits their daily use, career or academic studies. Visiting laptop stores physically can be very costly in terms of time and money for certain users, especially those from rural areas. However, the user cannot judge if the laptop is suitable for them without enough information, or without actually testing the performance of the laptop and getting some advice from the seller. 

Besides, they also need to spend a significant amount of time surfing the internet so they do not miss out the best deal. They neither want to be scammed by product offerings with prices that are too good to be true, nor joining the overpayer club. They hope to make the laptop selection process smoother and make their life easier.
Objectives/Aims

The objectives of the Laptop Recommender System are:
Best-Fit laptop selection: The system is able to recommend a suitable laptop based on the current trend, latest technology and a holistic view of user scenario, covering at least 80% of the user requirements.

Improve user satisfaction: The system is focused on providing better recommendations by actively prompting the user with relevant questions and refining the criterias.
Reduce Internet fraud: The system could reduce the fraud victims of laptop online purchasing.