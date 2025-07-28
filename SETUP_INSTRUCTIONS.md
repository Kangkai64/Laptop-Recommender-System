# MCP Setup Instructions for Laptop Price Prediction

This document provides step-by-step instructions to set up and run the Machine Learning project for laptop price prediction using sklearn and pandas.

## 🎯 Project Overview

This project implements a complete Machine Learning pipeline for predicting laptop prices based on specifications like brand, processor, RAM, storage, GPU, screen size, and weight.

## 📁 Project Structure

```
AI Assignment/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── main.py                     # Main training script
├── predict.py                  # Interactive prediction script
├── setup.py                    # Package setup
├── .gitignore                  # Git ignore file
├── SETUP_INSTRUCTIONS.md       # This file
├── data/                       # Data directory
│   └── Cleaned_Laptop_data.csv # Dataset (download required)
├── src/                        # Source code modules
│   ├── data_loader.py         # Data loading utilities
│   ├── data_preprocessing.py  # Data preprocessing functions
│   ├── model_training.py      # Model training script
│   └── model_evaluation.py    # Model evaluation utilities
├── models/                     # Trained models (created after training)
├── notebooks/                  # Jupyter notebooks
│   └── analysis_script.py     # Interactive analysis script
├── results/                    # Results and visualizations (created after training)
└── Tutorial_on_Conversational_Recommendation_Systems.pdf
```

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 2: Download Dataset

1. Go to the Kaggle dataset: [Latest Laptop Price List](https://www.kaggle.com/datasets/kuchhbhi/latest-laptop-price-list/data?select=Cleaned_Laptop_data.csv)
2. Download the `Cleaned_Laptop_data.csv` file
3. Place it in the `data/` directory

### Step 3: Run the Complete Pipeline

```bash
# Run the main training script
python main.py
```

This will:
- Load and preprocess the data
- Train multiple ML models (Linear Regression, Random Forest, Gradient Boosting, etc.)
- Evaluate model performance
- Save the best model
- Generate visualizations and reports

### Step 4: Try the Chatbot Interface

```bash
# Command-line chatbot
python chatbot.py

# Web-based chatbot
python web_chatbot.py
# Then open http://localhost:5000 in your browser
```

### Step 5: Make Predictions

```bash
# Run the interactive prediction script
python predict.py
```

This allows you to input laptop specifications and get price predictions.

### Step 6: Run the Demo

```bash
# Comprehensive demo of all features
python demo.py
```

## 📊 Available Models

The project includes the following machine learning models:

1. **Linear Regression** - Baseline linear model
2. **Ridge Regression** - Regularized linear regression
3. **Lasso Regression** - Sparse linear regression
4. **Random Forest** - Ensemble tree-based model
5. **Gradient Boosting** - Advanced ensemble model
6. **Support Vector Regression (SVR)** - Kernel-based regression

## 💬 Chatbot Features

The conversational AI interface provides:

1. **Natural Language Understanding** - Understands user preferences in natural language
2. **Interactive Recommendations** - Provides personalized laptop suggestions
3. **Preference Learning** - Remembers and builds on user preferences
4. **Multiple Interfaces** - Command-line and web-based options
5. **Context Awareness** - Maintains conversation context and history

## 🔧 Advanced Usage

### Interactive Analysis

For interactive analysis, you can use the analysis script:

```python
# In Python or Jupyter notebook
from notebooks.analysis_script import run_complete_analysis

# Run complete analysis
results = run_complete_analysis()
```

### Individual Module Usage

You can also use individual modules:

```python
from src.data_loader import DataLoader
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

# Load data
loader = DataLoader()
data = loader.load_data()

# Preprocess
preprocessor = DataPreprocessor()
X_processed, feature_names = preprocessor.create_preprocessing_pipeline(X)

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(X_processed, y)

# Evaluate
evaluator = ModelEvaluator()
report = evaluator.create_evaluation_report(results)
```

## 📈 Model Performance Metrics

The project evaluates models using:

- **R² Score** - Coefficient of determination
- **RMSE** - Root Mean Square Error
- **MAE** - Mean Absolute Error
- **MAPE** - Mean Absolute Percentage Error

## 🎨 Generated Outputs

After running the pipeline, you'll find:

### In `results/` directory:
- `model_comparison.png` - Model performance comparison plots
- `evaluation_report.txt` - Detailed evaluation report
- Individual model evaluation plots

### In `models/` directory:
- Trained model files (`.joblib` format)

### In root directory:
- `training.log` - Training process log

## 🔍 Feature Engineering

The preprocessing pipeline includes:

1. **Data Cleaning** - Handle missing values
2. **Feature Extraction** - Extract numeric values from text (RAM, Storage, etc.)
3. **Categorical Encoding** - Label encoding for categorical variables
4. **Feature Scaling** - Standard scaling for numeric features

## 🛠️ Customization

### Adding New Models

To add a new model, modify `src/model_training.py`:

```python
# In _initialize_models method
self.models['Your Model'] = YourModelClass()
```

### Modifying Preprocessing

To modify preprocessing steps, edit `src/data_preprocessing.py`:

```python
# Add new preprocessing steps in create_preprocessing_pipeline method
```

### Hyperparameter Tuning

The project includes hyperparameter tuning capabilities:

```python
# In model_training.py
best_model = trainer.hyperparameter_tuning('Random Forest', X, y)
```

## 📋 Requirements

- Python 3.8+
- pandas 2.1.4
- scikit-learn 1.3.2
- numpy 1.24.3
- matplotlib 3.8.2
- seaborn 0.13.0
- jupyter 1.0.0

## 🐛 Troubleshooting

### Common Issues:

1. **Dataset not found**: Make sure `Cleaned_Laptop_data.csv` is in the `data/` directory
2. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Memory issues**: Reduce dataset size or use smaller models for large datasets
4. **Plot display issues**: Use `plt.show()` in interactive environments

### Getting Help:

1. Check the `training.log` file for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure the dataset is properly formatted
4. Check Python version compatibility

## 🎓 Learning Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

## 📝 License

This project is for educational purposes. The dataset is sourced from Kaggle.

## 🤝 Contributing

Feel free to:
- Add new models
- Improve preprocessing
- Enhance evaluation metrics
- Add new features

---

**Happy Machine Learning! 🚀** 