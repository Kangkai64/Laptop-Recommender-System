# Contributing to Laptop Recommender System

Thank you for your interest in contributing to the Laptop Recommender System! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Basic knowledge of Machine Learning concepts

### Setup Development Environment
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Laptop-Recommender-System.git
   cd Laptop-Recommender-System
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kuchhbhi/latest-laptop-price-list/data?select=Cleaned_Laptop_data.csv)
5. Place the dataset in the `data/` directory

## 📝 How to Contribute

### 1. Reporting Issues
- Use the GitHub issue tracker
- Provide detailed description of the problem
- Include steps to reproduce the issue
- Add relevant error messages and logs

### 2. Suggesting Features
- Create a new issue with the "enhancement" label
- Describe the feature in detail
- Explain why this feature would be useful
- Provide examples if possible

### 3. Code Contributions

#### Pull Request Process
1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Test your changes thoroughly
4. Commit your changes with descriptive messages:
   ```bash
   git commit -m "Add feature: description of changes"
   ```
5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Create a Pull Request

#### Code Style Guidelines
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused
- Add comments for complex logic

### 4. Documentation
- Update README.md if you add new features
- Add docstrings to new functions
- Update setup instructions if needed

## 🧪 Testing

### Running Tests
```bash
# Run the demo to test all features
python demo.py

# Test the chatbot
python chatbot.py

# Test the web interface
python web_chatbot.py
```

### Testing Checklist
- [ ] ML pipeline works correctly
- [ ] Chatbot responds appropriately
- [ ] Web interface functions properly
- [ ] Predictions are reasonable
- [ ] No errors in console output

## 📊 Project Structure

```
Laptop-Recommender-System/
├── src/                    # Core ML modules
│   ├── data_loader.py     # Data loading utilities
│   ├── data_preprocessing.py # Data preprocessing
│   ├── model_training.py  # Model training
│   ├── model_evaluation.py # Model evaluation
│   └── chatbot_interface.py # Chatbot AI
├── main.py                # Main training script
├── predict.py             # Prediction interface
├── chatbot.py             # Command-line chatbot
├── web_chatbot.py         # Web chatbot interface
├── demo.py                # Comprehensive demo
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```

## 🤝 Areas for Contribution

### High Priority
- [ ] Add more ML models (Neural Networks, XGBoost, etc.)
- [ ] Improve chatbot natural language understanding
- [ ] Add more evaluation metrics
- [ ] Enhance web interface with more features

### Medium Priority
- [ ] Add unit tests
- [ ] Improve documentation
- [ ] Add more visualization options
- [ ] Optimize model performance

### Low Priority
- [ ] Add support for more datasets
- [ ] Create mobile app interface
- [ ] Add multilingual support
- [ ] Implement advanced recommendation algorithms

## 📋 Pull Request Template

When creating a Pull Request, please include:

### Description
Brief description of the changes made.

### Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

### Testing
- [ ] Tested locally
- [ ] All tests pass
- [ ] No breaking changes

### Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No console errors

## 🏷️ Commit Message Guidelines

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for code style changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

Examples:
```
feat: add neural network model
fix: resolve chatbot response error
docs: update setup instructions
```

## 📞 Getting Help

If you need help with contributing:
1. Check existing issues and pull requests
2. Create a new issue with your question
3. Join our discussions in the GitHub community

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to the Laptop Recommender System! 🚀 