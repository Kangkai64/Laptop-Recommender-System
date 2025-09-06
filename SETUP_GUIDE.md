# Laptop Recommender System - Setup Guide

## Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning the repository)

### 2. Installation

#### Option A: Using the provided requirements file
```bash
# Install dependencies
pip install -r requirements.txt

# For collaborative filtering specific requirements
pip install -r requirements_collaborative_filtering.txt

# For content-based filtering specific requirements  
pip install -r requirements_content_based_filtering.txt
```

#### Option B: Manual installation
```bash
# Core dependencies
pip install flask pandas numpy scikit-learn

# For advanced features
pip install transformers datasets pyarrow

# For web interface
pip install jinja2 werkzeug
```

### 3. Training Models (First Time Setup)

#### Step 1: Run the Training Pipeline
1. Open `laptop_recommender_training.ipynb` in Jupyter Notebook
2. Run all cells in sequence
3. This will:
   - Load and preprocess the dataset
   - Train content-based and collaborative filtering models
   - Create database mappings
   - Save all models as .pkl files in the `models/` directory

#### Step 2: Verify Model Files
After training, you should have these files in the `models/` directory:
```
models/
├── content_based_model.pkl
├── collaborative_model.pkl
├── database_mappings.pkl
├── database_mappings_laptop_metadata.pkl
├── database_mappings_user_profiles.pkl
├── database_mappings_brand_mapping.pkl
├── database_mappings_category_mapping.pkl
└── model_metadata.pkl
```

### 4. Running the Web Application

#### Start the Flask Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

#### Access the Web Interface
1. Open your web browser
2. Navigate to `http://localhost:5000`
3. Click on "Get Recommendations" to start using the system

### 5. Using the System

#### Web Interface
1. **Get Recommendations**: Fill out the preference form
2. **Choose Algorithm**: Select from Content-Based, Collaborative, or Hybrid
3. **View Results**: Browse personalized recommendations
4. **Explore Laptops**: Click on individual laptops for detailed information

#### API Usage
```python
from model_manager import get_recommendations

# Get recommendations
recommendations = get_recommendations(
    algorithm="content_based",
    top_n=10,
    preferences={
        'budget_range': (2000, 5000),
        'brand_preference': 'Dell'
    }
)
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get import errors, try:
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Model Loading Errors
- Ensure all model files are in the `models/` directory
- Check that the training pipeline completed successfully
- Verify file permissions

#### 3. Memory Issues
- Close other applications to free up RAM
- Reduce batch sizes in the training notebook
- Use a machine with more RAM for training

#### 4. Port Already in Use
```bash
# If port 5000 is busy, change the port in app.py:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Performance Optimization

#### For Training
- Use a machine with 8GB+ RAM
- Close unnecessary applications
- Consider using GPU acceleration for large datasets

#### For Production
- Use a production WSGI server (e.g., Gunicorn)
- Implement proper logging and monitoring
- Use a reverse proxy (e.g., Nginx)

## Development Setup

### For Contributors

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd Laptop-Recommender-System-6
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Development Dependencies
```bash
pip install -r requirements.txt
pip install jupyter notebook  # For development
```

#### 4. Run Tests
```bash
# Run the training notebook to test the pipeline
jupyter notebook laptop_recommender_training.ipynb
```

## Configuration

### Environment Variables
Create a `.env` file for configuration:
```
FLASK_ENV=development
MODELS_DIR=models
DEBUG=True
```

### Model Configuration
Edit the configuration in the training notebook:
- `content_config`: Content-based filtering parameters
- `collaborative_config`: Collaborative filtering parameters
- `split_ratio`: Train/test split ratio

## Deployment

### Production Deployment

#### 1. Prepare for Production
```bash
# Set production environment
export FLASK_ENV=production
export DEBUG=False
```

#### 2. Use Production WSGI Server
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### 3. Use Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Support

### Getting Help
1. Check the troubleshooting section above
2. Review the system logs for error messages
3. Ensure all dependencies are correctly installed
4. Verify that the training pipeline completed successfully

### System Requirements
- **Minimum**: 4GB RAM, 2GB storage, Python 3.8
- **Recommended**: 8GB RAM, 4GB storage, Python 3.9+
- **For Training**: 16GB RAM, 8GB storage, multi-core CPU

### Performance Expectations
- **Model Loading**: 5-10 seconds on first startup
- **Recommendation Generation**: <1 second per request
- **Web Interface**: Responsive on modern browsers
- **Training Time**: 10-30 minutes depending on hardware

---

**Last Updated**: 2024-12-19  
**Version**: 1.0
