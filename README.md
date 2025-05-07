# Credit Card Fraud Detection System

A web-based application for detecting fraudulent credit card transactions using machine learning models. The system analyzes transaction data and provides predictions using multiple ML algorithms.

## Features

- Multiple ML model predictions (XGBoost, Random Forest, SVM, Logistic Regression)
- Real-time transaction analysis
- Interactive visualizations
- User authentication system
- Secure data handling
- Statistical analysis of transactions

## Models Used

1. **XGBoost**
   - Gradient boosting framework
   - High accuracy in fraud detection
   - Handles imbalanced data well

2. **Random Forest**
   - Ensemble learning method
   - Robust against overfitting
   - Good for high-dimensional data

3. **Support Vector Machine (SVM)**
   - Linear kernel implementation
   - Effective for binary classification
   - Good for fraud detection patterns

4. **Logistic Regression**
   - Linear model for binary classification
   - Fast and interpretable
   - Good baseline model

## Technical Stack

- **Backend**: Python Flask
- **Database**: MongoDB
- **ML Libraries**: scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML, CSS, JavaScript

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Create a `.env` file
   - Add the following variables:
     ```
     SECRET_KEY=your_secret_key
     MONGODB_URI=your_mongodb_uri
     ```

3. Run the application:
   ```bash
   python app.py
   ```

## Data Requirements

The system expects CSV files with the following features:
- Transaction amount
- Time
- V1-V28 (anonymized features)
- Class (0 for legitimate, 1 for fraudulent)

## Security Features

- Password hashing using bcrypt
- Session management
- File upload restrictions
- Input validation
- Secure MongoDB connection

## Performance

The system uses multiple models to provide comprehensive fraud detection:
- Each model is trained on a subset of data for optimal performance
- Models are evaluated using accuracy metrics
- Results are presented with visual comparisons

## Note

This system is designed for educational and demonstration purposes. Always ensure proper security measures are in place when handling real financial data.
