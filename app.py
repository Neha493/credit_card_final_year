import pandas as pd
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
import bcrypt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
import logging
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
import xgboost as xgb
import requests
import gdown

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# for security purpose
app.secret_key = os.environ.get('SECRET_KEY', b'_5#y2L"F4Q8z\n\xec]/')
        
# Define the directory where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions and max file size (2MB)
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize MongoDB connection variables
db = None
collection = None

logger.info("Loading environment variables...")
load_dotenv()

# Get MongoDB URI from environment variable
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

# Try to connect to MongoDB, but continue without it if connection fails
try:
    logger.info("Attempting to connect to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # 5 second timeout
    client.server_info()  # will raise an exception if connection fails
    db = client['user_database']
    collection = db['users']
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.warning(f"Could not connect to MongoDB: {str(e)}")
    logger.warning("Continuing without database functionality")
    db = None
    collection = None

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download dataset if it doesn't exist
TRAINING_DATA_PATH = os.path.join('data', 'creditcard.csv')
if not os.path.exists(TRAINING_DATA_PATH):
    logger.info("Dataset not found. Downloading...")
    try:
        # Google Drive file ID (replace with your file ID)
        file_id = '1GNxFy8jlTZQLny81XoaYOqfQDQWNFQgh'
        output = TRAINING_DATA_PATH
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
        logger.info("Dataset downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        logger.error("Please manually download the dataset and place it in the data/ directory")

# Load and train models at startup
logger.info("Starting model training process...")
pretrained_models = {}
training_statistical_analysis = None
training_fraudulent_count = 0
training_non_fraudulent_count = 0

if os.path.exists(TRAINING_DATA_PATH):
    logger.info(f"Found training data at {TRAINING_DATA_PATH}")
    try:
        logger.info("Loading training data...")
        training_data = pd.read_csv(TRAINING_DATA_PATH)
        logger.info("Calculating statistics...")
        training_statistical_analysis = training_data.describe()
        training_fraudulent_count = (training_data['Class'] == 1).sum()
        training_non_fraudulent_count = (training_data['Class'] == 0).sum()
        
        logger.info("Preparing training data...")
        # Use a smaller subset of data for training (10% of the data)
        training_data = training_data.sample(frac=0.1, random_state=42)
        X_train = training_data.drop(columns=["Class"])
        y_train = training_data["Class"]
        
        # Scale the features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        logger.info("Training XGBoost model...")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        
        logger.info("Training Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        logger.info("Training SVM model...")
        # Use a smaller subset for SVM training (1% of the data)
        svm_data = training_data.sample(frac=0.01, random_state=42)
        X_svm = svm_data.drop(columns=["Class"])
        y_svm = svm_data["Class"]
        X_svm_scaled = scaler.transform(X_svm)  # Use the same scaler
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_svm_scaled, y_svm)
        
        logger.info("Training Logistic Regression model...")
        logistic_model = LogisticRegression(random_state=42, max_iter=1000)
        logistic_model.fit(X_train_scaled, y_train)
        
        pretrained_models = {
            'xgboost': xgb_model,
            'random_forest': rf_model,
            'svm': svm_model,
            'logistic': logistic_model,
            'scaler': scaler  # Store the scaler for later use
        }
        logger.info("All models trained successfully")
        # --- Calculate accuracy for each model ---
        # XGBoost
        xgb_preds = xgb_model.predict(X_train_scaled)
        xgb_acc = round(accuracy_score(y_train, xgb_preds) * 100, 2)
        # Random Forest
        rf_preds = rf_model.predict(X_train_scaled)
        rf_acc = round(accuracy_score(y_train, rf_preds) * 100, 2)
        # SVM
        svm_preds = svm_model.predict(X_svm_scaled)
        svm_acc = round(accuracy_score(y_svm, svm_preds) * 100, 2)
        # Logistic Regression
        logistic_preds = logistic_model.predict(X_train_scaled)
        logistic_acc = round(accuracy_score(y_train, logistic_preds) * 100, 2)
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        pretrained_models = None
        xgb_acc = rf_acc = svm_acc = logistic_acc = None
else:
    logger.error(f"Training data not found at {TRAINING_DATA_PATH}")
    pretrained_models = None
    xgb_acc = rf_acc = svm_acc = logistic_acc = None

# Set the style for all plots
plt.style.use('dark_background')
sns.set_style("darkgrid")

def generate_model_comparison_plot(result):
    """Generate a bar plot comparing all models' predictions with improved colors and readability."""
    plt.figure(figsize=(8, 5))
    models = list(result.keys())
    fraudulent = [result[model]['Fraudulent'] for model in models]
    non_fraudulent = [result[model]['Non-Fraudulent'] for model in models]
    x = range(len(models))
    width = 0.35
    # Consistent, clear colors
    fraud_color = '#FF6F61'  # Red/pink for Fraudulent
    non_fraud_color = '#4CAF50'  # Green for Non-Fraudulent
    bars1 = plt.bar([i - width/2 for i in x], fraudulent, width, label='Fraudulent', color=fraud_color)
    bars2 = plt.bar([i + width/2 for i in x], non_fraudulent, width, label='Non-Fraudulent', color=non_fraud_color)
    plt.xlabel('Models', color='white', fontsize=12, weight='bold')
    plt.ylabel('Number of Transactions', color='white', fontsize=12, weight='bold')
    plt.title('Model Comparison', color='white', fontsize=15, weight='bold')
    plt.xticks(x, models, rotation=20, color='white', fontsize=12, weight='bold')
    plt.yticks(color='white', fontsize=12, weight='bold')
    # Add value labels on bars
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(int(bar.get_height())),
                 ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(int(bar.get_height())),
                 ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
    # Legend outside right
    legend = plt.legend(
        facecolor='white', edgecolor='black', fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5)
    )
    for text in legend.get_texts():
        text.set_color('black')
    plt.gca().set_facecolor('#232b3e')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#232b3e')
    buf.seek(0)
    plt.close()
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_data

def generate_pie_chart(fraudulent, non_fraudulent, title):
    """Generate a pie chart for a single model's predictions with bar-graph-matching colors, no legend, and larger size."""
    plt.figure(figsize=(5.5, 5.5))
    labels = ['Fraudulent', 'Non-Fraudulent']
    sizes = [fraudulent, non_fraudulent]
    colors = ['#FF6F61', '#4CAF50']  # Red/pink and green (match bar chart)
    explode = (0.08, 0)  # Slightly explode the fraudulent slice
    wedges, texts, autotexts = plt.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, explode=explode, textprops={'color': 'white', 'fontsize': 16, 'weight': 'bold'}
    )
    plt.setp(autotexts, size=18, weight='bold')
    plt.title(title, color='white', fontsize=18, weight='bold', pad=24)
    # Remove legend and model name from inside the chart
    # (No plt.legend call, and title is only above the chart)
    plt.gca().set_facecolor('#232b3e')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#232b3e')
    buf.seek(0)
    plt.close()
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_data

def generate_statistical_plot(statistical_analysis):
    """Generate a clean, readable heatmap of the statistical analysis (mean, std, min, max only)."""
    plt.figure(figsize=(8, 3))
    # Only show mean, std, min, max for clarity
    if isinstance(statistical_analysis, pd.DataFrame):
        stats_to_show = ['mean', 'std', 'min', 'max']
        stat_df = statistical_analysis.loc[statistical_analysis.index.intersection(stats_to_show)]
    else:
        stat_df = pd.DataFrame(statistical_analysis)
    sns.heatmap(stat_df, annot=True, cmap='coolwarm', fmt='.2f',
                annot_kws={"size":10, "color":"black", "weight":"bold"},
                cbar=True, linewidths=0.5, linecolor='#232b3e',
                xticklabels=True, yticklabels=True)
    plt.title('Statistical Analysis (Key Metrics)', color='white', fontsize=13, weight='bold')
    plt.xticks(rotation=30, color='white', fontsize=10)
    plt.yticks(color='white', fontsize=11)
    plt.gca().set_facecolor('#232b3e')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#232b3e')
    buf.seek(0)
    plt.close()
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_data

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if db is None:
        flash("Database is not available. Please try again later.")
        return redirect(url_for('register_page'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        if collection.find_one({'username': username}):
            session['error'] = "Username already exists!"
            return redirect(url_for('register_page'))
        elif collection.find_one({'email': email}):
            session['error'] = "Email already exists!"
            return redirect(url_for('register_page'))

        user_data = {
            'username': username,
            'password': hashed_password,
            'email': email
        }
        collection.insert_one(user_data)

        return redirect(url_for('login_page'))

    return render_template("login.html")

@app.route('/login')
def login_page():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    if db is None:
        flash("Database is not available. Please try again later.")
        return redirect(url_for('login_page'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = collection.find_one({'username': username})

        if user and 'password' in user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['username'] = user['username']
            return redirect(url_for('dashboard', username=user['username']))
        else:
            session['error'] = "Invalid username or password"
            return redirect(url_for('login_page'))

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        username = session['username']
        return render_template("dashboard.html", username=username)
    else:
        return redirect(url_for('login_page'))
    
@app.route('/home')
def home():
    if 'username' in session:
        username = session['username']
        return render_template("home.html", username=username)
    else:
        return redirect(url_for('login_page'))
    
@app.route('/admin')
def admin_page():
    if 'username' in session:
        username = session['username']
        return render_template("admin.html", username=username)
    else:
        return redirect(url_for('login_page'))

@app.route('/profile')
def profile_page():
    if db is None:
        flash("Database is not available. Please try again later.")
        return redirect(url_for('login_page'))
        
    if 'username' in session:
        username = session['username']
        # Fetch user data from the database
        user_data = collection.find_one({'username': username})
        if user_data:
            # Pass user data to the template
            return render_template("profile.html", username=username, user_data=user_data)
        else:
            return "User not found in database"
    else:
        return redirect(url_for('login_page'))

@app.route('/profile/update', methods=['POST'])
def update_profile():
    if db is None:
        flash("Database is not available. Please try again later.")
        return redirect(url_for('profile_page'))
        
    if 'username' in session:
        username = session['username']
        # Fetch user data from the form
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        organization_name = request.form['organization_name']
        location = request.form['location']
        phone_number = request.form['phone_number']
        birthday = request.form['birthday']
        
        # Update user data in the database
        collection.update_one({'username': username}, {'$set': {
            'first_name': first_name,
            'last_name': last_name,
            'organization_name': organization_name,
            'location': location,
            'phone_number': phone_number,
            'birthday': birthday
        }})
        
        # Redirect to profile page
        return redirect(url_for('profile_page'))
    else:
        return redirect(url_for('login_page'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Starting prediction process...")
        if 'file' not in request.files:
            logger.error("No file part in request")
            flash("No file part in request", "danger")
            return redirect(url_for('admin_page'))
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            flash("No selected file", "danger")
            return redirect(url_for('admin_page'))
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            flash("Invalid file type. Only CSV files are allowed.", "danger")
            return redirect(url_for('admin_page'))
        # Check file size
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)
        if file_length > MAX_CONTENT_LENGTH:
            logger.error(f"File too large: {file_length} bytes")
            flash("File is too large. Maximum allowed size is 2MB.", "danger")
            return redirect(url_for('admin_page'))
        logger.info("Reading CSV file...")
        try:
            # Read CSV with error handling for bad lines
            data = pd.read_csv(file, on_bad_lines='skip')
            logger.info(f"Successfully read CSV file with {len(data)} rows")
            # Validate numerical columns
            required_columns = list(training_data.columns.drop('Class'))
            for col in required_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.dropna()
            if len(data) == 0:
                logger.error("No valid numerical data found in the CSV file")
                flash("The CSV file must contain valid numerical data in all required columns.", "danger")
                return redirect(url_for('admin_page'))
            logger.info(f"After cleaning, {len(data)} valid rows remain")
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            flash(f"Error reading CSV file: {str(e)}", "danger")
            return redirect(url_for('admin_page'))
        if pretrained_models is None:
            logger.error("Models are not trained")
            flash("Models are not trained. Please contact the administrator.", "danger")
            return redirect(url_for('admin_page'))
        # Validate that the uploaded data has the required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            flash(f"Uploaded CSV must contain columns: {', '.join(required_columns)}", "danger")
            return redirect(url_for('admin_page'))
        logger.info("Preparing data for prediction...")
        X = data[required_columns]
        # Scale the features using the same scaler used during training
        X_scaled = pretrained_models['scaler'].transform(X)
        # Use pre-trained models for prediction
        logger.info("Making predictions with XGBoost...")
        xgb_model = pretrained_models['xgboost']
        xgb_predictions = xgb_model.predict(X_scaled)
        logger.info("Making predictions with Random Forest...")
        rf_model = pretrained_models['random_forest']
        rf_predictions = rf_model.predict(X_scaled)
        logger.info("Making predictions with SVM...")
        svm_model = pretrained_models['svm']
        svm_predictions = svm_model.predict(X_scaled)
        logger.info("Making predictions with Logistic Regression...")
        logistic_model = pretrained_models['logistic']
        logistic_predictions = logistic_model.predict(X_scaled)
        # Prepare results
        logger.info("Preparing results...")
        result = {
            'XGBoost': {
                'Fraudulent': int((xgb_predictions == 1).sum()),
                'Non-Fraudulent': int((xgb_predictions == 0).sum())
            },
            'Random Forest': {
                'Fraudulent': int((rf_predictions == 1).sum()),
                'Non-Fraudulent': int((rf_predictions == 0).sum())
            },
            'SVM': {
                'Fraudulent': int((svm_predictions == 1).sum()),
                'Non-Fraudulent': int((svm_predictions == 0).sum())
            },
            'Logistic Regression': {
                'Fraudulent': int((logistic_predictions == 1).sum()),
                'Non-Fraudulent': int((logistic_predictions == 0).sum())
            }
        }
        # --- Uploaded Data Statistics ---
        # Only show a few key features for the heatmap
        key_features = [col for col in ['Amount', 'V1', 'V2', 'V3', 'V4', 'V5'] if col in data.columns]
        uploaded_stats = data[key_features].describe().loc[['mean', 'std', 'min', 'max']]
        uploaded_stat_plot = generate_statistical_plot(uploaded_stats)
        # Generate plots for uploaded data
        model_comparison_plot = generate_model_comparison_plot(result)
        xgb_plot = generate_pie_chart(
            result['XGBoost']['Fraudulent'],
            result['XGBoost']['Non-Fraudulent'],
            'XGBoost Predictions'
        )
        rf_plot = generate_pie_chart(
            result['Random Forest']['Fraudulent'],
            result['Random Forest']['Non-Fraudulent'],
            'Random Forest Predictions'
        )
        svm_plot = generate_pie_chart(
            result['SVM']['Fraudulent'],
            result['SVM']['Non-Fraudulent'],
            'SVM Predictions'
        )
        logistic_plot = generate_pie_chart(
            result['Logistic Regression']['Fraudulent'],
            result['Logistic Regression']['Non-Fraudulent'],
            'Logistic Regression Predictions'
        )
        logger.info("Rendering template with results...")
        return render_template('admin.html',
            username=session.get('username', 'Guest'),
            result=result,
            # Uploaded data stats/plots
            uploaded_stat_plot=uploaded_stat_plot,
            # Training data stats for collapsible card
            training_statistical_analysis=training_statistical_analysis,
            fraudulent_count=training_fraudulent_count,
            non_fraudulent_count=training_non_fraudulent_count,
            model_comparison_plot=model_comparison_plot,
            xgb_plot=xgb_plot,
            rf_plot=rf_plot,
            svm_plot=svm_plot,
            logistic_plot=logistic_plot,
            xgb_acc=xgb_acc,
            rf_acc=rf_acc,
            svm_acc=svm_acc,
            logistic_acc=logistic_acc
        )
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}", exc_info=True)
        flash(f"An unexpected error occurred: {str(e)}", "danger")
        return redirect(url_for('admin_page'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)