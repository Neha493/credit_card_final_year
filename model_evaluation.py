import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
import time

def preprocess_data(X_train, X_test, y_train):
    """
    Preprocess the data using various techniques
    """
    # Scale the features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_balanced, X_test_scaled, y_train_balanced

def get_best_parameters(model, param_dist, X_train, y_train):
    """
    Find the best parameters using RandomizedSearchCV
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=5,  # Further reduced iterations
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=1,  # Disabled parallel processing
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate a model's performance and calculate various metrics
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    processing_time = (time.time() - start_time) * 1000

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Processing Time: {processing_time:.2f} ms")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"False Negative Rate: {false_negative_rate:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'processing_time': processing_time,
        'confusion_matrix': cm
    }

def calculate_ensemble_weights(model_results):
    """
    Calculate weights for ensemble based on individual model performance
    """
    # Calculate total performance score for each model
    total_scores = []
    for result in model_results:
        # Weight accuracy, precision, and recall equally
        score = (result['accuracy'] + result['precision'] + result['recall']) / 3
        total_scores.append(score)
    
    # Normalize scores to get weights
    total = sum(total_scores)
    weights = [score/total for score in total_scores]
    
    return weights

def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocess the data
    X_train_balanced, X_test_scaled, y_train_balanced = preprocess_data(X_train, X_test, y_train)
    
    # Define parameter distributions for each model
    param_dists = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 6],
            'n_estimators': [100, 200]
        },
        'SVM': {
            'C': [1, 10],
            'gamma': ['scale', 'auto']
        },
        'Logistic Regression': {
            'C': [1, 10],
            'solver': ['liblinear']
        }
    }
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    # Evaluate each model with optimized parameters
    model_results = []
    optimized_models = {}
    for name, model in models.items():
        print(f"\nOptimizing {name}...")
        best_params = get_best_parameters(model, param_dists[name], X_train_balanced, y_train_balanced)
        print(f"Best parameters for {name}:", best_params)
        
        # Create model with best parameters
        if name == 'Random Forest':
            optimized_model = RandomForestClassifier(**best_params, random_state=42)
        elif name == 'XGBoost':
            optimized_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif name == 'SVM':
            optimized_model = SVC(**best_params, probability=True, random_state=42)
        else:
            optimized_model = LogisticRegression(**best_params, random_state=42)
        
        optimized_models[name] = optimized_model
        results = evaluate_model(optimized_model, X_train_balanced, X_test_scaled, y_train_balanced, y_test, name)
        model_results.append(results)
    
    # Calculate ensemble weights
    weights = calculate_ensemble_weights(model_results)
    
    # Create ensemble model with calculated weights
    estimators = [(name, model) for name, model in optimized_models.items()]
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights
    )
    
    # Evaluate ensemble model
    print("\nEvaluating Ensemble Model...")
    ensemble_results = evaluate_model(ensemble, X_train_balanced, X_test_scaled, y_train_balanced, y_test, "Ensemble")
    
    # Print ensemble weights
    print("\nEnsemble Weights:")
    for i, (name, weight) in enumerate(zip(models.keys(), weights)):
        print(f"{name}: {weight:.2%}")

if __name__ == "__main__":
    main() 