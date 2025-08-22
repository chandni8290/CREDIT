import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the credit data"""
    try:
        # Try to load the CSV file with a relative path
        csv_path = "credit.csv"
        if not os.path.exists(csv_path):
            # Try alternative names
            alternative_paths = ["credit .csv", "credit_data.csv", "data.csv"]
            for path in alternative_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            else:
                raise FileNotFoundError("Credit data CSV file not found. Please ensure 'credit.csv' exists in the current directory.")
        
        df = pd.read_csv(csv_path)
        print(f"✅ Data loaded successfully from {csv_path}")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("⚠️  Missing values found:")
            print(missing_values[missing_values > 0])
            # Fill missing values with mode for categorical and median for numerical
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def encode_categorical_data(df):
    """Encode categorical variables using LabelEncoder for consistency"""
    # Create label encoders for each categorical column
    label_encoders = {}
    
    categorical_columns = ['job', 'marital', 'education', 'housing']
    
    for col in categorical_columns:
        le = LabelEncoder()
        # Handle missing values by filling with mode
        df[col] = df[col].fillna(df[col].mode()[0])
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        
        print(f"📊 {col} encoding:")
        for i, category in enumerate(le.classes_):
            print(f"  {category} -> {i}")
    
    # Save the label encoders for use in Flask app
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("✅ Label encoders saved as 'label_encoders.pkl'")
    
    return df, label_encoders

def train_model():
    """Train the credit approval model"""
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        if df is None:
            return False
        
        # Encode categorical data
        df, label_encoders = encode_categorical_data(df)
        
        # Select features
        feature_columns = ['age', 'job_encoded', 'marital_encoded', 'education_encoded', 
                          'balance', 'housing_encoded', 'duration', 'campaign']
        
        X = df[feature_columns]
        y = df['approval']
        
        # Convert approval to binary (yes=1, no=0)
        y = (y == 'yes').astype(int)
        
        print(f"\n📊 Target distribution:")
        print(f"Approved (1): {sum(y == 1)}")
        print(f"Not Approved (0): {sum(y == 0)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model
        joblib.dump(model, 'model.lb')
        print(f"\n✅ Model trained and saved as 'model.lb'")
        print(f"📊 Training set size: {len(X_train)}")
        print(f"📊 Test set size: {len(X_test)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Credit Approval Model Training...")
    success = train_model()
    
    if success:
        print("\n🎉 Model training completed successfully!")
    else:
        print("\n💥 Model training failed!")
