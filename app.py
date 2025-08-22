from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Load model and label encoders
try:
    model = joblib.load('model.lb')
    label_encoders = joblib.load('label_encoders.pkl')
    print("✅ Model and encoders loaded successfully")
except Exception as e:
    print(f"❌ Error loading model/encoders: {e}")
    model = None
    label_encoders = None

def encode(data):
    """Encode input data using the same encoders as training"""
    if label_encoders is None:
        raise Exception("Label encoders not loaded")
    
    try:
        # Encode each categorical variable
        job_encoded = label_encoders['job'].transform([data['job']])[0]
        marital_encoded = label_encoders['marital'].transform([data['marital']])[0]
        education_encoded = label_encoders['education'].transform([data['education']])[0]
        housing_encoded = label_encoders['housing'].transform([data['housing']])[0]
        
        return [
            int(data['age']),
            job_encoded,
            marital_encoded,
            education_encoded,
            float(data['balance']),
            housing_encoded,
            int(data['duration']),
            int(data['campaign']),
        ]
    except Exception as e:
        print(f"❌ Encoding error: {e}")
        raise e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Here you would typically save the message to a database or send an email
        # For now, we'll just show a success message
        return render_template('contact.html', success=True)
    
    return render_template('contact.html')

@app.route('/history')
def history():
    try:
        if os.path.exists("history.csv"):
            df = pd.read_csv("history.csv")
            return render_template('history.html', data=df.to_dict(orient="records"))
        else:
            return render_template('history.html', data=[])
    except Exception as e:
        print(f"Error reading history: {e}")
        return render_template('history.html', data=[])

@app.route('/project', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template("project.html", prediction="Model not loaded")
        
        input_data = request.form.to_dict()
        input_encoded = encode(input_data)
        prediction = model.predict([input_encoded])[0]
        result = "Approved" if prediction == 1 else "Not Approved"

        # Save prediction
        df = pd.DataFrame([input_data])
        df["Prediction"] = result
        df.to_csv("history.csv", mode='a', header=not os.path.exists("history.csv"), index=False)

        return render_template("project.html", prediction=result)
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template("project.html", prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
