from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
import numpy as np
import pandas as pd
import joblib
import json




app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load the trained model and scaler
xgb_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Pregnancy-specific normal ranges
pregnancy_normal_ranges = {
    "Age": (18, 35),
    "Body Temperature(F)": (97, 99),
    "Heart rate(bpm)": (70, 110),
    "Systolic Blood Pressure(mm Hg)": (65, 140),
    "Diastolic Blood Pressure(mm Hg)": (70, 80),
    "BMI(kg/m 2)": (18.5, 24.9),
    "Blood Glucose(HbA1c)": (0, 42),
    "Blood Glucose(Fasting hour-mg/dl)": (3.3, 5.1),
}

# Risk level mapping
def risk_level(outcome):
    levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    return levels.get(outcome, "Unknown Risk")

def explain_risk_factors(user_input):
    explanations = []
    for feature, value in user_input.items():
        if feature in pregnancy_normal_ranges:
            low, high = pregnancy_normal_ranges[feature]
            if value < low:
                explanations.append(f"Low {feature} ({value})")
            elif value > high:
                explanations.append(f"High {feature} ({value})")
    return explanations

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

# models.py or inside app.py
class Tracker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    input_data = db.Column(db.String(500))
    prediction_result = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('trackers', lazy=True))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/information')
def information():
    return render_template('information.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if not current_user.is_authenticated:
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']
            user = User.query.filter_by(email=email).first()
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('profile'))
            else:
                flash('Invalid credentials', 'danger')
        return render_template('profile.html', logged_in=False)

    return render_template('profile.html', logged_in=True, name=current_user.name, email=current_user.email)


@app.route('/tracker')
@login_required
def tracker():
    entries = Tracker.query.filter_by(user_id=current_user.id).order_by(Tracker.timestamp.desc()).all()
    return render_template('tracker.html', entries=entries)


@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template('prediction.html', result=None, explanations=None)

@app.route('/submit', methods=['POST'])
@login_required
def submit():
    try:
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        body_temp = float(request.form['temperature'])
        heart_rate = float(request.form['heart-rate'])
        systolic_bp = float(request.form['systolic'])
        diastolic_bp = float(request.form['diastolic'])
        blood_glucose_hba1c = float(request.form['hba1c'])
        blood_glucose_fasting = float(request.form['fasting-glucose'])

        user_input = {
            "Age": age,
            "Body Temperature(F) ": body_temp,
            "Heart rate(bpm)": heart_rate,
            "Systolic Blood Pressure(mm Hg)": systolic_bp,
            "Diastolic Blood Pressure(mm Hg)": diastolic_bp,
            "BMI(kg/m 2)": bmi,
            "Blood Glucose(HbA1c)": blood_glucose_hba1c,
            "Blood Glucose(Fasting hour-mg/dl)": blood_glucose_fasting
        }

        feature_order = [
            "Age", "Body Temperature(F) ", "Heart rate(bpm)",
            "Systolic Blood Pressure(mm Hg)", "Diastolic Blood Pressure(mm Hg)",
            "BMI(kg/m 2)", "Blood Glucose(HbA1c)", "Blood Glucose(Fasting hour-mg/dl)"
        ]

        input_array = [[user_input[feature] for feature in feature_order]]
        input_scaled = scaler.transform(input_array)
        input_df = pd.DataFrame(input_scaled, columns=feature_order)

        predicted_outcome = int(xgb_model.predict(input_df))
        risk = risk_level(predicted_outcome)
        explanations = explain_risk_factors(user_input)

        result = {
            "prediction": risk,
            "explanations": explanations
        }

        # 📝 Tracker log
        # 📝 Tracker log
        if current_user.is_authenticated:
            input_summary = json.dumps(user_input)
            print(f"Input Data: {input_summary}") 
            new_entry = Tracker(
                user_id=current_user.id,
                input_data=json.dumps(user_input),  # ✅ Proper JSON storage
                prediction_result=risk,
                timestamp=datetime.utcnow()
            )
            db.session.add(new_entry)
            db.session.commit()


        return render_template('prediction.html', result=result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.template_filter('from_json')
def from_json_filter(s):
    import json
    try:
        return json.loads(s)  # Converts JSON string to Python dictionary
    except Exception:
        return {}  # Return empty dictionary in case of error



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(name=name, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('profile'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


