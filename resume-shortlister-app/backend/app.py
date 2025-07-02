import torch
import joblib
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from model import MultimodalResumeClassifier
from flask_cors import CORS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_encoders = joblib.load('label_encoders.pkl')
scalers = joblib.load('scalers.pkl')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

model = MultimodalResumeClassifier(tabular_input_dim=5)  # Changed back to 5
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.eval()
model.to(DEVICE)

app = Flask(__name__)
# Configure CORS properly
CORS(app, origins=["http://localhost:3000"], 
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type"])

def validate_input(data):
    errors = {}
    required_fields = ['degree', 'field', 'gpa', 'yoe', 'age', 'resume']  # Added age back
    for field in required_fields:
        if field not in data or data[field] in [None, ""]:
            errors[field] = f"{field} is required."
    try:
        float(data.get("gpa", -1))
    except:
        errors["gpa"] = "GPA must be a number."
    try:
        float(data.get("yoe", -1))
    except:
        errors["yoe"] = "Years of experience must be a number."
    try:
        int(data.get("age", -1))
    except:
        errors["age"] = "Age must be a number."
    return errors

def preprocess_input(data):
    # Tokenize resume text
    encoded = tokenizer.encode_plus(
        data['resume'],
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    # Encode categorical features
    degree_encoded = label_encoders['degree'].transform([data['degree']])[0]
    field_encoded = label_encoders['field_of_study'].transform([data['field']])[0]

    # Scale numerical features
    gpa_scaled = scalers['GPA'].transform([[float(data['gpa'])]])[0][0]
    yoe_scaled = scalers['years_of_experience'].transform([[float(data['yoe'])]])[0][0]
    age_scaled = scalers['age'].transform([[float(data['age'])]])[0][0]  # Added age back

    # Combine tabular features (back to 5 features)
    tabular_features = torch.FloatTensor([[degree_encoded, field_encoded, gpa_scaled, yoe_scaled, age_scaled]]).to(DEVICE)

    return input_ids, attention_mask, tabular_features

@app.route('/shortlist', methods=['POST', 'OPTIONS'])
def shortlist():
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 200
        
    data = request.json
    errors = validate_input(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    input_ids, attention_mask, tabular_features = preprocess_input(data)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, tabular_features)
        prob = torch.sigmoid(logits).item()
        shortlisted = prob > 0.5

    return jsonify({"success": True, "shortlisted": shortlisted, "probability": prob})

if __name__ == '__main__':
    app.run(debug=True)