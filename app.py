from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and accuracy
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("accuracy.txt", "r") as f:
    accuracy = f.read()

gender_map = {"Male": 0, "Female": 1}
jaundice_map = {"Yes": 1, "No": 0}
autism_map = {"Yes": 1, "No": 0}
prediction_map = {1: "Yes", 0: "No"}

features = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "gender", "jundice", "austim"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        gender = gender_map.get(data.get("gender"))
        jaundice = jaundice_map.get(data.get("jaundice"))
        autism = autism_map.get(data.get("autism"))

        if None in [gender, jaundice, autism]:
            return "Invalid input values"
        
        scores = [
            int(data["A1_Score"]), int(data["A2_Score"]), int(data["A3_Score"]),
            int(data["A4_Score"]), int(data["A5_Score"]), int(data["A6_Score"]),
            int(data["A7_Score"]), int(data["A8_Score"]), int(data["A9_Score"]),
            int(data["A10_Score"])
        ]

        patient_data = pd.DataFrame([[ 
            int(data["A1_Score"]), int(data["A2_Score"]), int(data["A3_Score"]),
            int(data["A4_Score"]), int(data["A5_Score"]), int(data["A6_Score"]),
            int(data["A7_Score"]), int(data["A8_Score"]), int(data["A9_Score"]),
            int(data["A10_Score"]), gender, jaundice, autism
        ]], columns=features)

        prediction_numeric = model.predict(patient_data)[0]
        prediction_text = prediction_map[prediction_numeric]

        total_score = sum(scores)
        average_score = total_score / 10
        intensity = (total_score / 10) * 100

        # Risk Level
        if total_score >= 7:
            risk_level = "High Risk"
        elif total_score >= 4:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        # Categorize questions (for new chart types)
        social_indices = [2, 4, 5, 8, 9]  # Example: A3, A5, A6, A9, A10
        pattern_indices = [0, 1, 6]       # Example: A1, A2, A7
        emotional_indices = [3, 7]        # Example: A4, A8

        score_groups = {
            "Social": sum([scores[i] for i in social_indices]),
            "Pattern": sum([scores[i] for i in pattern_indices]),
            "Emotional": sum([scores[i] for i in emotional_indices]),
        }

        # Confidence score for individual user
        confidence_score = model.predict_proba(patient_data)[0][prediction_numeric] * 100
        confidence_score = f"{confidence_score:.2f}"

        return render_template('result.html',
                               prediction=prediction_text,
                               accuracy=accuracy,
                               confidence=confidence_score,
                               scores=scores,
                               total_score=total_score,
                                average_score=average_score,
                                intensity=intensity,
                                risk_level=risk_level,
                                score_groups=score_groups
                               )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
