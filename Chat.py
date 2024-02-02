import pandas as pd
import gradio as gr
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
loaded_model = joblib.load('decision_tree_model.joblib')
# try:
#     with open('decision_tree_model.joblib', 'rb') as f:
#         loaded_model = pickle.load(f)
# except Exception as e:
#     print(f"Error loading the model: {e}")
#     loaded_model = None

def predict(Fever, Cough, Fatigue, Difficulty_Breathing, Age, Gender, Blood_Pressure, Cholesterol_Level):
    # if loaded_model is None:
    #     return "Error: Model not loaded."

    symptoms = {
        'Fever': Fever,
        'Cough': Cough,
        'Fatigue': Fatigue,
        'Difficulty_Breathing': Difficulty_Breathing,
        'Age': Age,
        'Gender': Gender,
        'Blood_Pressure': Blood_Pressure,
        'Cholesterol_Level': Cholesterol_Level
    }

    symptoms_data = {
        'Fever': [symptoms['Fever']],
        'Cough': [symptoms['Cough']],
        'Fatigue': [symptoms['Fatigue']],
        'Difficulty_Breathing': [symptoms['Difficulty_Breathing']],
        'Age': [symptoms['Age']],
        'Gender': [label_encoder.fit_transform([symptoms['Gender']])[0]],
        'Blood_Pressure': [label_encoder.fit_transform([symptoms['Blood_Pressure']])[0]],
        'Cholesterol_Level': [label_encoder.fit_transform([symptoms['Cholesterol_Level']])[0]]
    }
    loaded_model = joblib.load('decision_tree_model.joblib')
    input_df = pd.DataFrame(symptoms_data)

    prediction = loaded_model.predict(input_df)
    print(prediction)
    return prediction[0]

iface = gr.Interface(
    predict,
    inputs=[
        gr.Radio(choices=["No", "Yes"], label="Fever", type="index"),
        gr.Radio(choices=["No", "Yes"], label="Cough", type="index"),
        gr.Radio(choices=["No", "Yes"], label="Fatigue", type="index"),
        gr.Radio(choices=["No", "Yes"], label="Difficulty Breathing", type="index"),
        gr.Number(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender", type="value"),
        gr.Radio(["Low", "Normal", "High"], label="Blood Pressure", type="value"),
        gr.Radio(["Low", "Normal", "High"], label="Cholesterol Level", type="value"),
    ],
    outputs="textbox",
    title="Symptom Checker",
    description="Enter your symptoms to get a prediction.",
)

iface.launch()
