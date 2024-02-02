import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import gradio as gr
# model = joblib.load("my_model.pkl")
label_encoder = LabelEncoder()


# with open('decision_tree_model.joblib', 'rb') as f:
#     model_contents = pickle.load(f)

# print(model_contents)
def predict(Fever,Cough,Fatigue,Difficulty_Breathing,Age,Gender,Blood_Pressure,Cholesterol_Level):
    # data = pd.read_csv(csv_file)
    # Perform any necessary preprocessing on the data
    
    symptoms = {
    'Fever': Fever,  # 1 for Yes, 0 for No
    'Cough': Cough,
    'Fatigue': Fatigue,
    'Difficulty_Breathing': Difficulty_Breathing,
    'Age': Age,
    'Gender': Gender,  # 'Male' or 'Female'
    'Blood_Pressure': Blood_Pressure,  # 'Low', 'Normal', 'High'
    'Cholesterol_Level': Cholesterol_Level  # 'Low', 'Normal', 'High'
    }
    print(symptoms)
    symptoms_data = {
        'Fever': [symptoms['Fever']],
        'Cough': [symptoms['Cough']],
        'Fatigue': [symptoms['Fatigue']],
        'Difficulty Breathing': [symptoms['Difficulty_Breathing']],
        'Age': [symptoms['Age']],
        'Gender': [label_encoder.fit_transform([symptoms['Gender']])[0]],
        'Blood Pressure': [label_encoder.fit_transform([symptoms['Blood_Pressure']])[0]],
        'Cholesterol Level': [label_encoder.fit_transform([symptoms['Cholesterol_Level']])[0]]
    }
    input_df = pd.DataFrame(symptoms_data)

    # Load the trained model and make predictions
    loaded_model = joblib.load('decision_tree_model.joblib')
    prediction = loaded_model.predict(input_df)
    return prediction[0]

iface = gr.Interface(
    predict,
    inputs=[
        gr.Radio(
           choices= ["No", "Yes"], label="Fever", type="index",
        ),
        gr.Radio(
             choices= ["No", "Yes"], label="Cough", type="index", 
        ),
        gr.Radio(
             choices= ["No", "Yes"], label="Fatigue", type="index",  
        ),
        gr.Radio(
             choices= ["No", "Yes"],
            label="Difficulty Breathing",
            type="index",
        
            
        ),
        gr.Number(label="Age"),
        gr.Radio(
            ["Male", "Female"], label="Gender", type="value", 
        ),
        gr.Radio(
            ["low", "Normal", "High"],
            label="Blood Pressure",
            type="value",
            
        ),
        gr.Radio(
            ["low", "Normal", "High"],
            label="Cholesterol Level",
            type="value",
            
        ),
    ],
    outputs="textbox",
    title="Symptom Checker 🩺",
    description="Enter your symptoms to get a prediction.",
)

iface.launch()