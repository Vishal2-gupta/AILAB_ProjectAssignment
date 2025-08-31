import requests
import json

url = "http://localhost:8000/predict"
sample_passenger = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
    "FamilySize": 2,
    "IsAlone": 0,
    "Title": "Mr"
}

response = requests.post(url, json=sample_passenger)
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")