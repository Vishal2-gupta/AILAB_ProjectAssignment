import requests
import pandas as pd
from pathlib import Path

# ----------------------------
# 1. Test multiple single passengers
# ----------------------------

single_url = "http://localhost:8000/predict"

test_passengers = [
    {
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
    },
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 38.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 71.2833,
        "Embarked": "C",
        "FamilySize": 2,
        "IsAlone": 0,
        "Title": "Mrs"
    },
    {
        "Pclass": 3,
        "Sex": "female",
        "Age": 26.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.925,
        "Embarked": "S",
        "FamilySize": 1,
        "IsAlone": 1,
        "Title": "Miss"
    },
    {
        "Pclass": 1,
        "Sex": "male",
        "Age": 35.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 53.1,
        "Embarked": "S",
        "FamilySize": 1,
        "IsAlone": 1,
        "Title": "Mr"
    },
    {
        "Pclass": 2,
        "Sex": "male",
        "Age": 27.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 13.0,
        "Embarked": "S",
        "FamilySize": 1,
        "IsAlone": 1,
        "Title": "Mr"
    }
]

print("🔹 Single passenger predictions:")
for i, passenger in enumerate(test_passengers, start=1):
    response = requests.post(single_url, json=passenger)
    print(f"Passenger {i}: {response.json()}")


# ----------------------------
# 2. Batch CSV test (top 10 rows)
# ----------------------------
batch_url = "http://localhost:8000/predict_batch"
csv_path = Path("data/raw/test.csv")

if csv_path.exists():
    df = pd.read_csv(csv_path)

    # Take only first 10 rows
    df = df.head(10)

    # Drop irrelevant columns if they exist
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Extract Title from Name if column exists, else default "Mr/Miss"
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).fillna("Mr")
    else:
        df["Title"] = "Mr"

    # Fill missing values
    df = df.fillna({
        "Age": 0.0,
        "Fare": 0.0,
        "Embarked": "S",
        "Sex": "male",
        "Title": "Mr",
        "FamilySize": 0,
        "IsAlone": 1
    })

    passengers = df.to_dict(orient="records")


    response = requests.post(batch_url, json=passengers)

if response.status_code == 200:
    preds = [int(p) for p in response.json()["survived"]]   # ✅ force int
    df["survived"] = preds

    print("\n🔹 Batch predictions (top 10 rows):")
    for i, row in df.iterrows():
        print(
            f"Row {i+1} | Pclass: {row['Pclass']}, Sex: {row['Sex']}, "
            f"Age: {row['Age']} → Survived: {row['survived']}"
        )
else:
    print("Error (batch):", response.text)




    





