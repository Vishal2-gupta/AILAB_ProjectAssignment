from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from pydantic import BaseModel
import pandas as pd

# Set the MLflow tracking URI to match the running server
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = FastAPI()

# Load the registered model from MLflow
try:
    model = mlflow.pyfunc.load_model("models:/TitanicModel/Production")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Define the input data structure
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    FamilySize: int
    IsAlone: int
    Title: str

@app.post("/predict")
def predict(passenger: Passenger):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([passenger.model_dump()])  # Use this instead
        # Make prediction
        prediction = model.predict(data)
        return {"survived": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





# from fastapi import FastAPI, HTTPException
# import mlflow.pyfunc
# from pydantic import BaseModel
# import pandas as pd

# # Set the MLflow tracking URI to match the running server
# import mlflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# app = FastAPI()

# # Load the registered model from MLflow
# try:
#     model = mlflow.pyfunc.load_model("models:/TitanicModel/Production")
# except Exception as e:
#     raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# # Define the input data structure for POST
# class Passenger(BaseModel):
#     Pclass: int
#     Sex: str
#     Age: float
#     SibSp: int
#     Parch: int
#     Fare: float
#     Embarked: str
#     FamilySize: int
#     IsAlone: int
#     Title: str

# @app.post("/predict")
# def predict_post(passenger: Passenger):
#     try:
#         data = pd.DataFrame([passenger.dict()])
#         prediction = model.predict(data)
#         return {"survived": int(prediction[0])}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# @app.get("/predict")
# def predict_get():
#     # Hardcoded sample data for GET request
#     sample_data = pd.DataFrame([{
#         "Pclass": 3, "Sex": "male", "Age": 22.0, "SibSp": 1, "Parch": 0,
#         "Fare": 7.25, "Embarked": "S", "FamilySize": 2, "IsAlone": 0, "Title": "Mr"
#     }])
#     prediction = model.predict(sample_data)
#     return {"survived": int(prediction[0])}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)