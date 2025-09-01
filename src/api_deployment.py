from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from pydantic import BaseModel
import pandas as pd
import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = FastAPI(title="Titanic Survival Prediction API")

# Load model from MLflow registry
try:
    model = mlflow.pyfunc.load_model("models:/TitanicModel/Production")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Define input schema for a single passenger
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
        # Convert input into DataFrame
        data = pd.DataFrame([passenger.model_dump()])
        # Run prediction
        prediction = model.predict(data)
        return {"survived": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# ✅ New endpoint for batch prediction (multiple passengers at once)
    
@app.post("/predict_batch")
async def predict_batch(passengers: list[Passenger]):
    try:
        data = [p.dict() for p in passengers]
        df = pd.DataFrame(data)
        preds = model.predict(df)

        # Ensure output is always a Python list
        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        return {"survived": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
