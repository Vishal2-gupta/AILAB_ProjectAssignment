from mlflow.tracking import MlflowClient

# Initialize MLflow client with the tracking URI
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

# Transition version 4 to Production stage
client.transition_model_version_stage(
    name="TitanicModel",
    version=7,
    stage="Production"
)
print(f"Transitioned version 7 of TitanicModel to Production")