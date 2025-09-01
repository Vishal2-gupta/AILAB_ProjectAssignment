from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session with resource config
spark = SparkSession.builder \
    .appName("CH24M584_TitanicTraining") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

try:
    # Load processed data
    logger.info("Loading processed data...")
    df = spark.read.csv("data/processed/train_processed.csv", header=True, inferSchema=True)
    if df.count() == 0:
        raise ValueError("Processed data is empty!")

    # Prepare features
    categorical_cols = ["Sex", "Embarked", "Title"]
    numerical_cols = ["Age", "Fare", "Pclass", "FamilySize", "IsAlone"]

    stages = []
    for cat_col in categorical_cols:
        indexer = StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_index", handleInvalid="keep")
        encoder = OneHotEncoder(inputCol=f"{cat_col}_index", outputCol=f"{cat_col}_vec")
        stages += [indexer, encoder]

    assembler = VectorAssembler(inputCols=numerical_cols + [f"{cat_col}_vec" for cat_col in categorical_cols], outputCol="features")
    stages.append(assembler)

    rf = RandomForestClassifier(labelCol="Survived", featuresCol="features")
    stages.append(rf)

    pipeline = Pipeline(stages=stages)

    # Hyperparam tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    evaluator = BinaryClassificationEvaluator(labelCol="Survived")
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

    # Train-test split
    train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)  # Added seed for reproducibility

    # Start MLflow run for experiment tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ensure `mlflow ui` is running
    with mlflow.start_run():
        logger.info("Starting model training...")
        cv_model = cv.fit(train_df)
        predictions = cv_model.transform(val_df)
        auc = evaluator.evaluate(predictions)
        print(f"AUC: {auc}")

        # Log parameters and metrics
        best_model = cv_model.bestModel
        mlflow.log_param("numTrees", best_model.stages[-1].getNumTrees)
        mlflow.log_param("maxDepth", best_model.stages[-1].getMaxDepth)
        mlflow.log_metric("auc", auc)

        # Extra metrics (convert Spark DataFrame to Pandas for sklearn metrics)
        preds_pd = predictions.select("prediction", "Survived").toPandas()
        y_true = preds_pd["Survived"].to_numpy()
        y_pred = preds_pd["prediction"].to_numpy()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Confusion matrix
        cm = confusion_matrix(preds_pd["Survived"], preds_pd["prediction"])
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1], ["Not Survived", "Survived"])
        plt.yticks([0, 1], ["Not Survived", "Survived"])
        plt.colorbar()
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig("reports/figures/confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("reports/figures/confusion_matrix.png")

        # Feature importance plot
        best_rf_model = cv_model.bestModel.stages[-1]  # RandomForestClassifier
        importances = best_rf_model.featureImportances.toArray()

        # Use indices as feature names since exact names are hard to map
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances, align='center')
        plt.xticks(range(len(importances)), feature_names, rotation=45, ha='right')
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance in Random Forest")
        plt.tight_layout()
        plt.savefig("reports/figures/feature_importance.png")
        plt.close()
        mlflow.log_artifact("reports/figures/feature_importance.png")
        # Save model locally (backup)
        os.makedirs("models", exist_ok=True)
        best_model.write().overwrite().save("models/best_rf_model")

        # Log and register model
        mlflow.spark.log_model(best_model, "model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "TitanicModel")
        logger.info(f"Model registered at {model_uri}")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    # Stop Spark session
    logger.info("Stopping Spark session...")
    spark.stop()