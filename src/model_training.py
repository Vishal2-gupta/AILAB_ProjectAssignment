from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session with resource config
spark = SparkSession.builder \
    .appName("TitanicTraining") \
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

        # Confusion matrix
        preds_pd = predictions.select("prediction", "Survived").toPandas()
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