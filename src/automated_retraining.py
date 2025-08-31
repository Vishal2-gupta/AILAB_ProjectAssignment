from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
import logging
from pyspark.sql.functions import lit  # Added import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spark = SparkSession.builder.appName("AutomatedRetraining").getOrCreate()

try:
    # Load combined data
    train_data = spark.read.csv("data/processed/train_processed.csv", header=True, inferSchema=True)
    new_data = spark.read.csv("data/processed/test_processed.csv", header=True, inferSchema=True)
    
    # Add Survived column to new_data with nulls to match schema
    new_data = new_data.withColumn("Survived", lit(None).cast("int"))

    # Union the dataframes
    combined_data = train_data.union(new_data)

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
    paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).addGrid(rf.maxDepth, [5, 10]).build()
    evaluator = BinaryClassificationEvaluator(labelCol="Survived")
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

    # Train-test split (use only rows with Survived non-null for training)
    train_df = combined_data.filter(combined_data.Survived.isNotNull())
    val_df = combined_data.filter(combined_data.Survived.isNull()).select(train_df.columns)  # Align schema

    if train_df.count() == 0:
        raise ValueError("No training data with Survived labels available!")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run():
        logger.info("Starting retraining...")
        cv_model = cv.fit(train_df)
        predictions = cv_model.transform(val_df)
        auc = evaluator.evaluate(predictions)
        logger.info(f"New AUC: {auc}")

        best_model = cv_model.bestModel
        mlflow.log_param("numTrees", best_model.stages[-1].getNumTrees)
        mlflow.log_param("maxDepth", best_model.stages[-1].getMaxDepth)
        mlflow.log_metric("auc", auc)

        # Register new model
        mlflow.spark.log_model(best_model, "model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "TitanicModel")
        logger.info(f"New model registered at {model_uri}")

except Exception as e:
    logger.error(f"Retrain failed: {str(e)}")
    raise
finally:
    spark.stop()