from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, regexp_extract, lit
from pyspark.sql.types import DoubleType, IntegerType

spark = SparkSession.builder.appName("TitanicPreprocessing").getOrCreate()

# Load data
df_train = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)
df_test = spark.read.csv("data/raw/test.csv", header=True, inferSchema=True)

# Combine for preprocessing (add 'source' column to separate later)
df_train = df_train.withColumn("source", lit("train"))
df_test = df_test.withColumn("source", lit("test"))
df = df_train.unionByName(df_test, allowMissingColumns=True)

# Feature engineering
df = df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
df = df.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))
df = df.withColumn("Title", regexp_extract(col("Name"), r' ([A-Za-z]+)\.', 1))  # Extract title from Name

# Handle missing values
age_mean = df.select(mean(col("Age"))).collect()[0][0]
df = df.na.fill({"Age": age_mean, "Embarked": "S", "Fare": df.select(mean(col("Fare"))).collect()[0][0]})
df = df.withColumn("Cabin", when(col("Cabin").isNull(), "Unknown").otherwise(col("Cabin")))

# Drop unnecessary columns
df = df.drop("Name", "Ticket", "Cabin")

# Save processed data
train_processed = df.filter(col("source") == "train").drop("source")
test_processed = df.filter(col("source") == "test").drop("source", "Survived")
train_processed.write.csv("data/processed/train_processed.csv", header=True, mode="overwrite")
test_processed.write.csv("data/processed/test_processed.csv", header=True, mode="overwrite")
print("Preprocessing completed. Data saved to data/processed/")

spark.stop()