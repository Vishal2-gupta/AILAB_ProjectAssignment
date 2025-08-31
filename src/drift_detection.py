from scipy.stats import ks_2samp
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load processed data (handle Spark directory output)
train_data_dir = "data/processed/train_processed.csv"
new_data_dir = "data/processed/test_processed.csv"

# Read all CSV files in the directory
train_data = pd.concat([pd.read_csv(os.path.join(train_data_dir, f)) for f in os.listdir(train_data_dir) if f.endswith('.csv')])
new_data = pd.concat([pd.read_csv(os.path.join(new_data_dir, f)) for f in os.listdir(new_data_dir) if f.endswith('.csv')])

# Select feature for drift detection (e.g., Age)
feature = "Age"
stat, p_value = ks_2samp(train_data[feature], new_data[feature])

logger.info(f"KS-test for {feature}: statistic={stat}, p_value={p_value}")
if p_value < 0.05:
    logger.warning("Drift detected! Triggering retraining...")
    import os
    os.system("python src/automated_retraining.py")
else:
    logger.info("No significant drift detected.")