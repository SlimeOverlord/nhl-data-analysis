import os
import pandas as pd

def load_training_and_test_sets():
    """
    Loads the training and test CSV files generated previously in feature_engineering_1.ipynb.
    Automatically detects the project root (folder 'nhl-data-analysis'),
    so it works regardless of where you run the script from. This addresses the issue of relative paths
    when running .py or .ipynb scripts from different directories.

    Returns:
        train_df (pd.DataFrame): Training set dataframe
        test_df (pd.DataFrame): Test set dataframe
    Raises:
        FileNotFoundError: If the training set file is not found
    """
    current_path = os.path.abspath(__file__)
    project_root = current_path.split("ift6758")[0] + "ift6758"

    data_dir = os.path.join(project_root, "data", "NHLData")
    train_path = os.path.join(data_dir, "training_set.csv")
    test_path = os.path.join(data_dir, "test_set.csv")

    print(f"Searching for data in: {data_dir}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training set file not found: {train_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df
