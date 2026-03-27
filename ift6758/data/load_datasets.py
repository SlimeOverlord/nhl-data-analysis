import os
import pandas as pd

def load_training_and_test_sets():
    """
    Loads the training and test CSV files generated previously in get_train_and_test_sets_demo.ipynb.
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

def load_feature_engineered2_training_and_test_sets():
    """
    Loads the feature-engineered2 training and test CSV files.
    """
    current_path = os.path.abspath(__file__)
    project_root = current_path.split("ift6758")[0] + "ift6758"

    data_dir = os.path.join(project_root, "data", "NHLData")
    train_path = os.path.join(data_dir, "feature_engineered_training_set_2.csv")
    test_path = os.path.join(data_dir, "feature_engineered_test_set_2.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Feature-engineered training set file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Feature-engineered test set file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df
