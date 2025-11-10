import wandb
import pandas as pd
import numpy as np
import os
from ift6758.data.data_cleaning import feature_engineering_2

def main():
    dir = os.getcwd()
    nhl_data_dir = os.path.join(dir, "NHLData")

    training_set = pd.read_csv(os.path.join(nhl_data_dir, "training_set.csv"))
    wpg_v_wsh_df = training_set[training_set["game_id"] == 2017021065]

    feature_engineered_training_set = feature_engineering_2(wpg_v_wsh_df)

    run = wandb.init(project="ift6758-milestone2", entity="IFT6758-2025-A09", name="dataset_snippet_upload")

    artifact = wandb.Artifact("wpg_v_wsh_2017021065", type="dataset")  
    my_table = wandb.Table(dataframe=feature_engineered_training_set)
    wandb.log({"Feature Engineered Dataset Snippet": my_table})

    artifact.add(my_table, "wpg_v_wsh_2017021065") 
    run.log_artifact(artifact) 

if __name__ == "__main__":
    main()
