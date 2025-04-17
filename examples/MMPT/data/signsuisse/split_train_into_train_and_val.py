#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys

def main():
    # Fix the random seed for reproducibility.
    np.random.seed(42)
    
    # Load the CSV file.
    input_csv = "./data/signsuisse/metadata_train.csv"
    df = pd.read_csv(input_csv)
    
    # Check if there are enough rows for a 1000 row validation set.
    if len(df) < 1000:
        sys.exit("The dataset does not contain at least 1000 rows for a validation set. Aborting.")
    
    # Shuffle the dataframe using a fixed random state.
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Determine the number of training samples (all rows except the last 1000).
    train_size = len(df_shuffled) - 1000
    
    # Split the data: first part as training, last 1000 rows as validation.
    df_train = df_shuffled.iloc[:train_size]
    df_val = df_shuffled.iloc[train_size:]
    
    # Save the train and validation sets to CSV files.
    df_train.to_csv("./data/signsuisse/metadata_train_train.csv", index=False)
    df_val.to_csv("./data/signsuisse/metadata_train_val.csv", index=False)
    
    print("Successfully split the data into training and a validation set of 1000 rows.")

if __name__ == '__main__':
    main()
