import pandas as pd
import os
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold


def load_datasets():
    """Load the raw Mendeley and PhiUSIIL datasets from CSV files."""
    mendeley_df = pd.read_csv(r"data\raw\Mendeley Dataset.csv")
    phiusiil_df = pd.read_csv(r"data\raw\PhiUSIIL Dataset.csv")
    return mendeley_df, phiusiil_df


def process_data(df):
    """
    Preprocess the dataset: Check class distribution and missing values and Hash high-cardinality categorical features into sparse vectors
    """
    # Check class imbalance
    if "Label" in df.columns:
        print("\nClass Distribution:")
        print(df["Label"].value_counts(normalize=True))
    else:
        print("\nNo obvious class column found.")

    # Check missing values
    missing = df.isnull().sum().sum()
    print(f"\nTotal Missing Values: {missing}")

    # Hash Categorical Features
    cat_columns = df.select_dtypes(include=['object', 'category']).columns 
    if len(cat_columns) > 0: 
        df = hash_categorical_features(df) 
    else: print("No categorical features to hash") 
    
    print(f"Shape After Processing: {df.shape}")
    return df


def hash_categorical_features(df,):
    """
    Hash high-cardinality categorical features into sparse vectors.
    Uses HashingVectorizer with char or word n-grams depending on column.
    """
    n_features_map = {
        'FILENAME': 512,
        'URL': 8192,
        'Domain': 512,
        'TLD': 64,
        'Title': 1024
    }

    hashed_dfs = []
    columns_to_hash = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
    for col in columns_to_hash:
        #Use char n-grams for short strings (URL, FILENAME, Domain, TLD), word n-grams for Title
        analyzer = 'char_wb' if col in ['FILENAME', 'URL', 'Domain', 'TLD'] else 'word'
        ngram_range = (5,7) if analyzer=='char_wb' else (1,2)

        vectorizer = HashingVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            n_features=n_features_map[col],
            alternate_sign=False
        )

        #Transform column
        X = vectorizer.transform(df[col].astype(str))
        df_hashed = pd.DataFrame.sparse.from_spmatrix(X)
        df_hashed = df_hashed.reset_index(drop=True)
        hashed_dfs.append(df_hashed)

    
    #Preserve numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).copy()

    #Convert numeric columns to sparse to save memory
    for col in numeric_columns.columns:
        numeric_columns[col] = pd.arrays.SparseArray(numeric_columns[col], fill_value=0)

    #Combine 
    df_hashed_combined = pd.concat([numeric_columns.reset_index(drop=True), df_hashed.reset_index(drop=True)], axis=1)

    return df_hashed_combined


def save_processed_data(df, filename):
    """
    Save the processed DataFrame to CSV efficiently.
    Numeric columns are already sparse; hashed columns are sparse by default.
    """
    os.makedirs("data/processed", exist_ok=True)
    save_path = os.path.join("data/processed", filename)
    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")

def read_processed_data(filepath):
   """Read the processed data from sparse CSV files"""
   df = pd.read_csv(filepath)
   y =df["Label"]
   x = df.drop(columns=["Label"], axis=1)
   return x,y
   
def create_train_test_val_sets(x, y, label_col="Label", test_size=0.2, n_splits=5, random_state=42):
    """
    Split a dataset into:
      1. Set (train and validation) for K-Fold CV
      2. test set

    Parameters:
    - x: feature vectors
    - y: label vector
    - label_col: column name of target label
    - test_size: train/test split ratio
    - n_splits: number of folds for Stratified K-Fold CV
    - random_state: for reproducibility

    Returns:
    dict with keys:
    - 'x_train_val', 'y_train_val': dataset for CV
    - 'x_test', 'y_test': test set
    - 'cv_splits': list of (train_idx, val_idx) tuples for K-Fold CV on train_val
    """

    #Hold-out test set
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )

    #Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = [(train_idx, val_idx) for train_idx, val_idx in skf.split(x_train_val, y_train_val)]

    print(f"Train/validation/test split prepared: {len(y_train_val)} instances for training and validation, {len(y_test)} instances for testing")
    print(f"Stratified {n_splits}-fold CV splits created.")

    return {
        "x_train_val": x_train_val, #feature set for cv
        "y_train_val": y_train_val, #label set for cv
        "x_test": x_test,
        "y_test": y_test,
        "cv_splits": cv_splits #A list of tuples (train_idx, val_idx) representing each fold split for cross-validation
    }


def main():
    mendeley_df, phiusiil_df = load_datasets()

    print("----------Processing Mendeley Dataset----------")
    mendeley_processed = process_data(mendeley_df)
    save_processed_data(mendeley_processed, "mendeley_processed.csv")

    print("----------Processing PhisUSIIL Dataset----------")
    phiusiil_processed = process_data(phiusiil_df)
    save_processed_data(phiusiil_processed, "phiusiil_processed.csv")

    #Read Processed Files
    x_mendeley, y_mendeley = read_processed_data(r"data/processed/mendeley_processed.csv")
    x_phiusiil, y_phiusiil= read_processed_data(r"data\processed\phiusiil_processed.csv")

    #Split data
    medeley_sets = create_train_test_val_sets(x_mendeley,y_mendeley, label_col="Label", test_size=0.2, n_splits=5)
    phiusiil_sets = create_train_test_val_sets(x_phiusiil,y_phiusiil, label_col="Label", test_size=0.2, n_splits=5)

    # Accessing the folds (this is just how to access the first fold, you would need to do this in a loop)
    # train_idx, val_idx = medeley_sets["cv_splits"][0] #first fold 
    # x_train, x_val = medeley_sets["x_train_val"].iloc[train_idx], medeley_sets["x_train_val"].iloc[val_idx]
    # y_train, y_val = medeley_sets["y_train_val"].iloc[train_idx], medeley_sets["y_train_val"].iloc[val_idx]



if __name__ == "__main__":
    main()