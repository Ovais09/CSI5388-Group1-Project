import pandas as pd
import os
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.sparse import hstack


def load_dataset(file_path):
    """Load the raw Mendeley and PhiUSIIL datasets from CSV files."""
    return  pd.read_csv(file_path)


def process_data(df):
    """
    Preprocess the dataset: Check class distribution and missing values and Hash high-cardinality categorical features into sparse vectors
    """
    # Check class imbalance
    if "Label" in df.columns:
        print("\nClass Distribution:")
        print(df["Label"].value_counts(normalize=True))
        print(df["Label"].dtype)
        if df["Label"].dtype == 'object' or df["Label"].dtype == 'string':
            df["Label"] = df["Label"].map({"legitimate": 0, "phishing": 1})
    else:
        print("\nNo obvious class column found.")

    # Check missing values
    missing = df.isnull().sum().sum()
    print(f"\nTotal Missing Values: {missing}")

    # Hash Categorical Features
    cat_columns = df.select_dtypes(include=['object', 'category']).columns 
    if len(cat_columns) > 0: 
        df = hash_categorical_features(df) 
        #print(df.columns)
    else: print("No categorical features to hash") 
    
    print(f"Shape After Processing: {df.shape}")
    print('Label' in df.columns)
    return df


def hash_categorical_features(df,):
    """
    Hash high-cardinality categorical features into sparse vectors.
    Uses HashingVectorizer with char or word n-grams depending on column.
    """
    n_features_map = {
        'url': 32768,
    }


    hashed_columns = []
    columns_to_hash = ['url']
    for col in columns_to_hash:
        #Use char n-grams for short strings (URL, FILENAME, Domain, TLD), word n-grams for Title
        analyzer = 'char_wb' 
        ngram_range = (3,5)

        vectorizer = HashingVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            n_features=n_features_map[col],
            alternate_sign=False
        )

        #Transform column
        X = vectorizer.transform(df[col].astype(str))
        hashed_columns.append(X) #save the hashed column 
        #df_hashed = df_hashed.reset_index(drop=True)
        #hashed_dfs.append(df_hashed)

    
    #Preserve numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).astype('float32').copy()

    #Convert numeric columns to sparse to save memory
    for col in numeric_columns.columns:
        numeric_columns[col] = pd.arrays.SparseArray(numeric_columns[col], fill_value=0)

    # Combine all hashed columns 
    hashed_matrix = hstack(hashed_columns)

    # Convert to sparse dataframe
    cat_hashed_df = pd.DataFrame.sparse.from_spmatrix(hashed_matrix)
    cat_hashed_df.columns = [f"hash_{i}" for i in range(cat_hashed_df.shape[1])]

    #Combine 
    df_hashed_combined = pd.concat([numeric_columns.reset_index(drop=True), cat_hashed_df.reset_index(drop=True)], axis=1)

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
   
def create_train_test_val_sets(x, y, label_col="Label", test_size=0.15, n_splits=5, random_state=42):
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
    - 'x_train', 'y_train': dataset for CV
    - 'x_val', 'y_val': validation set
    - 'x_test', 'y_test': test set
    - 'cv_splits': list of (train_idx, val_idx) tuples for K-Fold CV on train_val
    """

    #Hold-out test set
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    #Split the remaining data into Train and Validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42
    )

    #create stratified K-Fold CV splits on the train_val set
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = list(skf.split(x_train, y_train)) #list of (train_idx, val_idx) tuples for each fold

    #Stratified K-Fold CV splits on the train_val set
    print(f"Train/validation/test split prepared: {len(y_train)} instances for training, {len(y_val)} instances for validation, {len(y_test)} instances for testing")
    print(f"Stratified {n_splits}-fold CV splits created.")

    return {
        "x_train": x_train, #feature set for cv
        "y_train": y_train, #label set for cv
        "x_val": x_val, #feature set for validation
        "y_val": y_val, #label set for validation
        "x_test": x_test,
        "y_test": y_test,
        "cv_splits": cv_splits #A list of tuples (train_idx, val_idx) representing each fold split for cross-validation
    }

def get_processed_df(dataset_path, dataset_name=None):
    """function to read processed data and return feature and label sets"""
    df_raw = load_dataset(dataset_path)
    print(f"----------Processing {dataset_name} Dataset----------")
    processed_df= process_data(df_raw)
    y =processed_df["Label"]
    x = processed_df.drop(columns=["Label"])

    return x, y

def main():
    x_mendeley, y_mendeley = get_processed_df(r"data\raw\Mendeley Dataset.csv", "Mendeley")
   # x_phiusiil, y_phiusiil = get_processed_df(r"data\raw\PhiUSIIL Dataset.csv", "PhiUSIIL")
    x_kaggle, y_kaggle = get_processed_df(r"data\raw\dataset_phishing.csv", "Kaggle")
    #Split data
    medeley_sets = create_train_test_val_sets(x_mendeley,y_mendeley, label_col="Label", test_size=0.2, n_splits=5)
   # phiusiil_sets = create_train_test_val_sets(x_phiusiil,y_phiusiil, label_col="Label", test_size=0.2, n_splits=5)
    kaggle_sets = create_train_test_val_sets(x_kaggle, y_kaggle, label_col="Label", test_size=0.2, n_splits=5)

    # Accessing the folds (this is just how to access the first fold, you would need to do this in a loop)
    # train_idx, val_idx = medeley_sets["cv_splits"][0] #first fold 
    # x_train, x_val = medeley_sets["x_train_val"].iloc[train_idx], medeley_sets["x_train_val"].iloc[val_idx]
    # y_train, y_val = medeley_sets["y_train_val"].iloc[train_idx], medeley_sets["y_train_val"].iloc[val_idx]



if __name__ == "__main__":
    main()