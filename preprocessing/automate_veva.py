import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

def run_preprocessing():
    file_path = 'smart_shipment_logistics.csv'
    df = pd.read_csv(file_path)
    
    cols_to_drop = ['shipment_id', 'created_date', 'eta_date']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
        
    df.drop_duplicates(inplace=True)
    
    for col in df.select_dtypes(include=['object', 'bool']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if 'status' in num_cols:
        num_cols = num_cols.drop('status')
        
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    os.makedirs('dataset_preprocessing', exist_ok=True)
    df.to_csv('dataset_preprocessing/dataset_ready.csv', index=False)
    joblib.dump(scaler, 'dataset_preprocessing/scaler.pkl')
    print("Automasi preprocessing sukses.")

if __name__ == "__main__":
    run_preprocessing()