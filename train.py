import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# 1. 데이터 로드 및 전처리
def preprocess_data(file_path):
    print("데이터 로딩 및 전처리 시작")
    df = pd.read_csv(file_path, low_memory=False)

    cols_to_use = [
        'loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'emp_length',
        'home_ownership', 'annual_inc', 'verification_status', 'purpose',
        'dti', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths',
        'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'loan_status' 
    ]
    df = df[cols_to_use].copy()
    
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    df['loan_status'] = df['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
    
    df.dropna(inplace=True)

    df['term'] = df['term'].apply(lambda x: int(x.strip().split()[0]))
    df['revol_util'] = df['revol_util'].astype(str).str.rstrip('%').astype('float') / 100.0
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
    df['credit_history_length'] = (pd.to_datetime('today') - df['earliest_cr_line']).dt.days
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df = df.drop(columns=['earliest_cr_line'])
    
    print("전처리 완료.")
    return df

# 모델 학습 및 MLflow 로깅
def train_and_log_models(df):
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 데이터 불균형 처리를 위한 가중치 계산

    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count
    print(f"클래스 불균형 처리를 위한 가중치(scale_pos_weight): {scale_pos_weight_value:.2f}")

    # 모델 정의 (데이터 불균형 처리 파라미터 추가)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight_value)
    }

    mlflow.set_experiment("best_predict_want123")

    best_run_id = None
    best_auc = 0.0

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"{name} 모델 학습 및 로깅 시작")
            
            model.fit(X_train_scaled, y_train)
            
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
            
            auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({'auc': auc, 'accuracy': accuracy, 'f1_score': f1})
            
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            if not os.path.exists("artifacts"):
                os.makedirs("artifacts")
            
            cm_path = f"artifacts/confusion_matrix_{name}.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path, "confusion_matrices")
            plt.close()

            mlflow.sklearn.log_model(model, artifact_path=name)

            print(f"{name} 모델 AUC: {auc:.4f}, F1-Score: {f1:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_run_id = run.info.run_id
    
    # 최고 성능 모델에 스케일러와 컬럼 목록 아티팩트 저장
    scaler_path = "artifacts/scaler.joblib"
    joblib.dump(scaler, scaler_path)
    
    columns_path = "artifacts/training_columns.txt"
    with open(columns_path, 'w') as f:
        f.write(str(list(X.columns)))

    with mlflow.start_run(best_run_id):
        mlflow.log_artifact(scaler_path, "model_essentials")
        mlflow.log_artifact(columns_path, "model_essentials")
        mlflow.set_tag("best_model", "true")
        
    print(f"\n최고 성능 모델(Run ID: {best_run_id})에 스케일러와 컬럼 목록을 저장")

if __name__ == "__main__":
    file_path = r'C:\Loan_Default_Prediction_MLOps\data\accepted_2007_to_2018Q4.csv' 
    if os.path.exists(file_path):
        processed_df = preprocess_data(file_path)
        train_and_log_models(processed_df)
    else:
        print(f"오류: 데이터 파일 '{file_path}'를 찾을 수 없음.")
        print("Kaggle에서 데이터를 다운로드하여 'data' 폴더에 넣기.")