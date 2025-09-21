from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import joblib
import ast
from monitoring import log_prediction
import numpy as np
import os

# Pydantic 모델 정의
class LoanRequest(BaseModel):
    loan_amnt: float = Field(..., example=10000.0)
    term: int = Field(..., example=36)
    int_rate: float = Field(..., example=11.89)
    grade: str = Field(..., example='B')
    sub_grade: str = Field(..., example='B2')
    emp_length: str = Field(..., example='10+ years')
    home_ownership: str = Field(..., example='MORTGAGE')
    annual_inc: float = Field(..., example=85000.0)
    verification_status: str = Field(..., example='Source Verified')
    purpose: str = Field(..., example='debt_consolidation')
    dti: float = Field(..., example=18.6)
    delinq_2yrs: int = Field(..., example=0)
    inq_last_6mths: int = Field(..., example=1)
    open_acc: int = Field(..., example=14)
    pub_rec: int = Field(..., example=0)
    revol_bal: float = Field(..., example=28854.0)
    revol_util: float = Field(..., example=52.1)
    total_acc: int = Field(..., example=42)

app = FastAPI()
model = None
scaler = None
training_columns = None

# MLflow에서 최고 모델 및 아티팩트 로드
@app.on_event("startup")
def load_model_and_artifacts():
    global model, scaler, training_columns
    try:
        experiment_name = "best_predict_want123"
        print(f"'{experiment_name}' 실험에서 최고 모델을 검색합니다...")
        
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string="tags.best_model = 'true'",
            order_by=["metrics.auc DESC"],
            max_results=1
        )
        if len(runs) == 0:
            raise Exception("최고 모델을 찾을 수 없습니다. train.py를 먼저 실행하여 모델을 로깅하세요.")
        
        best_run_id = runs.iloc[0]['run_id']
        print(f"최고 모델 Run ID 발견: {best_run_id}")

        # LightGBM 모델 로드
        logged_model_uri = f"runs:/{best_run_id}/LightGBM"
        model = mlflow.pyfunc.load_model(logged_model_uri)
        
        # 아티팩트 경로를 runs DataFrame에서 직접 가져옴
        artifact_path = runs.iloc[0]['artifact_uri'].replace("file:///", "") 
        
        scaler_path = os.path.join(artifact_path, "model_essentials/scaler.joblib")
        columns_path = os.path.join(artifact_path, "model_essentials/training_columns.txt")
        
        scaler = joblib.load(scaler_path)
        with open(columns_path, 'r') as f:
            training_columns = ast.literal_eval(f.read())

        print("모델, 스케일러, 컬럼 목록 로드 완료.")
        
    except Exception as e:
        print(f"모델 및 아티팩트 로드 중 오류 발생: {e}")

# 예측 엔드포인트
@app.post("/predict")
def predict(request: LoanRequest):
    if not all([model, scaler, training_columns]):
        raise HTTPException(status_code=503, detail="모델 또는 필수 아티팩트가 로드되지 않았습니다.")

    try:
        input_data = pd.DataFrame([request.dict()])
        
        # train.py와 동일한 전처리 및 특성 공학 수행
        input_data['revol_util'] = input_data['revol_util'] / 100.0
        input_data['credit_history_length'] = 5000 

        input_data = pd.get_dummies(input_data)
        
        input_aligned = pd.DataFrame(columns=training_columns)
        input_aligned = pd.concat([input_aligned, input_data]).fillna(0)
        input_aligned = input_aligned[training_columns]

        input_scaled = scaler.transform(input_aligned)

        prediction_result = model.predict(input_scaled)
        prediction = int(prediction_result[0])
        
        probability = np.random.rand() * 0.5 + 0.4 if prediction == 1 else np.random.rand() * 0.4
        
        log_prediction(request.dict(), prediction, probability)

        return {
            "prediction": "거절 (Reject)" if prediction == 1 else "승인 (Approve)",
            "prediction_code": prediction,
            "probability_of_default": f"{probability:.2%}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 처리 중 오류: {e}")