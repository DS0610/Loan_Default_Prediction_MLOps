import pandas as pd
import os
from datetime import datetime

LOG_FILE = "prediction_logs.csv"

def log_prediction(request_data: dict, prediction: int, probability: float):
    file_exists = os.path.exists(LOG_FILE)
    
    log_data = request_data.copy()
    log_data['prediction'] = prediction
    log_data['probability'] = probability
    log_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_df = pd.DataFrame([log_data])

    if not file_exists:
        log_df.to_csv(LOG_FILE, index=False)
    else:
        log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)