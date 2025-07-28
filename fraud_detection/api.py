import json
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
from xgboost.spark import SparkXGBClassifierModel

app = FastAPI()

spark = SparkSession.builder.getOrCreate()

model = SparkXGBClassifierModel.load("models/xgb_fraud_detection_v1_0.spark")
pipe = PipelineModel.load("models/processing_pipeline")


class DataFrameInput(BaseModel):
    data: List[Dict[str, Any]]


EXPECTED_INPUT_COLUMNS = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "payment_type",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "credit_risk_score",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "source",
    "session_length_in_minutes",
    "device_os",
    "keep_alive_session",
    "device_distinct_emails_8w",
]


def preprocess_data(df: DataFrame) -> DataFrame:
    """Preprocess the input data to match model expectations."""

    processed_df = df
    if "month" in processed_df.columns:
        processed_df = processed_df.drop("month")

    processed_df = pipe.transform(processed_df)

    if "id" in processed_df.columns:
        processed_df = processed_df.select(
            col("scaled_features").alias("features"), col("fraud_bool").alias("label"), col("id")
        )
    else:
        processed_df = processed_df.select(
            col("scaled_features").alias("features"), col("fraud_bool").alias("label")
        )
    return processed_df


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Fraud Detection API",
        "description": "This API provides fraud detection predictions using a XGBoost model",
    }


@app.post("/predict")
async def predict_batch(input_data: DataFrameInput):
    start = time.time()
    try:
        df = spark.createDataFrame(input_data.data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    missing_cols = []
    missing_cols = [c for c in EXPECTED_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=500,
            detail=f"The dataframe is missing the following columns: {missing_cols}",
        )

    try:
        processed_df = preprocess_data(df)

        predictions: DataFrame = model.transform(processed_df)

        extract_first_prob = udf(lambda v: float(v[0]), FloatType())

        # separate prob column
        predictions = predictions.withColumn(
            "prob_class_0", extract_first_prob("probability")
        ).withColumn("prob_class_1", 1 - col("prob_class_0"))

        selected_columns = ["prediction", "prob_class_0", "prob_class_1"]
        if "id" in df.columns:
            selected_columns.append("id")
        predictions = predictions.select(selected_columns)

        json_data = predictions.toJSON().collect()  # list of json strings
        result = [json.loads(row) for row in json_data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    end = time.time()
    latency = (end - start) * 1000
    return {"predictions": result, "latency": latency}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
