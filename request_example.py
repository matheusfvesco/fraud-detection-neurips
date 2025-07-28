import asyncio

import httpx
from pyspark.sql import DataFrame, SparkSession

spark = SparkSession.builder.getOrCreate()


def create_sample_dataframe() -> DataFrame:
    df = spark.read.parquet("data/processed/Base_test.parquet", header=True, inferSchema=True)

    df.drop("fraud_bool")

    sampled_df = df.sample(0.01, seed=42)

    return sampled_df


async def make_api_request(df: DataFrame):
    # Convert dataframe to dictionary format expected by the API
    json_df = [row.asDict() for row in df.collect()]
    input_data = {"data": json_df}

    url = "http://127.0.0.1:8000/predict"

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.post(url, json=input_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise Exception(
                f"API request failed with status {e.response.status_code}: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise Exception(f"Request error occurred: {str(e)}")


if __name__ == "__main__":
    sample_data = create_sample_dataframe()
    sample_data = sample_data.limit(5)

    try:
        result = asyncio.run(make_api_request(sample_data))
        preds_df = spark.createDataFrame(result["predictions"])
        print("Dataframe:")
        preds_df.show()
        print("Latency: ")
        print(f"{result['latency']} ms")
    except Exception as e:
        print(f"Error making API request: {str(e)}")
