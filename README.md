# Fraud Detection System 🕵️‍♂️

[![Cookiecutter Data Science](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A machine learning system for detecting fraudulent bank opening requests using the NeurIPS 2022 Bank Account Fraud Dataset.

## Features

- End-to-end fraud detection pipeline
- REST API for model serving
- Comprehensive data analysis notebooks
- Docker support for easy deployment

## Project Structure

```
├── artifacts/               <- Temporary artifacts from notebooks
├── data/                    <- All project data
│   ├── external/            <- Third-party data sources
│   ├── interim/            <- Intermediate processed data
│   ├── processed/          <- Final processed data
│   └── raw/                <- Original immutable data
├── docs/                    <- Project documentation
├── fraud_detection/         <- Python package source code
│   ├── api.py              <- FastAPI application for model serving
│   └── __init__.py         <- Package initialization
├── models/                  <- Trained model binaries
│   ├── xgb_fraud_detection_v1_0.spark
│   └── xgb_fraud_detection_v1_0.json
├── notebooks/               <- Jupyter notebooks for analysis
│   ├── 0.0-mfv-exploratory-data-analysis.ipynb
│   ├── 0.1-mfv-pre-processing.ipynb
│   ├── 0.2-mfv-statistical-tests.ipynb
│   └── 1.0-mfv-model-training.ipynb
├── references/              <- Data dictionaries, research papers
├── reports/                 <- Generated reports and visualizations
│   └── figures/            <- Saved visualization assets
├── Dockerfile               <- Container configuration
├── LICENSE                  <- MIT License
├── Makefile                 <- Project automation
├── pyproject.toml           <- Python project configuration
├── README.md                <- This file
├── request_example.py       <- Example API client script
└── uv.lock                  <- Dependency lock file
```

## Dataset

This project uses the [Bank Account Fraud Dataset from NeurIPS 2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), which contains:

- 1 million synthetic financial bank account applications
- feature engineer features
- Realistic fraud patterns based on actual fraud scenarios

## Quick Start

### Installation

1. Clone this repository
2. Install dependencies:

```bash
make requirements
```

3. Download the dataset:

```bash
make download-data
```

### Running the Analysis

Execute the notebooks in order:
1. `0.0-mfv-exploratory-data-analysis.ipynb`
2. `0.1-mfv-pre-processing.ipynb`
3. `0.2-mfv-statistical-tests.ipynb`
4. `1.0-mfv-model-training.ipynb`

## Model Serving

### Using Docker

Build the Docker image:

```bash
docker build . -t fraud_detection_api
```

Run the container:

```bash
docker run -p 8000:8000 fraud_detection_api:latest
```

### Making API Requests

Use the included `request_example.py` to test the API:

```bash
python request_example.py
```

Or manually make requests:

```python
import requests

data = {
    "transaction": {
        "amount": 1500.00,
        "customer_age": 35,
        # ... other features
    }
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.