# Smart Agriculture: Crop Yield Prediction & Decision Support

An end-to-end MLOps starter project that predicts crop yield (tonnes/hectare) from soil, weather, crop and soil-type features.

## Structure

This project intentionally follows a compact, script-oriented layout: `train.py`, `main.py`, `models/`, `tests/`, `Dockerfile`, `requirements.txt`, and one GitHub Actions workflow with exactly two jobs.

## Important dataset note

`data/crop_yield.csv` in this starter is **synthetic demonstration data** created to make the pipeline runnable. It is not suitable for scientific conclusions or real farming recommendations. Replace it with a validated real-world dataset before presenting model quality as meaningful. Keep the same column names:

- nitrogen, phosphorus, potassium
- temperature, humidity, ph, rainfall
- crop_type, soil_type
- yield_tons_per_hectare (target)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Start MLflow tracking server

In terminal 1:

```bash
mlflow server --host 127.0.0.1 --port 6001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

## Train and register

In terminal 2:

```bash
python train.py
```

The script compares RandomForestRegressor and GradientBoostingRegressor using MAE, RMSE and R². It logs runs to MLflow, saves a local joblib copy, registers the winning model as `crop-yield-predictor`, and assigns the `champion` alias.

The default RMSE quality threshold is intentionally permissive for synthetic demo data. Set `MAX_ACCEPTABLE_RMSE` to a meaningful threshold only after evaluating a real dataset.

## Run API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`.

Example request to `POST /predict`:

```json
{
  "nitrogen": 80,
  "phosphorus": 45,
  "potassium": 40,
  "temperature": 26.5,
  "humidity": 70,
  "ph": 6.5,
  "rainfall": 650,
  "crop_type": "Wheat",
  "soil_type": "Loamy"
}
```

## Run tests

```bash
pytest -q
```

Tests use an injected fake model for API checks and do not require a running MLflow server. Training tests use the included sample CSV.

## Docker

The API loads the champion model from MLflow at startup. Therefore, the MLflow tracking server and registry must be reachable from the container. Configure `MLFLOW_TRACKING_URI` to a reachable host address; `127.0.0.1` inside a container refers to the container itself.

```bash
docker build -t smart-agriculture-api .
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:6001 \
  smart-agriculture-api
```

On Linux, use the host gateway configuration or put the API and MLflow server on the same Docker network.

## CI workflow

`.github/workflows/mlops.yml` contains exactly two jobs:
1. `test`: install dependencies and run pytest.
2. `build-and-deploy`: build and inspect the Docker image after tests pass.

The second job currently builds and verifies the image; actual cloud deployment requires configuring a deployment provider and its secrets.

## Decision-support extension

This starter returns a yield estimate only. Add recommendations only from verified, crop-specific agronomic rules or trusted agricultural advisory sources. Do not infer fertilizer or irrigation prescriptions from yield predictions alone.
