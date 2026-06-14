# Loan Approval Prediction API

## Project Overview

This project is an end-to-end Machine Learning application for loan approval prediction. The model predicts whether a loan application should be approved or rejected based on applicant information.

The project includes:

* Data preprocessing pipeline
* Machine learning model training
* FastAPI REST API
* Single and batch predictions
* Request validation using Pydantic
* Partial success handling for batch requests
* SQLite database for prediction logging
* Docker containerization
* Persistent storage using Docker volumes


## Key Features
* End-to-end ML pipeline
* Batch prediction with partial failure handling
* Input validation using Pydantic
* Dockerized deployment
* Persistent database storage (SQLite)

---

## Model

Model Performance:
* Accuracy: 99.7% (Random Forest)
* Best model selected after comparing multiple algorithms

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* FastAPI
* Pydantic
* SQLite
* Joblib
* Docker
* Uvicorn

---

## System Architecture

1. User sends request (FastAPI)
2. Request validated using Pydantic
3. Data preprocessing pipeline applied
4. Model prediction using trained Random Forest model
5. Result stored in SQLite database
6. Response returned to user


---

## Project Structure

```text
Loan/
│
├── src/
│   ├── app.py
│   ├── schemas.py
│   ├── database.py
│   ├── loan_preprocessor.py
│   ├── final_loan_model.pkl
│   └── database.db
│
├── training/
├── notebooks/
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── README.md
```

---

## API Endpoints

### Health Check

```http
GET /
```

Response:

```json
{
  "message": "Loan Approval API Running"
}
```

---

### Single Prediction

```http
POST /predict
```

Returns:

* application_id
* timestamp
* loan_status

---

### Batch Prediction

```http
POST /predict_batch
```

Features:

* Batch inference
* Partial success
* Error handling
* Validation
* Database logging

---

## Running Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start FastAPI

```bash
uvicorn src.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Docker

Build image:

```bash
docker build -t loan-api .
```

Run container:

```bash
docker run -p 8000:8000 loan-api
```

Run with persistent database:

```bash
docker run -p 8000:8000 -v %cd%\src:/app/src loan-api
```

---

## Database

SQLite database is used to store prediction history.

Stored fields:

* application_id
* timestamp
* loan_status

---

## Future Improvements

* PostgreSQL integration
* Unit testing
* CI/CD pipeline
* Model versioning
* Cloud deployment
* Authentication and authorization

---

## Author

Rehab Alsayed

Machine Learning Engineer
