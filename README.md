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

---

## Model

* Algorithm: Random Forest Classifier
* Recall: 100%
* Accuracy: 99.7%

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
