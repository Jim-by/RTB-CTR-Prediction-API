# RTB-style_pet  
**Real‑Time Bidding (RTB) — Click‑Through Rate (CTR) Prediction API**  

Predict ad click probability in real‑time bidding scenarios.  

---

## Project description  
This project is a **FastAPI‑based REST API** designed to predict the **Click‑Through Rate (CTR)** for ad impressions in **Real‑Time Bidding (RTB)** systems.  

Key features:  
- `/predict` endpoint — returns predicted CTR for a given ad impression.  
- `/train` endpoint — trains a new LightGBM model on the provided dataset.  
- `/health` endpoint — health‑check.  
- Fully containerized with **Docker**.  

**Technologies:** Python, FastAPI, LightGBM, Pandas, Docker, Docker‑Compose.  

---

## Project structure  

```
RTB-style_pet/
├── app/                     # Main application code
│   ├── __init__.py
│   ├── main.py              # FastAPI app & endpoints
│   ├── schemas.py           # Pydantic request/response models
│   ├── utils.py             # Helper functions
│   └── models/              # Directory for saved models
├── docker/                   # Docker configurations
│   ├── Dockerfile
│   ├── Dockerfile.airflow
│   └── docker-compose.yml
├── data/                     # Place your datasets here (ignored by .gitignore)
├── requirements.txt          # Python dependencies
├── README.md
└── .gitignore
```

---

## How to run  

### Prerequisites  
- Python 3.9+  
- Docker (recommended for easiest setup)  

---

### **Option 1: Run locally (without Docker)**  

1.  Clone the repository:  
    ```bash
    git clone https://github.com/your-username/RTB-style_pet.git
    cd RTB-style_pet
    ```

2.  Install dependencies:  
    ```bash
    pip install -r requirements.txt
    ```

3.  Start the API server:  
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```

4.  Open **Swagger UI** in your browser:  
    `http://localhost:8000/docs`  

---

### **Option 2: Run with Docker (Recommended)**  

1.  Build and launch containers:  
    ```bash
    docker-compose up --build
    ```

2.  The API will be available at:  
    `http://localhost:8000`  

 **Note:**  
- The `train.csv` file **must** be placed inside the `data/` folder before training a model.  

---

##  API Endpoints  

### 1. **Predict CTR**  
**POST** `http://localhost:8000/predict`  

**Request Body (JSON):**  

```json
{
  "hour": 14102100,
  "banner_pos": 0,
  "site_id": "1fbe01fe",
  "site_domain": "f3845767",
  "site_category": "28905ebd",
  "app_id": "ecad2386",
  "app_domain": "7801e8d9",
  "app_category": "07d7df22",
  "device_id": "a99f214a",
  "device_type": 1,
  "device_conn_type": 2,
  "C1": 1005
}
```

**Response:**  

```json
{
  "predicted_ctr": 0.00123,
  "is_weekend": false
}
```

---

### 2. **Health check**  
**GET** `http://localhost:8000/health`  

**Response:**  
```json
{"status": "healthy"}
```

---

### 3. **Train model**  
**GET** `http://localhost:8000/train`  

Triggers model training using `data/train.csv`.  
After successful training, the model is saved to `app/models/`.  

**Response example:**  

```json
{
  "status": "success",
  "auc": 0.92345,
  "logloss": 0.12345,
  "model_path": "/app/models/avazu_improved_auc_0.92345.txt"
}
```

>  **Requirement:** `train.csv` must exist in `data/`.

---

##  Docker details  

- **`Dockerfile`** — builds the FastAPI application image.  
- **`docker-compose.yml`** — orchestrates the API service (and optionally Airflow).  

To stop containers:  
```bash
docker-compose down
```

---

##  Model training process  

When `/train` is called, the following steps are executed:  

1.  Load `train.csv` (≈1.2 M rows).  
2.  **Feature engineering**  
    - Extract `day`, `is_weekend`, cyclic time features (`hour_sin`, `hour_cos`).  
    - Frequency & Target encoding for categorical fields.  
    - Interaction features (`hour_site`, `hour_app`, `site_app`).  
3.  Train‑Validation split (85%/15%).  
4.  Train **LightGBM** model with early stopping.  
5.  Evaluate using **AUC‑ROC** and **LogLoss**.  
6.  Save the model to `app/models/`.  

---

##  Technologies used  

- **Python** 3.9  
- **FastAPI** – web framework  
- **LightGBM** – gradient boosting machine  
- **Pandas / NumPy** – data manipulation  
- **Docker** & **Docker‑Compose** – containerization  

---


 **Tip:**  
Check the interactive API documentation at `http://localhost:8000/docs` after launching the server!
