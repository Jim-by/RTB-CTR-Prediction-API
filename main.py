from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from typing import Optional
from pathlib import Path
import gc
import shutil
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RTB CTR Prediction API",
    description="API for predicting ad click probability",
    version="1.0.0"
)

# Setting up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Downloading the latest model
def load_latest_model():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    logger.info(f"Search for models in the directory: {model_dir}")

    if not os.path.exists(model_dir):
        logger.error(f"Models folder is not found: {model_dir}")
        return None

    # Looking at all the model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.txt')]
    if not model_files:
        logger.error("Model files is not found")
        return None

    # Take the file with the latest modification date
    latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
    model_path = os.path.join(model_dir, latest_model)

    try:
        logger.info(f"Loading a model from: {model_path}")
        return lgb.Booster(model_file=model_path)
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return None

# Load the model when the application starts
model = load_latest_model()

if model is None:
    logger.warning("Warning: Model not loaded at startup")
else:
    logger.info("Model loaded successfully")

class PredictionRequest(BaseModel):
    hour: int = Field(..., example=14102100)
    banner_pos: int = Field(..., example=0)
    site_id: str = Field(..., example="1fbe01fe")
    site_domain: str = Field(..., example="f3845767")
    site_category: str = Field(..., example="28905ebd")
    app_id: str = Field(..., example="ecad2386")
    app_domain: str = Field(..., example="7801e8d9")
    app_category: str = Field(..., example="07d7df22")
    device_id: str = Field(..., example="a99f214a")
    device_type: int = Field(..., example=1)
    device_conn_type: int = Field(..., example=2)
    C1: Optional[int] = Field(None, example=1005)

class PredictionResponse(BaseModel):
    predicted_ctr: float
    is_weekend: bool

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing data for the model"""
    logger.info("Start of data preprocessing")

    # Time
    df['hour'] = df['hour'] % 100
    df['day'] = (df['hour'] // 100) % 7
    df['is_weekend'] = (df['day'] >= 5).astype('int8')
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Categorical features
    cat_cols = ['site_id', 'site_domain', 'site_category', 'app_id',
                'app_domain', 'app_category', 'device_id']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Interactions
    if 'hour' in df.columns and 'site_id' in df.columns:
        df['hour_site'] = df['hour'].astype(str) + "_" + df['site_id'].astype(str)
    if 'hour' in df.columns and 'app_id' in df.columns:
        df['hour_app'] = df['hour'].astype(str) + "_" + df['app_id'].astype(str)
    if 'site_id' in df.columns and 'app_id' in df.columns:
        df['site_app'] = df['site_id'].astype(str) + "_" + df['app_id'].astype(str)

    # Removing unnecessary columns
    df = df.drop(['day'], axis=1, errors='ignore')

    logger.info("Data preprocessing is complete")
    return df

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model

    logger.info("A request for prediction has been received")

    if model is None:
        logger.error("Error: Model is not loaded")
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        logger.info("Converting a Query to a DataFrame")
        input_data = pd.DataFrame([request.dict()])
        logger.debug(f"Input data: {input_data.to_dict()}")

        # We get a list of features that the model expects
        model_features = model.feature_name()
        logger.info(f"The model expects signs: {model_features}")

        # Data preprocessing
        logger.info("Data preprocessing")
        processed_data = preprocess_data(input_data)

        # Make sure that all the features of the model are present in the data
        for feature in model_features:
            if feature not in processed_data.columns:
                # For missing features, create a column with zero values
                processed_data[feature] = 0
                logger.warning(f"Added missing feature: {feature}")

        # We select only those features that the model expects
        processed_data = processed_data[model_features]

        # Transforming categorical features
        cat_features = ['site_id', 'site_domain', 'site_category', 'app_id',
                        'app_domain', 'app_category', 'device_id', 'device_type',
                        'banner_pos', 'device_conn_type', 'hour_site', 'hour_app', 'site_app']

        # Аilter only those categorical features that are present in the data
        available_cat_features = [f for f in cat_features if f in processed_data.columns]

        logger.info(f"Categorical features for prediction: {available_cat_features}")

        for col in available_cat_features:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].astype('category')

        logger.info("Making a prediction")
        prediction = model.predict(processed_data)
        logger.debug(f"Prediction: {prediction}")

        return {
            "predicted_ctr": float(prediction[0]),
            "is_weekend": bool(processed_data['is_weekend'].iloc[0] if 'is_weekend' in processed_data.columns else False)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    logger.info("Checking the functionality of the API")
    return {"status": "healthy"}

@app.post("/load-model")
async def load_model():
    """Endpoint for manual model loading"""
    global model

    logger.info("Attempting to load model")

    try:
        model = load_latest_model()
        if model is None:
            logger.error("Failed to load model")
            raise HTTPException(status_code=500, detail="Failed to load model")

        model_path = os.path.join(os.path.dirname(__file__), "models", "model.txt")
        logger.info(f"Model successfully loaded from: {model_path}")
        return {
            "status": "success",
            "message": "Model successfully loaded",
            "model_path": model_path
        }
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train")
async def train_model():
    """Endpoint for model training"""
    global model

    logger.info("Starting model training")

    try:
        # Absolute path to a file in a container
        train_path = Path("/app/data/train.csv")

        # Checking the existence of a file
        if not train_path.exists():
            logger.error(f"File is not found: {train_path}")
            raise FileNotFoundError(f"File is not found: {train_path}")

        logger.info(f"Dataset used: {train_path}")

        # === 2. Loading Data ===
        logger.info("Loading Data...")
        df = pd.read_csv(
            train_path,
            nrows=1_200_000,
            usecols=['click', 'hour', 'banner_pos', 'site_id', 'site_domain',
                    'site_category', 'app_id', 'app_domain', 'app_category',
                    'device_id', 'device_type', 'device_conn_type', 'C1']
        )

        logger.info(f"Lines loaded: {len(df):,}")

        # === 3. Feature engineering ===
        logger.info("Creating features...")
        # Time
        df['hour'] = df['hour'] % 100
        df['day'] = (df['hour'] // 100) % 7
        df['is_weekend'] = (df['day'] >= 5).astype('int8')
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Transformation into categorical features
        obj_cols = ['site_id', 'site_domain', 'site_category', 'app_id',
                    'app_domain', 'app_category', 'device_id']
        for col in obj_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Frequency coding
        for col in ['site_id', 'site_domain', 'app_id', 'device_id']:
            freq = df[col].value_counts()
            df[f'{col}_freq'] = df[col].cat.codes.map(freq).fillna(0).astype('int16')

        # Target coding with anti-aliasing
        global_ctr = df['click'].mean()
        smoothing = 100
        for col in ['site_id', 'site_domain', 'app_id', 'device_id']:
            stats = df.groupby(col)['click'].agg(['mean', 'count'])
            df[f'{col}_te'] = (stats['mean'] * stats['count'] + global_ctr * smoothing) / (stats['count'] + smoothing)
            df[f'{col}_te'] = df[col].cat.codes.map(df[f'{col}_te']).fillna(global_ctr)

        # Signs of interaction
        df['hour_site'] = df['hour'].astype(str) + "_" + df['site_id'].astype(str)
        df['hour_app'] = df['hour'].astype(str) + "_" + df['app_id'].astype(str)
        df['site_app'] = df['site_id'].astype(str) + "_" + df['app_id'].astype(str)

        # Categorical features
        cat_features = ['site_id', 'site_domain', 'site_category', 'app_id',
                        'app_domain', 'app_category', 'device_id', 'device_type',
                        'banner_pos', 'device_conn_type', 'hour_site', 'hour_app', 'site_app']
        cat_features = [col for col in cat_features if col in df.columns]

        for col in cat_features:
            df[col] = df[col].astype('category')

        features = [c for c in df.columns if c != 'click']
        logger.info(f"Features created: {len(features)}")

        # === 4. Data separation ===
        split = int(len(df) * 0.85)
        X_train = df.iloc[:split][features]
        y_train = df.iloc[:split]['click']
        X_valid = df.iloc[split:][features]
        y_valid = df.iloc[split:]['click']

        del df
        gc.collect()

        # === 5. LightGBM training ===
        logger.info("Model training...")
        lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss,auc',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_data_in_leaf': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'seed': 42,
            'force_row_wise': True,
            'verbosity': -1,
            'max_bin': 127
        }

        # Model training
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=200, 
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=10)
            ]
        )

        # === 6. Model evaluation ===
        y_pred = model.predict(X_valid)
        from sklearn.metrics import roc_auc_score, log_loss
        auc = roc_auc_score(y_valid, y_pred)
        logloss = log_loss(y_valid, y_pred)

        logger.info(f"AUC = {auc:.6f}")
        logger.info(f"LogLoss = {logloss:.6f}")

        # Saving the model
        os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
        model_path = os.path.join(os.path.dirname(__file__), "models", f"avazu_improved_auc_{auc:.5f}.txt")
        model.save_model(model_path)

        shutil.copy(model_path, os.path.join(os.path.dirname(__file__), "models", "model.txt"))

        logger.info(f"Model saved: {model_path}")
        return {
            "status": "success",
            "auc": auc,
            "logloss": logloss,
            "model_path": model_path
        }

    except Exception as e:
        logger.error(f"Error while training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
