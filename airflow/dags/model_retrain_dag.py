"""
Model Retraining DAG
====================
Automated model retraining pipeline with drift detection.

Schedule: Monthly on the 1st at 3:00 AM KST
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# Default arguments
default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=4),
}


def check_data_drift(**context) -> str:
    """Check for data drift to determine if retraining is needed.

    Returns:
        Task ID to branch to
    """
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy import stats

    data_dir = Path(os.environ.get("DATA_DIR", "/opt/airflow/data"))
    features_path = data_dir / "processed" / "combined_features.parquet"

    if not features_path.exists():
        print("No feature data found")
        return "skip_retrain"

    df = pd.read_parquet(features_path)

    # Split into reference (old) and current (recent) data
    split_date = df.index.max() - timedelta(days=30)
    reference = df[df.index < split_date]
    current = df[df.index >= split_date]

    if len(current) < 20:
        print("Insufficient recent data for drift detection")
        return "skip_retrain"

    # Check drift for key features using KS test
    drift_detected = False
    drift_features = []

    key_features = ["return_20d", "volatility_20d", "rsi_14", "ma_alignment"]

    for feature in key_features:
        if feature not in df.columns:
            continue

        ref_values = reference[feature].dropna()
        cur_values = current[feature].dropna()

        if len(ref_values) < 10 or len(cur_values) < 10:
            continue

        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(ref_values, cur_values)

        if p_value < 0.05:  # Significant drift
            drift_detected = True
            drift_features.append(feature)
            print(f"Drift detected in {feature}: KS={statistic:.4f}, p={p_value:.4f}")

    # Also check model performance degradation
    model_path = data_dir / "models" / "regime_classifier_latest.joblib"
    if model_path.exists():
        # In production, load model and check recent performance
        pass

    if drift_detected:
        print(f"Data drift detected in features: {drift_features}")
        context["ti"].xcom_push(key="drift_features", value=drift_features)
        return "retrain_model"
    else:
        print("No significant drift detected")
        return "skip_retrain"


def retrain_model(**context) -> dict:
    """Retrain the regime classification model.

    Returns:
        Dictionary with training results
    """
    import os
    import sys
    import joblib
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    from xgboost import XGBClassifier

    data_dir = Path(os.environ.get("DATA_DIR", "/opt/airflow/data"))
    model_dir = data_dir / "models"
    model_dir.mkdir(exist_ok=True)

    # Load features
    features_path = data_dir / "processed" / "combined_features.parquet"
    df = pd.read_parquet(features_path)

    # Use Ultra-5 features
    feature_cols = ["return_20d", "volatility_20d", "rsi_14", "ma_alignment", "volume_ratio"]
    available_cols = [c for c in feature_cols if c in df.columns]

    if len(available_cols) < 3:
        print(f"Insufficient features: {available_cols}")
        return {"status": "error", "message": "Insufficient features"}

    X = df[available_cols].dropna()

    # Generate labels (simplified for DAG)
    # In production, use RegimeLabeler from crypto-ml
    y = pd.Series("NOT_BULL", index=X.index)
    y[df.loc[X.index, "return_20d"] > 0.02] = "BULL_TREND"

    # Encode and scale
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Model trained: Accuracy={accuracy:.4f}, F1={f1:.4f}")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"regime_classifier_{timestamp}.joblib"

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": available_cols,
        "metrics": {"accuracy": accuracy, "f1": f1},
        "trained_at": timestamp,
    }, model_path)

    # Update latest symlink
    latest_path = model_dir / "regime_classifier_latest.joblib"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(model_path.name)

    print(f"Model saved to {model_path}")

    return {
        "status": "success",
        "model_path": str(model_path),
        "accuracy": accuracy,
        "f1": f1,
        "timestamp": timestamp,
    }


def validate_new_model(**context) -> str:
    """Validate the newly trained model.

    Returns:
        Task ID to branch to
    """
    ti = context["ti"]
    train_result = ti.xcom_pull(task_ids="retrain_model")

    if not train_result or train_result.get("status") != "success":
        print("Training failed, skipping validation")
        return "training_failed"

    accuracy = train_result.get("accuracy", 0)
    f1 = train_result.get("f1", 0)

    # Validation thresholds
    min_accuracy = 0.65
    min_f1 = 0.60

    if accuracy >= min_accuracy and f1 >= min_f1:
        print(f"Model passed validation: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        return "deploy_model"
    else:
        print(f"Model failed validation: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        return "validation_failed"


def deploy_model(**context) -> dict:
    """Deploy the validated model.

    Returns:
        Dictionary with deployment results
    """
    import os
    import shutil
    from pathlib import Path

    ti = context["ti"]
    train_result = ti.xcom_pull(task_ids="retrain_model")

    if not train_result:
        return {"status": "error", "message": "No training result"}

    model_path = Path(train_result["model_path"])
    data_dir = Path(os.environ.get("DATA_DIR", "/opt/airflow/data"))

    # Copy to production location
    prod_path = data_dir / "models" / "production" / "regime_classifier.joblib"
    prod_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(model_path, prod_path)

    print(f"Model deployed to {prod_path}")

    # Log deployment event
    deployment_log = {
        "timestamp": context["execution_date"].isoformat(),
        "model_path": str(model_path),
        "accuracy": train_result["accuracy"],
        "f1": train_result["f1"],
    }

    return {"status": "deployed", "deployment": deployment_log}


def send_retrain_report(**context) -> None:
    """Send retraining report notification."""
    ti = context["ti"]

    train_result = ti.xcom_pull(task_ids="retrain_model")
    deploy_result = ti.xcom_pull(task_ids="deploy_model")

    report = f"""
    Model Retraining Report
    =======================
    Date: {context['execution_date']}

    Training Results:
    - Status: {train_result.get('status', 'N/A') if train_result else 'N/A'}
    - Accuracy: {train_result.get('accuracy', 'N/A') if train_result else 'N/A'}
    - F1 Score: {train_result.get('f1', 'N/A') if train_result else 'N/A'}

    Deployment:
    - Status: {deploy_result.get('status', 'N/A') if deploy_result else 'N/A'}
    """

    print(report)


# Create the DAG
with DAG(
    dag_id="crypto_model_retrain",
    default_args=default_args,
    description="Automated model retraining with drift detection",
    schedule_interval="0 3 1 * *",  # 1st of every month at 3:00 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["data-engineering", "crypto", "ml", "retraining"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # Check for data drift
    check_drift = BranchPythonOperator(
        task_id="check_data_drift",
        python_callable=check_data_drift,
    )

    # Skip retraining
    skip = EmptyOperator(task_id="skip_retrain")

    # Retrain model
    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    # Validate model
    validate = BranchPythonOperator(
        task_id="validate_model",
        python_callable=validate_new_model,
    )

    # Training failed
    train_failed = EmptyOperator(task_id="training_failed")

    # Validation failed
    val_failed = EmptyOperator(task_id="validation_failed")

    # Deploy model
    deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )

    # Send report
    report = PythonOperator(
        task_id="send_report",
        python_callable=send_retrain_report,
        trigger_rule="none_failed_min_one_success",
    )

    # Set dependencies
    start >> check_drift
    check_drift >> skip >> report >> end
    check_drift >> retrain >> validate
    validate >> train_failed >> report
    validate >> val_failed >> report
    validate >> deploy >> report >> end
