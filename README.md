# Comparative Analysis of Machine Learning Models for Heart Disease Prediction Using Clinical Data

## Problem
Predict whether a patient has heart disease using clinical patient data.

## Dataset
Place the dataset at `data/heart.csv`.

## Project Logic
This project matches the notebook workflow:

1. Train candidate models
2. Tune hyperparameters with validation
3. Select the best model based on validation ROC-AUC
4. Evaluate that selected model once on the held-out test set
5. Save and deploy the selected model

## Files
- `dataset.py` - data loading and cleaning
- `model.py` - preprocessing pipeline and candidate models
- `train.py` - hyperparameter tuning, validation selection, artifact saving
- `evaluate.py` - evaluation plots for the selected deployment model
- `inference.py` - load model and run prediction
- `demo.py` - simple command-line demo

## Run
Install requirements:

```bash
pip install -r requirements.txt
```

Train and save artifacts:

```bash
python train.py
```

Evaluate the selected deployment model:

```bash
python evaluate.py
```

Run the simple demo:

```bash
python demo.py
```

## Flask Demo App
Run the web demo after training artifacts are created:

```bash
python demo.py
```

Then open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## You can also Run the model from the notebooks.