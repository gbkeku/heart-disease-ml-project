# Heart Disease Prediction using Machine Learning

## 📌 Problem
Heart disease is a leading cause of death worldwide.  
This project aims to predict whether a patient has heart disease using clinical data.

---

## 📊 Dataset
- Source: UCI Heart Disease Dataset
- 13 clinical features
- Target: 0 (No Disease), 1 (Disease Present)

---

## ⚙️ Machine Learning Pipeline
Data → Preprocessing → Model → Training → Evaluation → Deployment

---

## 🤖 Models Used
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

Hyperparameter tuning was performed using GridSearchCV.

---

## 📈 Results
- Best Model (Validation): SVM  
- Best Test Performance: Logistic Regression  
- F1 Score ≈ 0.83  
- ROC-AUC ≈ 0.90  

---

## 🖥️ Demo

### Run locally:
```bash
pip install -r requirements.txt
python train.py
python demo.py

Then open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used for clinical diagnosis.

## 📚 References

UCI Heart Disease Dataset

Scikit-learn documentation

