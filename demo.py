from flask import Flask, request, render_template_string
from inference import predict_from_payload, load_artifacts

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Heart Disease Demo</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            margin: 0;
            background: #f5f7fb;
            color: #1f2937;
        }
        .wrap {
            max-width: 1100px;
            margin: 24px auto;
            padding: 0 16px;
        }
        .header {
            background: #0f4c81;
            color: white;
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.08);
        }
        .grid {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.06);
            border: 1px solid #e5e7eb;
        }
        .field-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0,1fr));
            gap: 14px;
        }
        label {
            display: block;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 6px;
            color: #374151;
        }
        input, select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            box-sizing: border-box;
            background: white;
        }
        .actions {
            margin-top: 16px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            background: #0f4c81;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 16px;
            cursor: pointer;
            font-weight: 700;
        }
        .secondary {
            background: #64748b;
        }
        .badge {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .high {
            background: #fee2e2;
            color: #991b1b;
        }
        .low {
            background: #dcfce7;
            color: #166534;
        }
        .muted {
            color: #6b7280;
            font-size: 14px;
        }
        ul {
            margin-top: 8px;
            padding-left: 20px;
        }
        .disclaimer {
            margin-top: 16px;
            padding: 12px 14px;
            background: #fff8e1;
            color: #7a4b00;
            border: 1px solid #f3d27a;
            border-left: 5px solid #f39c12;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.5;
        }
        .meta {
            margin-top: 8px;
            font-size: 14px;
            color: #dbeafe;
        }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
            .field-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="wrap">
    <div class="header">
        <h1 style="margin:0 0 6px 0;">Heart Disease Clinical Decision Support Demo</h1>
        <div class="meta">Loaded deployment model: <strong>{{ deployment_model }}</strong></div>
        <div class="meta">Selection basis: <strong>{{ selection_metric }}</strong></div>
    </div>

    <div class="grid">
        <div class="card">
            <h2 style="margin-top:0;">Patient Input</h2>
            <form method="post">
                <div class="field-grid">
                    <div>
                        <label>Age</label>
                        <input type="number" step="1" name="age" value="{{ form_values.age }}">
                    </div>
                    <div>
                        <label>Sex</label>
                        <select name="sex">
                            {% for v in sex_options %}
                            <option value="{{ v }}" {% if form_values.sex == v %}selected{% endif %}>{{ v }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div>
                        <label>Chest Pain</label>
                        <select name="cp">
                            {% for v in cp_options %}
                            <option value="{{ v }}" {% if form_values.cp == v %}selected{% endif %}>{{ v }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label>Resting Blood Pressure</label>
                        <input type="number" step="1" name="trestbps" value="{{ form_values.trestbps }}">
                    </div>

                    <div>
                        <label>Cholesterol</label>
                        <input type="number" step="1" name="chol" value="{{ form_values.chol }}">
                    </div>
                    <div>
                        <label>Fasting Blood Sugar</label>
                        <select name="fbs">
                            {% for v in fbs_options %}
                            <option value="{{ v }}" {% if form_values.fbs == v %}selected{% endif %}>{{ v }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div>
                        <label>Resting ECG</label>
                        <select name="restecg">
                            {% for v in restecg_options %}
                            <option value="{{ v }}" {% if form_values.restecg == v %}selected{% endif %}>{{ v }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label>Max Heart Rate (thalch)</label>
                        <input type="number" step="1" name="thalch" value="{{ form_values.thalch }}">
                    </div>

                    <div>
                        <label>Exercise Induced Angina</label>
                        <select name="exang">
                            {% for v in exang_options %}
                            <option value="{{ v }}" {% if form_values.exang == v %}selected{% endif %}>{{ v }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label>Oldpeak</label>
                        <input type="number" step="0.1" name="oldpeak" value="{{ form_values.oldpeak }}">
                    </div>

                    <div>
                        <label>Slope</label>
                        <select name="slope">
                            {% for v in slope_options %}
                            <option value="{{ v }}" {% if form_values.slope == v %}selected{% endif %}>{{ v }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label>CA</label>
                        <input type="number" step="0.1" name="ca" value="{{ form_values.ca }}">
                    </div>

                    <div>
                        <label>Thal</label>
                        <select name="thal">
                            {% for v in thal_options %}
                            <option value="{{ v }}" {% if form_values.thal == v %}selected{% endif %}>{{ v }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>

                <div class="actions">
                    <button type="submit">Predict Risk</button>
                    <button type="submit" name="sample" value="1" class="secondary">Load Demo Sample</button>
                </div>
            </form>
        </div>

        <div class="card">
            <h2 style="margin-top:0;">Prediction Result</h2>

            {% if result %}
                <div class="badge {% if result.prediction == 1 %}high{% else %}low{% endif %}">
                    {{ result.prediction_label }}
                </div>
                <p><strong>Risk probability:</strong> {{ "%.1f"|format(result.risk_probability * 100) }}%</p>
                <p><strong>Recommendation:</strong> {{ result.recommendation }}</p>
                <p><strong>Top contributing factors:</strong></p>
                <ul>
                    {% for factor in result.top_factors %}
                    <li>{{ factor }}</li>
                    {% endfor %}
                </ul>
                <div class="disclaimer">
                    <strong>Disclaimer:</strong> {{ result.disclaimer }}
                </div>
            {% else %}
                <p class="muted">Enter patient values on the left, then click <strong>Predict Risk</strong>.</p>
            {% endif %}
        </div>
    </div>
</div>
</body>
</html>
"""

DEFAULT_SAMPLE = {
    "age": "63",
    "sex": "Male",
    "cp": "asymptomatic",
    "trestbps": "145",
    "chol": "233",
    "fbs": "True",
    "restecg": "normal",
    "thalch": "150",
    "exang": "False",
    "oldpeak": "2.3",
    "slope": "flat",
    "ca": "0",
    "thal": "fixed defect",
}

SEX_OPTIONS = ["Male", "Female"]
CP_OPTIONS = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
FBS_OPTIONS = ["True", "False"]
RESTECG_OPTIONS = ["normal", "st-t abnormality", "lv hypertrophy"]
EXANG_OPTIONS = ["True", "False"]
SLOPE_OPTIONS = ["upsloping", "flat", "downsloping"]
THAL_OPTIONS = ["normal", "fixed defect", "reversable defect"]

@app.route("/", methods=["GET", "POST"])
def home():
    _, summary = load_artifacts()
    result = None
    form_values = DEFAULT_SAMPLE.copy()

    if request.method == "POST":
        if request.form.get("sample") == "1":
            form_values = DEFAULT_SAMPLE.copy()
        else:
            form_values = {k: request.form.get(k, "") for k in DEFAULT_SAMPLE.keys()}
            try:
                result = predict_from_payload(form_values)
            except Exception as exc:
                result = {
                    "prediction": 0,
                    "prediction_label": "Input Error",
                    "risk_probability": 0.0,
                    "recommendation": f"Please review input values. {exc}",
                    "top_factors": ["The submitted values could not be processed."],
                    "disclaimer": "Educational demonstration only. Not for clinical diagnosis or treatment."
                }

    return render_template_string(
        HTML_TEMPLATE,
        deployment_model=summary.get("deployment_model_name", "Unknown"),
        selection_metric=summary.get("deployment_selection_metric", "Unknown"),
        result=result,
        form_values=form_values,
        sex_options=SEX_OPTIONS,
        cp_options=CP_OPTIONS,
        fbs_options=FBS_OPTIONS,
        restecg_options=RESTECG_OPTIONS,
        exang_options=EXANG_OPTIONS,
        slope_options=SLOPE_OPTIONS,
        thal_options=THAL_OPTIONS,
    )

if __name__ == "__main__":
    app.run(debug=True)
