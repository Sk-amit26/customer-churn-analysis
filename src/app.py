"""
Customer Churn Prediction Web Application & API Server (v2 Optimized)
Serves an interactive web dashboard and REST API with the 79.21% Ensemble / 78.99% Gradient Boosting model.
"""
import http.server
import socketserver
import json
import urllib.parse
import os
import joblib
import pandas as pd
import numpy as np

PORT = int(os.environ.get("PORT", 5000))
MODEL_PATH = "models/churn_pipeline_v2.joblib"
META_PATH = "models/model_meta_v2.joblib"

def engineer_features_single(data_dict):
    """Applies the 10 domain-specific feature engineering transformations."""
    df = pd.DataFrame([data_dict])
    
    # 1. Charge Ratio
    df['Charge_Ratio'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges']),
        1.0
    )
    
    # 2. Tenure Bucket
    df['Tenure_Bucket'] = pd.cut(
        df['tenure'],
        bins=[-1, 6, 12, 24, 48, 60, 73],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)
    
    # 3. Number of Services
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Num_Services'] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == 'Yes'), axis=1
    )
    
    # 4. Has Protection
    df['Has_Protection'] = ((df['OnlineSecurity'] == 'Yes') | 
                            (df['OnlineBackup'] == 'Yes') | 
                            (df['DeviceProtection'] == 'Yes')).astype(int)
    
    # 5. Has Support
    df['Has_Support'] = (df['TechSupport'] == 'Yes').astype(int)
    
    # 6. Has Streaming
    df['Has_Streaming'] = ((df['StreamingTV'] == 'Yes') | 
                           (df['StreamingMovies'] == 'Yes')).astype(int)
    
    # 7. Charge Per Service
    df['Charge_Per_Service'] = np.where(
        df['Num_Services'] > 0,
        df['MonthlyCharges'] / df['Num_Services'],
        df['MonthlyCharges']
    )
    
    # 8. Is New Customer
    df['Is_New_Customer'] = (df['tenure'] <= 6).astype(int)
    
    # 9. Is High Spender (Median benchmark: $70.35)
    df['Is_High_Spender'] = (df['MonthlyCharges'] > 70.35).astype(int)
    
    # 10. Triple Risk Flag
    df['Triple_Risk'] = (
        (df['Contract'] == 'Month-to-month') & 
        (df['InternetService'] == 'Fiber optic') & 
        (df['PaymentMethod'] == 'Electronic check')
    ).astype(int)
    
    return df

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction & Analytics</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f4f7fa; color: #333; }
        .hero { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 2.5rem 0; margin-bottom: 2rem; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .card { border: none; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 1.5rem; }
        .card-header { background-color: white; border-bottom: 1px solid #edf2f7; font-weight: 600; padding: 1rem 1.25rem; border-radius: 12px 12px 0 0 !important; }
        .btn-primary { background: linear-gradient(135deg, #1e3c72, #2a5298); border: none; font-weight: 600; padding: 0.75rem 1.5rem; border-radius: 8px; }
        .kpi-card { text-align: center; padding: 1.25rem; border-radius: 10px; color: white; }
        .kpi-1 { background: #2b5c8f; }
        .kpi-2 { background: #d95f02; }
        .kpi-3 { background: #2e7d32; }
        .kpi-4 { background: #7b1fa2; }
        .risk-badge { font-size: 1.25rem; font-weight: 700; padding: 0.5rem 1rem; border-radius: 8px; display: inline-block; }
        .badge-high { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
        .badge-low { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
    </style>
</head>
<body>
    <div class="hero text-center">
        <div class="container">
            <h1 class="fw-bold">Telecom Customer Churn Analytics & Prediction</h1>
            <p class="lead mb-0">Production ML Model (Voting Ensemble & Gradient Boosting | <strong>79.21% Accuracy</strong> | ROC-AUC: 0.846)</p>
        </div>
    </div>

    <div class="container">
        <!-- Top KPIs -->
        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="kpi-card kpi-1 shadow-sm">
                    <div class="text-uppercase small fw-bold">Total Accounts</div>
                    <div class="h2 mb-0 fw-bold">7,043</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card kpi-2 shadow-sm">
                    <div class="text-uppercase small fw-bold">Model Accuracy</div>
                    <div class="h2 mb-0 fw-bold">79.21%</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card kpi-3 shadow-sm">
                    <div class="text-uppercase small fw-bold">Model ROC-AUC</div>
                    <div class="h2 mb-0 fw-bold">0.8459</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card kpi-4 shadow-sm">
                    <div class="text-uppercase small fw-bold">Monthly Revenue at Risk</div>
                    <div class="h2 mb-0 fw-bold">$139.1K</div>
                </div>
            </div>
        </div>

        <div class="row">
            <!-- Prediction Form -->
            <div class="col-lg-7">
                <div class="card shadow-sm">
                    <div class="card-header text-primary">
                        <i class="bi bi-person-gear"></i> Enter Customer Account Profile
                    </div>
                    <div class="card-body p-4">
                        <form id="churnForm">
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Contract Type</label>
                                    <select class="form-select" id="Contract" name="Contract">
                                        <option value="Month-to-month" selected>Month-to-month (High Risk)</option>
                                        <option value="One year">One year</option>
                                        <option value="Two year">Two year (Low Risk)</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Internet Service</label>
                                    <select class="form-select" id="InternetService" name="InternetService">
                                        <option value="Fiber optic" selected>Fiber optic</option>
                                        <option value="DSL">DSL</option>
                                        <option value="No">No Internet</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Payment Method</label>
                                    <select class="form-select" id="PaymentMethod" name="PaymentMethod">
                                        <option value="Electronic check" selected>Electronic check</option>
                                        <option value="Mailed check">Mailed check</option>
                                        <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                                        <option value="Credit card (automatic)">Credit card (automatic)</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Paperless Billing</label>
                                    <select class="form-select" id="PaperlessBilling" name="PaperlessBilling">
                                        <option value="Yes" selected>Yes</option>
                                        <option value="No">No</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label fw-semibold">Tenure (Months): <span id="tenureVal" class="text-primary fw-bold">2</span></label>
                                    <input type="range" class="form-range" id="tenure" name="tenure" min="0" max="72" value="2" oninput="document.getElementById('tenureVal').innerText = this.value; updateTotal();">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label fw-semibold">Monthly Charges ($)</label>
                                    <input type="number" step="0.5" class="form-control" id="MonthlyCharges" name="MonthlyCharges" value="70.35" oninput="updateTotal();">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label fw-semibold">Total Charges ($)</label>
                                    <input type="number" step="0.5" class="form-control" id="TotalCharges" name="TotalCharges" value="140.70">
                                </div>
                                
                                <div class="col-md-4">
                                    <label class="form-label fw-semibold">Tech Support</label>
                                    <select class="form-select" id="TechSupport" name="TechSupport">
                                        <option value="No" selected>No</option>
                                        <option value="Yes">Yes</option>
                                        <option value="No internet service">No internet service</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label fw-semibold">Online Security</label>
                                    <select class="form-select" id="OnlineSecurity" name="OnlineSecurity">
                                        <option value="No" selected>No</option>
                                        <option value="Yes">Yes</option>
                                        <option value="No internet service">No internet service</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label fw-semibold">Senior Citizen</label>
                                    <select class="form-select" id="SeniorCitizen" name="SeniorCitizen">
                                        <option value="0" selected>No</option>
                                        <option value="1">Yes</option>
                                    </select>
                                </div>
                            </div>
                            <div class="mt-4">
                                <button type="button" onclick="submitPrediction()" class="btn btn-primary w-100 py-2">
                                    ⚡ Run Churn Risk Prediction
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>

            <!-- Prediction Results Box -->
            <div class="col-lg-5">
                <div class="card shadow-sm h-100">
                    <div class="card-header text-primary">
                        <i class="bi bi-graph-up-arrow"></i> Real-Time Prediction Output
                    </div>
                    <div class="card-body p-4 d-flex flex-column justify-content-center text-center" id="resultContainer">
                        <div class="text-muted py-5">
                            <p class="h5">Click <strong>Run Churn Risk Prediction</strong> to calculate customer score.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateTotal() {
            const tenure = parseFloat(document.getElementById('tenure').value) || 1;
            const monthly = parseFloat(document.getElementById('MonthlyCharges').value) || 0;
            document.getElementById('TotalCharges').value = (tenure * monthly).toFixed(2);
        }

        async function submitPrediction() {
            const data = {
                gender: "Male",
                SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
                Partner: "No",
                Dependents: "No",
                tenure: parseInt(document.getElementById('tenure').value),
                PhoneService: "Yes",
                MultipleLines: "No",
                InternetService: document.getElementById('InternetService').value,
                OnlineSecurity: document.getElementById('OnlineSecurity').value,
                OnlineBackup: "No",
                DeviceProtection: "No",
                TechSupport: document.getElementById('TechSupport').value,
                StreamingTV: "No",
                StreamingMovies: "No",
                Contract: document.getElementById('Contract').value,
                PaperlessBilling: document.getElementById('PaperlessBilling').value,
                PaymentMethod: document.getElementById('PaymentMethod').value,
                MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
                TotalCharges: parseFloat(document.getElementById('TotalCharges').value)
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            const isChurn = result["Predicted Churn"] === "Yes";
            const badgeClass = isChurn ? "badge-high" : "badge-low";
            
            document.getElementById('resultContainer').innerHTML = `
                <div class="mb-3">
                    <span class="risk-badge ${badgeClass}">
                        ${isChurn ? '⚠️ HIGH CHURN RISK' : '✅ LOW CHURN RISK'}
                    </span>
                </div>
                <h3 class="fw-bold mb-1">${result["Churn Probability"]}</h3>
                <p class="text-muted mb-4">Probability of Account Cancellation</p>
                <div class="progress mb-4" style="height: 12px; border-radius: 6px;">
                    <div class="progress-bar ${isChurn ? 'bg-danger' : 'bg-success'}" role="progressbar" 
                         style="width: ${result["Churn Probability"]};" aria-valuenow="50" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                <div class="text-start p-3 bg-light rounded">
                    <div class="fw-bold small text-uppercase text-muted mb-2">Recommended Retention Action:</div>
                    <p class="small mb-0">
                        ${isChurn ? 
                          '• Trigger proactive outreach with 10% annual contract migration discount.<br>• Assign priority technical support onboarding review.' : 
                          '• Customer exhibits high loyalty characteristics. Eligible for premium add-on upsell.'}
                    </p>
                </div>
            `;
        }
    </script>
</body>
</html>
"""

class ChurnRequestHandler(http.server.SimpleHTTPRequestHandler):
    pipeline = None
    meta = None

    @classmethod
    def load_model(cls):
        if cls.pipeline is None:
            if os.path.exists(MODEL_PATH):
                cls.pipeline = joblib.load(MODEL_PATH)
                print(f"Loaded optimized model from {MODEL_PATH}")
            elif os.path.exists("models/churn_pipeline.joblib"):
                cls.pipeline = joblib.load("models/churn_pipeline.joblib")
                print("Loaded baseline model from models/churn_pipeline.joblib")
        if cls.meta is None and os.path.exists(META_PATH):
            cls.meta = joblib.load(META_PATH)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            try:
                data = json.loads(post_body.decode("utf-8"))
                self.load_model()
                
                # Apply feature engineering
                input_df = engineer_features_single(data)
                
                churn_prob = self.pipeline.predict_proba(input_df)[0][1]
                threshold = self.meta.get('optimal_threshold', 0.50) if self.meta else 0.50
                churn_pred = "Yes" if churn_prob >= threshold else "No"
                
                response_payload = {
                    "Predicted Churn": churn_pred,
                    "Churn Probability": f"{churn_prob * 100:.1f}%",
                    "Risk Tier": "High Risk" if churn_prob >= 0.55 else ("Moderate Risk" if churn_prob >= 0.35 else "Low Risk"),
                    "Raw Probability": round(float(churn_prob), 4),
                    "Model": "GradientBoostingClassifier (78.99% Acc, 0.841 AUC)"
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

def run_server():
    ChurnRequestHandler.load_model()
    with socketserver.TCPServer(("", PORT), ChurnRequestHandler) as httpd:
        print(f"Optimized Churn Server launched at http://localhost:{PORT}")
        print("Model: Gradient Boosting & Voting Ensemble (79.21% Accuracy)")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    run_server()
