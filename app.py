import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load AdaBoost Model
MODEL_PATH = "adaboost.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: '{MODEL_PATH}' not found. Please place it in the application directory.")

# Feature names identified from the pickle metadata
FEATURE_NAMES = [
    "age", "gender", "state", "family_income", "high_school_gpa", 
    "sat_score", "act_score", "attendance_rate", "ap_courses", 
    "extracurricular_count", "volunteer_hours", "leadership_positions", 
    "coding_projects", "social_media_hours", "online_certifications", 
    "essay_score", "recommendation_score", "interview_score"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Student Evaluation Predictor</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            min-height: 100vh;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .glass-input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #f8fafc;
            transition: all 0.3s ease;
        }

        .glass-input:focus {
            background: rgba(15, 23, 42, 0.8);
            border-color: #6366f1;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
            outline: none;
        }

        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 15px rgba(99, 102, 241, 0.4); }
            50% { box-shadow: 0 0 25px rgba(129, 140, 248, 0.7); }
        }

        .btn-glow {
            animation: pulseGlow 3s infinite;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .animate-fade-in {
            animation: fadeIn 0.5s ease-out forwards;
        }
    </style>
</head>
<body class="text-slate-200 flex flex-col justify-between py-10 px-4 sm:px-6 lg:px-8">

    <div class="max-w-5xl mx-auto w-full">
        <!-- Header -->
        <header class="text-center mb-8 animate-fade-in">
            <div class="inline-flex items-center justify-center p-3 bg-indigo-600/30 rounded-full mb-4 border border-indigo-500/30">
                <i class="fa-solid fa-brain text-indigo-400 text-3xl"></i>
            </div>
            <h1 class="text-4xl font-extrabold text-white tracking-tight sm:text-5xl">
                AdaBoost Predictor Model
            </h1>
            <p class="mt-2 text-slate-400 text-sm sm:text-base">
                Enter student metrics below to compute class outcome using your trained ensemble model.
            </p>
        </header>

        <!-- Main Form Card -->
        <div class="glass-card rounded-2xl p-6 sm:p-10 shadow-2xl mb-8 animate-fade-in">
            <form id="predictForm" class="space-y-6">
                
                <h2 class="text-xl font-semibold text-indigo-300 border-b border-slate-700/60 pb-2 mb-4">
                    <i class="fa-solid fa-sliders mr-2"></i> Input Parameters
                </h2>

                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                    <!-- Dynamic Input Fields -->
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Age</label>
                        <input type="number" step="any" name="age" value="18" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Gender (0 = M, 1 = F)</label>
                        <input type="number" step="any" name="gender" value="0" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">State Code</label>
                        <input type="number" step="any" name="state" value="1" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Family Income ($)</label>
                        <input type="number" step="any" name="family_income" value="65000" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">High School GPA</label>
                        <input type="number" step="0.01" name="high_school_gpa" value="3.8" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">SAT Score</label>
                        <input type="number" step="any" name="sat_score" value="1350" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">ACT Score</label>
                        <input type="number" step="any" name="act_score" value="29" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Attendance Rate (%)</label>
                        <input type="number" step="0.1" name="attendance_rate" value="95.5" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">AP Courses</label>
                        <input type="number" step="any" name="ap_courses" value="4" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Extracurricular Count</label>
                        <input type="number" step="any" name="extracurricular_count" value="3" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Volunteer Hours</label>
                        <input type="number" step="any" name="volunteer_hours" value="50" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Leadership Positions</label>
                        <input type="number" step="any" name="leadership_positions" value="2" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Coding Projects</label>
                        <input type="number" step="any" name="coding_projects" value="3" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Social Media Hours</label>
                        <input type="number" step="any" name="social_media_hours" value="2.5" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Online Certifications</label>
                        <input type="number" step="any" name="online_certifications" value="2" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Essay Score</label>
                        <input type="number" step="any" name="essay_score" value="88" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Recommendation Score</label>
                        <input type="number" step="any" name="recommendation_score" value="90" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Interview Score</label>
                        <input type="number" step="any" name="interview_score" value="85" required class="glass-input w-full px-3 py-2 rounded-lg text-sm">
                    </div>
                </div>

                <div class="pt-4 flex justify-center">
                    <button type="submit" class="btn-glow w-full sm:w-auto px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl transition duration-200 flex items-center justify-center space-x-2">
                        <i class="fa-solid fa-wand-magic-sparkles"></i>
                        <span>Run Model Prediction</span>
                    </button>
                </div>
            </form>
        </div>

        <!-- Result Card Output (Hidden by Default) -->
        <div id="resultCard" class="hidden glass-card rounded-2xl p-6 text-center animate-fade-in border-t-4 border-indigo-500">
            <h3 class="text-lg font-medium text-slate-400 uppercase tracking-wider">Prediction Output</h3>
            <div id="resultValue" class="text-4xl font-extrabold text-white my-3">--</div>
            <p id="resultDetails" class="text-slate-300 text-sm max-w-xl mx-auto"></p>
        </div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = parseFloat(value) || 0;
            });

            const resultCard = document.getElementById('resultCard');
            const resultValue = document.getElementById('resultValue');
            const resultDetails = document.getElementById('resultDetails');

            resultCard.classList.remove('hidden');
            resultValue.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-indigo-400"></i> Processing...`;
            resultDetails.innerText = 'Calculating prediction using AdaBoost classification...';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (response.ok) {
                    resultValue.innerHTML = `<span class="text-emerald-400">${result.prediction}</span>`;
                    resultDetails.innerText = `Prediction generated successfully with model input size of ${result.features_used} features.`;
                } else {
                    resultValue.innerHTML = `<span class="text-rose-400">Error</span>`;
                    resultDetails.innerText = result.error || 'An error occurred during prediction.';
                }
            } catch (err) {
                resultValue.innerHTML = `<span class="text-rose-400">Network Error</span>`;
                resultDetails.innerText = 'Unable to reach backend endpoint.';
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file 'adaboost.pkl' not found on server."}), 500

    try:
        data = request.get_json(force=True)
        # Build array based on model features in correct order
        input_vector = [float(data.get(feat, 0)) for feat in FEATURE_NAMES]
        input_array = np.array([input_vector])
        
        prediction = model.predict(input_array)[0]
        
        # Format prediction response
        pred_output = int(prediction) if isinstance(prediction, (np.integer, int)) else str(prediction)
        
        return jsonify({
            "status": "success",
            "prediction": pred_output,
            "features_used": len(FEATURE_NAMES)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
