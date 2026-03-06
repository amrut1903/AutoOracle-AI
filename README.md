# 🚗 AutoOracle AI — Used Car Resale Price Predictor

An AI-powered web application that predicts the resale value of used cars in the Indian market using a Gradient Boosting model trained on the CarDekho dataset.

---

## 📁 Project Structure

```
AUTOORACLE AI/
├── dataset/
│   └── car_dekho.csv           # Raw dataset from CarDekho India
├── static/
│   ├── css/
│   │   └── style.css
│   ├── images/logo/            # Brand logo PNGs (8 brands)
│   │   ├── chevrolet.png
│   │   ├── ford.png
│   │   ├── honda.png
│   │   ├── hyundai.png
│   │   ├── mahindra.png
│   │   ├── maruti_suzuki.png
│   │   ├── tata.png
│   │   └── toyota.png
│   └── videos/
│       ├── user_bg.mp4         # Background video for predictor page
│       └── teacher_bg.mp4      # Background video for dashboard page
├── templates/
│   ├── user.html               # Predictor UI (main page)
│   └── teacher.html            # Model dashboard / analytics page
├── app.py                      # Flask backend
├── train_model.py              # Model training script
├── model_artifacts.pkl         # Saved model + metrics (generated after training)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/autooracle-ai.git
cd autooracle-ai
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🤖 Train the Model

Run this **once** before starting the app. It reads the dataset, engineers features, trains the model, and saves `model_artifacts.pkl`.

```bash
python train_model.py
```

Expected output:
```
🚗 Loading dataset...
   Rows before cleaning : XXXX
   Rows after cleaning  : XXXX

🚀 Training model (this takes ~60 seconds)...

📊 Results:
   R² Score : 0.8XXX
   MAE      : ₹XX,XXX
   RMSE     : ₹XX,XXX

✅ Saved → model_artifacts.pkl
```

---

## 🚀 Run the App

```bash
python app.py
```

Then open your browser at:
```
http://127.0.0.1:5000
```

---

## 🌐 Pages

| Route | Page | Description |
|---|---|---|
| `/` | Predictor | Select brand, model, KM driven, specs → get price prediction |
| `/teacher` | Dashboard | View model metrics, charts, feature importance, ML pipeline |

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Dataset | CarDekho India (`car_dekho.csv`) |
| Target | `selling_price` (₹) |
| Target Transform | `log1p` → `expm1` |
| Train / Test Split | 80% / 20% |
| n_estimators | 1500 |
| learning_rate | 0.015 |
| max_depth | 7 |
| subsample | 0.85 |
| max_features | 0.8 |
| R² Score | ~0.82–0.84 |
| Encoding | OneHotEncoder (5 categorical features) |

### Engineered Features
| Feature | Formula |
|---|---|
| `car_age` | `current_year − year` |
| `km_per_year` | `km_driven ÷ (car_age + 1)` |
| `log_km` | `log(km_driven + 1)` |
| `age_sq` | `car_age²` |
| `km_age_ratio` | `km_driven × car_age` |

---

## 🚘 Supported Brands

Maruti · Hyundai · Mahindra · Tata · Honda · Ford · Toyota · Chevrolet

---

## 📦 Dependencies

```
flask==3.0.3
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
```

---

## 📌 Notes

- Run `train_model.py` first — the app will crash without `model_artifacts.pkl`
- The `generated/` folders inside `dataset/` and `static/` are auto-created and can be ignored
- Video backgrounds (`user_bg.mp4`, `teacher_bg.mp4`) must be placed manually in `static/videos/`

---

## 👤 Author

Built as a machine learning + full-stack web project using Flask, scikit-learn, and vanilla HTML/CSS/JS.
