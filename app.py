from flask import Flask, render_template, request, jsonify
import pickle, numpy as np

app = Flask(__name__)

# ── Load ─────────────────────────────────────────────────
with open("model_artifacts.pkl", "rb") as f:
    data = pickle.load(f)

pipeline   = data['pipeline']
r2         = data['r2']
mae        = data['mae']
rmse       = data['rmse']
cat_unique = data['cat_unique']

# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('user.html', cat_unique=cat_unique)

@app.route('/teacher')
def teacher():
    return render_template('teacher.html',
                           r2=r2, mae=mae, rmse=rmse,
                           cat_unique=cat_unique)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        b = request.get_json()

        year   = int(b['Year'])
        km     = int(b['Mileage'])
        fuel   = b['Fuel Type']
        trans  = b['Transmission']
        owner  = b['Owner']
        seller = b['Seller Type']
        brand  = b['Brand']

        import pandas as pd
        current_year  = pd.Timestamp.now().year
        car_age       = current_year - year
        km_per_year   = km / (car_age + 1)
        log_km        = np.log1p(km)
        age_sq        = car_age ** 2
        km_age_ratio  = km * car_age

        row = pd.DataFrame([{
            "brand":        brand,
            "fuel":         fuel,
            "seller_type":  seller,
            "transmission": trans,
            "owner":        owner,
            "car_age":      car_age,
            "km_driven":    km,
            "km_per_year":  km_per_year,
            "log_km":       log_km,
            "age_sq":       age_sq,
            "km_age_ratio": km_age_ratio,
        }])

        log_pred = pipeline.predict(row)[0]
        price    = np.expm1(log_pred)

        return jsonify({
            'formatted':  f"₹{price:,.0f}",
            'r2_score':   f"{r2:.4f}",
            'mae':        f"₹{mae:,.0f}",
            'rmse':       f"₹{rmse:,.0f}",
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)