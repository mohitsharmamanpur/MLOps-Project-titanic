from flask import Flask, request, jsonify
import boto3
import joblib

app = Flask(__name__)

BUCKET_NAME = "mohit-titanic-mlops-lw-2025"
MODEL_FILE = "titanic_model.pkl"

def download_model_from_s3():
    """Download model from S3 and load it with joblib."""
    s3 = boto3.client("s3")
    s3.download_file(BUCKET_NAME, MODEL_FILE, MODEL_FILE)
    return joblib.load(MODEL_FILE)

# Load model at startup
model = download_model_from_s3()

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "GET":
            # Example: /predict?features=3,1,22.0,1,0,7.25
            features_str = request.args.get("features")
            if not features_str:
                return jsonify({"error": "Missing 'features' parameter"}), 400
            features = [float(x) for x in features_str.split(",")]

        elif request.method == "POST":
            data = request.get_json()
            if not data or "features" not in data:
                return jsonify({"error": "Missing 'features' in JSON body"}), 400
            features = data["features"]

        # Make prediction
        prediction = model.predict([features])
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
