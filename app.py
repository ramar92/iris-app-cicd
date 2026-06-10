from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pkl")

# Class labels
classes = [
    "setosa",
    "versicolor",
    "virginica"
]

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Iris Classifier API is Running!"
    })

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        features = np.array(
            data["features"]
        ).reshape(1, -1)

        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": classes[prediction]
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )