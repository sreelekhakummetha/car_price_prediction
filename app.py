from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        year = int(request.form["year"])
        present_price = float(request.form["present_price"])
        kms_driven = int(request.form["kms_driven"])
        fuel = int(request.form["fuel"])
        seller = int(request.form["seller"])
        transmission = int(request.form["transmission"])
        owner = int(request.form["owner"])

        input_data = np.array([[year, present_price, kms_driven,
                                fuel, seller, transmission, owner]])

        prediction = round(model.predict(input_data)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
