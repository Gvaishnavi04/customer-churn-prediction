from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/churn_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        tenure = int(request.form["tenure"])
        monthly = float(request.form["monthly"])
        contract = int(request.form["contract"])
        internet = int(request.form["internet"])

        features = np.array([[tenure, monthly, contract, internet]])
        result = model.predict(features)

        prediction = "Customer Likely to Churn" if result[0] == 1 else "Customer Will Stay"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
