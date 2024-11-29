import sys
import os
# Add the parent directory (prods2) to the Python path

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
kode_dir = os.path.join(parent_dir, "Kode")

# Add Kode directory to sys.path
sys.path.append(kode_dir)
from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from integrated_labeling import make_labeled_file

app = Flask(__name__)

# Function to run periodically
def periodic_task():
    pass

def scheduler_run():
    scheduler = BackgroundScheduler()
    #scheduler.add_job(periodic_task, 'interval', days=5)  # Run every 5 days
    scheduler.add_job(periodic_task, 'interval', seconds=10)  # Temporary for testing
    scheduler.start()

@app.route("/make_new_labelling_data", methods=["GET"])
def label():
    key = request.args.get('kata_kunci')
    if key != "prodstimcitrasatelithore":
        return jsonify({"Error": "Can't Label Data With Unauthorized"})
    runtime = make_labeled_file()
    return jsonify({"Success": f"Runtime: {runtime}"})

@app.route("/home", methods=["GET", "POST"])
def calculator():
    result = None
    if request.method == "POST":
        try:
            num1 = float(request.form["num1"])
            num2 = float(request.form["num2"])
            operation = request.form["operation"]

            if operation == "add":
                result = add(num1, num2)
            elif operation == "subtract":
                result = subtract(num1, num2)
            elif operation == "multiply":
                result = multiply(num1, num2)
            elif operation == "divide":
                result = divide(num1, num2)
            else:
                result = "Invalid operation"
        except ValueError:
            result = "Error: Please enter valid numbers"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
