import sys
import os
# Add the parent directory (prods2) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler
from calculator import add, subtract, multiply, divide

app = Flask(__name__)

# Function to run periodically
def periodic_task():
    num1, num2 = 10, 5
    # print("Running periodic calculation...")
    results = {
        "add": add(num1, num2),
        "subtract": subtract(num1, num2),
        "multiply": multiply(num1, num2),
        "divide": divide(num1, num2),
    }
    print(f"Periodic Task Results: {results}")

scheduler = BackgroundScheduler()
#scheduler.add_job(periodic_task, 'interval', days=5)  # Run every 5 days
scheduler.add_job(periodic_task, 'interval', seconds=10)  # Temporary for testing
scheduler.start()

@app.route("/test_calculator")
def test_calculator():
    # Use the calculator functions
    num1, num2 = 10, 5
    results = {
        "add": add(num1, num2),
        "subtract": subtract(num1, num2),
        "multiply": multiply(num1, num2),
        "divide": divide(num1, num2),
    }
    return f"Results: {results}"

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
