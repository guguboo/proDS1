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
from make_dta import make_all_dta

app = Flask(__name__)

# Function to run periodically
def periodic_task():
    pass

# def make_dta():
#     make_all_dta()

def scheduler_run():
    scheduler = BackgroundScheduler()
    #scheduler.add_job(periodic_task, 'interval', days=5)  # Run every 5 days
    scheduler.add_job(periodic_task, 'interval', seconds=10)  # Temporary for testing
    scheduler.start()

@app.route("/make_new_labelling_data", methods=["GET"])
def label():
    key = request.args.get('kata_kunci')
    if key != "prodstimcitrasatelit":
        return jsonify({"Error": "Can't Label Data With Unauthorized"})
    runtime = make_labeled_file()
    return jsonify({"Success": f"Runtime: {runtime}"})

@app.route("/classify_at_location", methods=["GET"])
def classify():
    
    pass

@app.route("/make_dta", methods=["GET"])
def make_dta():
    # result = make_all_dta()
    # return jsonify({"Success": result})
    pass

if __name__ == "__main__":
    app.run(debug=True)
