"""Combine results_dca.json and results_ph.json into results.json (no number is altered)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
out = {"decision_curve_and_calibration": json.load(open(os.path.join(HERE, "results_dca.json"))),
       "proportional_hazards": json.load(open(os.path.join(HERE, "results_ph.json")))}
with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote results.json")
