import pandas as pd
from kbmod.work_unit import WorkUnit
from astropy.table import Table
from kbmod.run_search import SearchRunner
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

wu = WorkUnit.from_fits("/epyc/projects/kbmod/data/20210908_B1h_047_test_data/20210908_B1h_047.wu")
t = Table.read("inframe_fakes_b1h.ecsv")
orbids = np.unique(t["ORBITID"])

for i in range(19, len(orbids)):
  oid = orbids[i]
  cols = ["oid", "total_num_results",
          "num_results_no_filter", "check_results_no_filter",
          "num_results_peak", "check_results_peak",
          "num_results_pred", "check_results_pred",
          "mean_dx", "max_dx", "min_dx", "max_dx_diff", "min_dx_diff",
          "mean_dy", "max_dy", "min_dy", "max_dy_diff", "min_dy_diff"]

  t0 = t[t["ORBITID"] == oid]
  total_num_results = len(t0)
  xs, ys = wu.get_pixel_coordinates(t0["RA"], t0["DEC"])

  dx = []
  for i in range(len(xs) - 2):
      dx.append(xs[i+1] - xs[i])

  dy = []
  for i in range(len(ys) - 2):
      dy.append(ys[i+1] - ys[i])
      
  mean_dx, min_dx, max_dx = np.mean(dx), np.min(dx), np.max(dx)
  mean_dy, min_dy, max_dy = np.mean(dy), np.min(dy), np.max(dy)
  max_dx_diff, min_dx_diff = max_dx - mean_dx, min_dx - mean_dx
  max_dy_diff, min_dy_diff = max_dy - mean_dy, min_dy - mean_dy

  time_diff = t0["mjd_mid"][-1] - t0["mjd_mid"][0]
  mean_vx = (xs[-1] - xs[0]) / time_diff
  mean_vy = (ys[-1] - ys[0]) / time_diff

  gen = {
    "name": "VelocityGridSearch",
    "vx_steps": 25,
    "min_vx": mean_vx - .125,
    "max_vx": mean_vx + .125,
    "vy_steps": 25,
    "min_vy": mean_vy - .125,
    "max_vy": mean_vy + .125,
  }

  wu.config.set("generator_config", gen)
  wu.config.set("do_clustering", False)
  wu.config.set("sigmaG_filter", False)
  wu.config.set("peak_offset_max", None)
  wu.config.set("predictive_line_cluster", False)
  wu.config.set("lh_level", 10.0)

  wu.config.set("result_filename", f"./no_filter/{oid}_pencil.ecsv")
  wu.reprojection_frame = "original"
  r = SearchRunner().run_search_from_work_unit(wu)
  num_results_no_filter = len(r)
  r_check = r[(r["x"] >= xs[0] - 1) & (r["x"] <= xs[0] + 1) & (r["y"] >= ys[0] - 1) & (r["y"] <= ys[0] + 1)]
  check_results_no_filter = len(r_check)

  if (check_results_no_filter > 0):
    wu.config.set("peak_offset_max", 6)
    wu.config.set("result_filename", f"./peak/{oid}_pencil.ecsv")
    
    r = SearchRunner().run_search_from_work_unit(wu)
    num_results_peak = len(r)
    r_check = r[(r["x"] >= xs[0] - 1) & (r["x"] <= xs[0] + 1) & (r["y"] >= ys[0] - 1) & (r["y"] <= ys[0] + 1)]
    check_results_peak = len(r_check)
    
    wu.config.set("peak_offset_max", None)
    wu.config.set("predictive_line_cluster", True)
    wu.config.set("result_filename", f"./predictive_line/{oid}_pencil.ecsv")
    
    r = SearchRunner().run_search_from_work_unit(wu)
    num_results_pred = len(r)
    r_check = r[(r["x"] >= xs[0] - 1) & (r["x"] <= xs[0] + 1) & (r["y"] >= ys[0] - 1) & (r["y"] <= ys[0] + 1)]
    check_results_pred = len(r_check)
    
  else:
    num_results_peak, check_results_peak, num_results_pred, check_results_pred = None, None, None, None
    
  l = [oid, total_num_results,
          num_results_no_filter, check_results_no_filter,
          num_results_peak, check_results_peak,
          num_results_pred, check_results_pred,
          mean_dx, max_dx, min_dx, max_dx_diff, min_dx_diff,
          mean_dy, max_dy, min_dy, max_dy_diff, min_dy_diff]

  df = pd.DataFrame(l).T
  df.to_csv("./summary2.csv", index=False, header=False, mode='a')
  