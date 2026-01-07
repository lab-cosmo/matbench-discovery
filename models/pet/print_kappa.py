import matbench_discovery.phonons.thermal_conductivity as ltc
from matbench_discovery import phonons
from matbench_discovery.enums import DataFiles
from matbench_discovery.metrics.phonons import calc_kappa_metrics_from_dfs, Key

import pandas as pd
import sys

for disp in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:

    df_kappa = pd.read_json(f"orblike_kappa_results_{sys.argv[1]}/pet-phononDB-LTC-FIRE_force0.0001_sym1e-05_ispmTrue_disp{disp}.json.gz").set_index("material_id")
    df_dft = pd.read_json(DataFiles.phonondb_pbe_103_kappa_no_nac.path).set_index("mp_id")
    df_ml_metrics = calc_kappa_metrics_from_dfs(df_kappa, df_dft)
    # Compute and print summary metrics
    kappa_sre = df_ml_metrics[Key.sre].mean()
    kappa_srme = df_ml_metrics[Key.srme].mean()
    print(f"Disp: {disp:.2f} SRE: {kappa_sre:.3f} SRME: {kappa_srme:.3f}")
