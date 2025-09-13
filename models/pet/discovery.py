from matbench_discovery.preds import discovery
import pandas as pd
from tabulate import tabulate

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(tabulate(discovery.df_metrics, headers='keys', tablefmt='psql'))
