from matbench_discovery.enums import DataFiles
import pandas as pd
import ase.io


atoms_list = ase.io.read(DataFiles.phonondb_pbe_103_structures.path, index=":10")
df_dft = pd.read_json(DataFiles.phonondb_pbe_103_kappa_no_nac.path).set_index("mp_id")

for atoms in atoms_list:
    id = atoms.info["material_id"]
    kappa_tot_avg = df_dft.loc[id, "kappa_tot_avg"]
    name = df_dft.loc[id, "name"]
    print(id, name, kappa_tot_avg[0])
