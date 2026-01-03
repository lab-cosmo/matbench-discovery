import numpy as np

kappa_dft = []
kappa_pet = []

with open("bro.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        kappa_dft.append(float(line.strip().strip("[]")))

with open("slurm-2208482.out", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "Total mode kappa does not sum to total kappa" in line:
            kappa = line.split("sum_mode_kappa_tot=array([[")[-1].split(",")[0]
            # kappa = lines[i + 1].split("kappa_p_rta=array([[")[-1].split(",")[0]
            kappa_pet.append(float(kappa))

kappa_dft = np.array(kappa_dft)
kappa_pet = np.array(kappa_pet)

# calculate srme
srme = np.mean(np.abs(kappa_dft - kappa_pet) / (kappa_dft + kappa_pet))
print(f"SRME: {srme:.4f}")
