from metatomic.torch.ase_calculator import MetatomicCalculator
from metatomic.torch.ase_calculator import SymmetrizedCalculator


calc = SymmetrizedCalculator(MetatomicCalculator("pet-oam-1epoch-55.pt", device="cuda"))
for name, param in calc._model.named_parameters():
    print(name, param.shape)
