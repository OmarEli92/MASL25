from Model.ImmuneCell import ImmuneCell
from Model.Patient import Patient
from Model.Enumerazioni import MacrophagePhenotype
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.RCCModel import RCCModel

class Macrophage(ImmuneCell):
    def __init__(self, unique_id: int, model: "RCCModel", patient: Patient, phenotype: MacrophagePhenotype):
        super().__init__(unique_id, model, patient)
        self.phenotype = phenotype
        self.cytokines = defaultdict(int)  # ✅ Fix per evitare KeyError

    def secrete_cytokines(self):
        if self.phenotype == MacrophagePhenotype.M1:
            self.cytokines["IFN_gamma"] += 1
        else:
            self.cytokines["IL_10"] += 1

