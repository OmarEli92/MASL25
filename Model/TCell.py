
from Model.ImmuneCell import ImmuneCell
from Model.Patient import Patient
from Model.Enumerazioni import TCellSubtype
import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.RCCModel import RCCModel

class TCell(ImmuneCell):
    def __init__(self, unique_id: int, model: "RCCModel", patient: Patient,
                 subtype: TCellSubtype):  # Type-hint as RCCModel
        super().__init__(unique_id, model, patient)
        self.subtype = subtype
        self.target_priority = 0.8 if subtype == TCellSubtype.CD8 else 0.3
        self.activation_level = random.uniform(0.6, 1.0) if subtype == TCellSubtype.CD8 else random.uniform(0.4, 0.8)
        self.exhaustion = random.uniform(0.0, 1.0)
        self.activation = random.uniform(0.0, 1.0)

    def step(self):
        original_activation_level = self.activation_level
        self.activation_level = max(0.1, min(1.0, original_activation_level + (
                    self.activation - self.exhaustion)))
        super().step()
        self.exhaustion = min(1.0, self.exhaustion + random.uniform(0.001, 0.005))
        self.activation = max(0.0, self.activation - random.uniform(0.001, 0.003))

