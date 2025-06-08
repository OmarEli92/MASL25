from Model.ImmuneCell import ImmuneCell
from Model.Patient import Patient

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.RCCModel import RCCModel
class Neutrophil(ImmuneCell):
    def __init__(self,unique_id: int, model: "RCCModel", patient: Patient):
        super().__init__(unique_id, model, patient)
        self.lifespan = 24  # sono le ore di vita
    def release_nets(self):
        return {"inflammation_increase": 0.1}