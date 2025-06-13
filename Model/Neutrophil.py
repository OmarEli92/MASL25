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
    
    def step(self):
        result = self.release_nets()
        # Puoi fare qualcosa con `result`, ad esempio aggiornare l'infiammazione nel paziente
        self.lifespan -= 1
        if self.lifespan <= 0:
            self.model.schedule.remove(self)  # Se usi un scheduler tipo BaseScheduler
