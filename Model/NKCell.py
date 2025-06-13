from Model.ImmuneCell import ImmuneCell
from Model.Patient import Patient
from Model.TumorCell import TumorCell
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.RCCModel import RCCModel

class NKCell(ImmuneCell):
    def __init__(self, unique_id: int, model: "RCCModel", patient: Patient):
        super().__init__(unique_id, model, patient)
        self.activation = 0.6

    def attack(self, cancer_cell: TumorCell) -> bool:
        success_prob = (self.activation * 1.2) - (cancer_cell.pdl1_expression * 0.5)
        return random.random() < success_prob
    
    def step(self):
        super().step()        
        tumor_cells = [agent for agent in self.model.schedule.agents
                    if isinstance(agent, TumorCell) and not agent.isDead]
        if tumor_cells:
            target = random.choice(tumor_cells)
            if self.attack(target):
                target.take_damage(35)  
