from Model.Patient import Patient
from Model.ImmuneCell import ImmuneCell
from Model.Enumerazioni import THelperSubtype
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.RCCModel import RCCModel
class THelper(ImmuneCell):
    def __init__(self,unique_id: int, model: "RCCModel", patient: Patient, subtype: THelperSubtype):
        super().__init__(unique_id, model, patient)
        self.helper_subtype = subtype


    def secrete_cytokines(self):
        if not hasattr(self, 'cytokines'):
            self.cytokines = {"IFN_gamma": 0, "IL_4": 0, "IL_17": 0}
        secretion_amount = 2 * self.activation_level
        if self.helper_subtype == THelperSubtype.TH1:
            self.cytokines["IFN_gamma"] = min(10.0,
                                              self.cytokines.get("IFN_gamma", 0) + secretion_amount)  # Cap at 10.0
        elif self.helper_subtype == THelperSubtype.TH2:
            self.cytokines["IL_4"] = min(10.0, self.cytokines.get("IL_4", 0) + secretion_amount)
        elif self.helper_subtype == THelperSubtype.TH17:
            self.cytokines["IL_17"] = min(10.0, self.cytokines.get("IL_17", 0) + secretion_amount)

        for key in self.cytokines: # voglio evitare che crescano troppo in numero durante la simulazione
            self.cytokines[key] = max(0, self.cytokines[key] - 0.5)

    def step(self):
        super().step()
        self.secrete_cytokines()
        if self.is_active:
            other_immune_cells = [a for a in self.model.schedule.agents if isinstance(a, ImmuneCell) and a != self]
            if other_immune_cells:  # Check if the list is not empty
                cell_to_activate = random.choice(other_immune_cells)
                cell_to_activate.activate()
