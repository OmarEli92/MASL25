import random

from mesa import Agent

from Model.Patient import Patient
from Model.Enumerazioni import TumorHistology
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.RCCModel import RCCModel

class TumorCell(Agent):
    def __init__(self, unique_id: int, model: "RCCModel", histology: TumorHistology, patient: Patient):
        super().__init__(unique_id, model)
        self.histology = histology
        self.patient = patient
        self.health = 100
        self.isDead = False
        self.pdl1_expression = self._set_pdl1()
        self.resistance = 0.1
        self.x = random.uniform(0, model.grid_size)
        self.y = random.uniform(0, model.grid_size)
        # a seconda dell'istologia della cellula il rate di proliferazione sarà ovviamente diverso
        if self.histology == TumorHistology.PAPILLARY:
            # Carcinoma Papillare tende ad essere più proliferativo
            self.proliferation_rate = random.uniform(0.03, 0.05)
        elif self.histology == TumorHistology.CHROMOPHOBE:
            # Carcinoma Cromofobo tende ad essere solitamente meno proliferativo viste le prognosi leggermente migliori
            self.proliferation_rate = random.uniform(0.005, 0.02)
        else:  # CLEAR_CELL
            self.proliferation_rate = random.uniform(0.01, 0.05)

    def _set_pdl1(self) -> float:
        """Abbiamo tenuto conto dell'istologia della cellula tumorale ma anche dei fattori del paziente
        come sesso ed età e BMI """
        if self.histology == TumorHistology.CLEAR_CELL:
            base_pdl1 = 0.5
        elif self.histology == TumorHistology.PAPILLARY:
            base_pdl1 = 0.7
        else:                    # Chromophobe
            base_pdl1 = 0.3

        patient_factor_multiplier = 1.0
        if self.patient.sex == "female" and self.patient.age < 50:
            patient_factor_multiplier *= 1.5
        # in questo caso si verifica il paradosso dell'obesità
        if self.patient.bmi > 30:
            patient_factor_multiplier *= 1.2
        elif self.patient.bmi < 20:
            patient_factor_multiplier *= 0.8
        if self.patient.age > 70:
            patient_factor_multiplier *= 1.1
        elif self.patient.age < 30:
            patient_factor_multiplier *= 0.9

        return max(0.0, min(1.0, base_pdl1 * patient_factor_multiplier))

    def take_damage(self, damage: float):
        self.health -= damage
        if self.health <= 0:
            self.isDead = True
            # Rimuovi immediatamente dalla griglia
            if hasattr(self, 'pos') and self.pos is not None:
                self.model.grid.remove_agent(self)

    def step(self):
        if self.isDead:
            # Se è morta, rimuovila dallo scheduler
            self.model.schedule.remove(self)
            return
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        free_spaces = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if free_spaces:
            new_pos = random.choice(free_spaces)
            self.model.grid.move_agent(self, new_pos)
        # Proliferazione
        if random.random() < self.proliferation_rate:
            new_cell = TumorCell(self.model.get_next_id(), self.model, self.histology, self.patient)
            self.model.add_agent_to_grid(new_cell)
        if self.health <= 0:
            self.isDead = True
