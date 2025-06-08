from Model.Patient import Patient
from mesa import Agent
from Simulation import RCCModel
import random
from Model.Enumerazioni import TherapyType
from Model.TumorCell import TumorCell
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.RCCModel import RCCModel

class ImmuneCell(Agent):
    def __init__(self, unique_id: int, model: "RCCModel", patient: Patient):
        super().__init__(unique_id, model)
        self.patient = patient
        self.activation_level = random.uniform(0.1, 0.8)
        self.target_priority = 0.5
        self.is_active = False
        self.exhaustion = 0
        self.isDead = False
        self.cytokines = {}
        self.x = random.uniform(0, model.grid_size)
        self.y = random.uniform(0, model.grid_size)

    def activate(self):
        self.activation_level = min(1.0, self.activation_level + 0.2)
        self.is_active = True


    def step(self):
        #cera tutte le cellule vicine e verifica lo spazio libero
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        free_spaces = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if free_spaces:
            new_pos = random.choice(free_spaces)
            self.model.grid.move_agent(self, new_pos)

        # integrazione della terapia
        if self.model.treatment:
            self.model.treatment.apply([self])
            self.activate()

        self.findTumorCellsAndAttackThem(neighbors)

        # Decremento attivazione della cellula immunitaria per evitare una sovraattivazione che può portare ad un comportamento sballato nella simulazione
        self.activation_level = max(0.1, self.activation_level - 0.01)
        if self.activation_level <= 0.2:
            self.is_active = False



    def findTumorCellsAndAttackThem(self, neighbors):
        # Cerca cellule tumorali vicine che non siano morte
        tumor_cells_nearby = [a for pos in neighbors for a in self.model.grid.get_cell_list_contents(pos)
                              if isinstance(a, TumorCell) and not a.isDead]
        if tumor_cells_nearby and random.random() < self.target_priority and self.is_active:
            target_cell = random.choice(tumor_cells_nearby)
            damage = self.activation_level * 10
            pdl1_evasion_factor = target_cell.pdl1_expression
            ici_mitigation_strength = 0.0
            if self.model.treatment:
                if self.model.treatment.therapy_type == TherapyType.PD1_INHIBITOR:
                    ici_mitigation_strength = self.model.treatment.dosage * 0.7
                elif self.model.treatment.therapy_type == TherapyType.CTLA4_INHIBITOR:
                    ici_mitigation_strength = self.model.treatment.dosage * 0.6
                elif self.model.treatment.therapy_type == TherapyType.COMBO:
                    ici_mitigation_strength = self.model.treatment.dosage * 0.9
            pdl1_evasion_factor = max(0.0, pdl1_evasion_factor - ici_mitigation_strength)
            effective_damage = damage * (1 - pdl1_evasion_factor)
            target_cell.take_damage(effective_damage)