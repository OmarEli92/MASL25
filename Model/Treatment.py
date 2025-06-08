from Model.Enumerazioni import TherapyType
from Model.ImmuneCell import ImmuneCell
from Model.TCell import TCell
from typing import List
import random


class Treatment:
    def __init__(self, therapy_type: TherapyType, dosage: float):
        self.therapy_type = therapy_type
        self.dosage = dosage

    def apply(self, immune_cells: List[ImmuneCell]):
        """
        Applicazione della terapia: immunoterapia attiva le cellule immunitarie,
        inibitori (ICI) mitigano l'esaurimento e potenziano l'attivazione.
        La combo combina entrambi gli effetti.
        """
        for cell in immune_cells:
            if self.therapy_type == TherapyType.IMMUNO:
                # Attiva le cellule con probabilità proporzionale al dosaggio
                if random.random() < self.dosage * 0.1:
                    cell.activate()
            elif self.therapy_type == TherapyType.PD1_INHIBITOR or self.therapy_type == TherapyType.CTLA4_INHIBITOR:
                # Riduce l'esaurimento e aumenta l'attivazione direttamente
                cell.activation_level = min(1.0, cell.activation_level + self.dosage * 0.3)
                cell.exhaustion = max(0.0, cell.exhaustion - self.dosage * 0.4)
                cell.is_active = True
            elif self.therapy_type == TherapyType.COMBO:
                # Combina effetto immuno + inibitori
                if random.random() < self.dosage * 0.1:
                    cell.activate()
                cell.activation_level = min(1.0, cell.activation_level + self.dosage * 0.2)
                cell.exhaustion = max(0.0, cell.exhaustion - self.dosage * 0.3)
                cell.is_active = True
