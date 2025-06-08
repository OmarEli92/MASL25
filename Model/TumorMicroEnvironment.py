from Model.Patient import Patient
import random


class TumorMicroenvironment:
    def __init__(self, patient: Patient):
        self.patient = patient
        self.oxygen_level = random.uniform(0.8, 1.2)
        self.nutrient_availability = random.uniform(0.7, 1.3)

    def update(self):
        self.oxygen_level = max(0.1, min(1.5, self.oxygen_level + random.uniform(-0.05, 0.05)))
        self.nutrient_availability = max(0.1, min(1.5, self.nutrient_availability + random.uniform(-0.03, 0.03)))