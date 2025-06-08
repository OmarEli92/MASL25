from mesa import Agent
class BloodVessel(Agent):
    def __init__(self):
        self.oxygen_level = 0.8
        self.permeability = 0.5

    def update(self, inflammation: float):
        self.permeability = min(1.0, 0.5 + inflammation * 0.2)
        self.oxygen_level = max(0.3, 0.8 - inflammation * 0.1)