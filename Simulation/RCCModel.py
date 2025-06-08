from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from typing import Dict, Optional
from Model.Patient import Patient
from Model.Treatment import Treatment
from Model.TumorCell import TumorCell
from Model.ImmuneCell import ImmuneCell
from Model.Macrophage import Macrophage
from Model.THelper import THelper
from Model.NKCell import NKCell
from Model.TCell import TCell
from Model.Enumerazioni import TherapyType, TumorHistology, TCellSubtype, MacrophagePhenotype, THelperSubtype
from Model.TumorMicroEnvironment import TumorMicroenvironment
import random
import os
import datetime
class RCCModel(Model):
    """
    Il modello che rappresenta il contesto del RCC si occupa di inizializzare le cellule,
    portare avanti la simulazione e calcolare l'OS (overall survival) e il PFS (Progression-Free Survival)
    """

    def __init__(
            self,
            patient_params: Dict,
            therapy_params: Optional[Dict] = None,
            initial_tumor_cells: int = 100,
            initial_immune_cells: int = 50,
            seed: Optional[int] = None,
            grid_size: int = 100
    ):
        super().__init__()

        if seed is not None:
            random.seed(seed)
        self.patient = Patient(**patient_params)
        self.microenvironment = TumorMicroenvironment(self.patient)
        self.time_steps = 0
        self.grid_size = grid_size
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(grid_size, grid_size, torus=False)
        self.next_id = 0

        # Inizializza il trattamento scelto e il rispettivo dosaggio
        self.treatment = (
            Treatment(TherapyType[therapy_params["therapy_type"]], therapy_params["dosage"])
            if therapy_params and "therapy_type" in therapy_params and "dosage" in therapy_params
            else None
        )

        # Inizializza le cellule
        self.initial_tumor_cells_at_start = initial_tumor_cells
        self._initialize_cells(initial_tumor_cells, initial_immune_cells)

        self.datacollector = DataCollector(
            model_reporters={
                "Tumor Cells (current)": lambda m: m.get_total_tumor_cells(),
                "Immune Cells (total)": lambda m: m.get_total_immune_cells(),
                "Active Immune Cells": lambda m: m.get_active_immune_cells(),
                "Destroyed Tumor Cells": lambda m: m.get_destroyed_tumor_cells(),
                "Overall Survival": lambda m: m.get_overall_survival(),
                "Progression-Free Survival": lambda m: m.get_progression_free_survival(),
                "Average Cancer Cell PD-L1": lambda m: m.compute_avg_pdl1(),
                "Average TCell Activation": lambda m: m.compute_avg_tcell_activation(),
                "Average TCell Exhaustion": lambda m: m.compute_avg_tcell_exhaustion(),
                "Time Step": lambda m: m.time_steps
            },
            agent_reporters={
                "Health": lambda a: getattr(a, 'health', None),
                "Activation": lambda a: getattr(a, 'activation_level', None),
                "Is_Dead": lambda a: getattr(a, 'isDead', False)
            }
        )
        self.datacollector.collect(self)

    def get_next_id(self):
        current_id = self.next_id
        self.next_id += 1
        return current_id

    def add_agent_to_grid(self, agent):
        x = random.randrange(self.grid.width)
        y = random.randrange(self.grid.height)
        self.grid.place_agent(agent, (x, y))
        self.schedule.add(agent)

    def _initialize_cells(self, tumor_cells: int, immune_cells: int):
        """Inizializza le cellule tumorali e immunitarie"""
        for _ in range(tumor_cells):
            histology = random.choice(list(TumorHistology))
            cell = TumorCell(self.get_next_id(), self, histology, self.patient)
            self.add_agent_to_grid(cell)
        for _ in range(immune_cells):
            cell_type = random.choice(["T", "NK", "Macrophage", "THelper"])
            if cell_type == "T":
                subtype = random.choice(list(TCellSubtype))
                cell = TCell(self.get_next_id(), self, self.patient, subtype)
            elif cell_type == "NK":
                cell = NKCell(self.get_next_id(), self, self.patient)
            elif cell_type == "Macrophage":
                phenotype = random.choice(list(MacrophagePhenotype))
                cell = Macrophage(self.get_next_id(), self, self.patient, phenotype)
            elif cell_type == "THelper":
                subtype = random.choice(list(THelperSubtype))
                cell = THelper(self.get_next_id(), self, self.patient, subtype)
            self.add_agent_to_grid(cell)



    def _compute_os(self) -> float:
        """Il metodo restituisce l'Overall survival basandosi sulle cellule tumorali rimaste"""
        tumor_cells = sum(
            1 for agent in self.schedule.agents
            if isinstance(agent, TumorCell) and not getattr(agent, 'isDead', False)
        )

        if self.initial_tumor_cells_at_start == 0:
            return 100.0

        os_percentage = max(0.0, 100.0 * (1 - (tumor_cells / self.initial_tumor_cells_at_start)))
        return os_percentage

    def _compute_pfs(self) -> float:
        """Calcola il Progression-Free Survival"""
        tumor_cells = sum(
            1 for agent in self.schedule.agents
            if isinstance(agent, TumorCell) and not getattr(agent, 'isDead', False)
        )

        if self.initial_tumor_cells_at_start == 0:
            return 100.0

        if tumor_cells <= self.initial_tumor_cells_at_start:
            return 100.0
        else:
            progression_ratio = (tumor_cells - self.initial_tumor_cells_at_start) / self.initial_tumor_cells_at_start
            pfs_percentage = max(0.0, 100.0 * (1.0 - progression_ratio))
            return pfs_percentage

    def compute_avg_pdl1(self) -> float:
        """Calcola il PD-L1 medio delle cellule tumorali"""
        values = [
            getattr(agent, "pdl1_expression", 0)
            for agent in self.schedule.agents
            if isinstance(agent, TumorCell) and not getattr(agent, 'isDead', False)
        ]
        return sum(values) / len(values) if values else 0.0

    def compute_avg_tcell_activation(self) -> float:
        """Calcola l'attivazione media delle cellule T"""
        activation_levels = [
            getattr(agent, 'activation_level', 0)
            for agent in self.schedule.agents
            if isinstance(agent, TCell) and not getattr(agent, 'isDead', False)
        ]
        return sum(activation_levels) / len(activation_levels) if activation_levels else 0.0

    def compute_avg_tcell_exhaustion(self) -> float:
        """Calcola l'esaurimento medio delle cellule T"""
        exhaustion_levels = [
            getattr(agent, 'exhaustion', 0)
            for agent in self.schedule.agents
            if isinstance(agent, TCell) and not getattr(agent, 'isDead', False)
        ]
        return sum(exhaustion_levels) / len(exhaustion_levels) if exhaustion_levels else 0.0

    def remove_dead_agents(self):
        agents_to_remove = []
        for agent in self.schedule.agents:
            if hasattr(agent, 'isDead') and getattr(agent, 'isDead', False):
                agents_to_remove.append(agent)
        for agent in agents_to_remove:
            if hasattr(agent, 'pos') and agent.pos is not None:
                self.grid.remove_agent(agent)
            self.schedule.remove(agent)

    def step(self):
        self.time_steps += 1
        self.microenvironment.update()
        if self.treatment:
            immune_cells = [
                a for a in self.schedule.agents
                if isinstance(a, ImmuneCell) and not getattr(a, 'isDead', False)
            ]
            self.treatment.apply(immune_cells)
        self.schedule.step()
        self.remove_dead_agents()
        self.datacollector.collect(self)
        if self.time_steps == 75:
            self.save_data_to_csv()

    def save_data_to_csv(self, filename_prefix="simulation_data"):
        """Salva i dati raccolti su CSV"""
        model_df = self.datacollector.get_model_vars_dataframe()
        agent_df = self.datacollector.get_agent_vars_dataframe()
        output_dir = "simulation_outputs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        model_df.to_csv(os.path.join(output_dir, f"{filename}_model.csv"), index=False)
        agent_df.to_csv(os.path.join(output_dir, f"{filename}_agents.csv"), index=False)



    def get_total_tumor_cells(self):
        return sum(1 for agent in self.schedule.agents if isinstance(agent, TumorCell) and not agent.isDead)

    def get_total_immune_cells(self):
        return sum(1 for agent in self.schedule.agents if 'ImmuneCell' in type(agent).__name__)

    def get_active_immune_cells(self):
        return sum(1 for agent in self.schedule.agents if hasattr(agent, "is_active") and agent.is_active)

    def get_destroyed_tumor_cells(self):
        return self.initial_tumor_cells_at_start - self.get_total_tumor_cells()

    def get_overall_survival(self):
        return 1 if self.get_total_tumor_cells() > 0 else 0

    def get_progression_free_survival(self):
        return 1 if self.get_total_tumor_cells() <= self.initial_tumor_cells_at_start else 0


