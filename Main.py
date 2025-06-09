from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.UserParam import Slider, NumberInput, Choice, Checkbox
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from Simulation.RCCModel import RCCModel
from LegendModule import LegendElement
import random

# 1) Agent portrayal for grid display
def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5, "Layer": 0, "Color": "gray"}
    t = agent.__class__.__name__
    if t == "TumorCell":
        portrayal.update({"Color": "red", "r": 0.8, "Layer": 1})
    elif t == "TCell":
        portrayal.update({"Color": "blue", "r": 0.6, "Layer": 1})
    elif t == "NKCell":
        portrayal.update({"Color": "cyan", "r": 0.6, "Layer": 1})
    elif t == "Macrophage":
        portrayal.update({"Color": "green", "r": 0.7, "Layer": 1})
    elif t == "THelper":
        portrayal.update({"Color": "yellow", "r": 0.5, "Layer": 1})
    elif "ImmuneCell" in t:
        portrayal.update({"Color": "lightblue", "r": 0.5, "Layer": 1})
    return portrayal

# 2) Interactive model parameters
model_params = {
    "initial_tumor_cells": Slider("Initial Tumor Cells", 50, 0, 200, 1),
    "initial_immune_cells": Slider("Initial Immune Cells", 30, 0, 100, 1),
    "grid_size": Slider("Grid Size", 30, 10, 100, 1),
    "sex": Choice("Patient Sex", choices=["male", "female"], value="male"),
    "age": NumberInput("Patient Age", 60),
    "bmi": NumberInput("Patient BMI", 25.0),
    "use_treatment": Checkbox("Enable Treatment", True),
    "therapy_type": Choice("Therapy Type", choices=["NONE", "PD1_INHIBITOR", "CTLA4_INHIBITOR"], value="PD1_INHIBITOR"),
    "dosage": Slider("Dosage (mg/kg)", 0.5, 0.0, 5.0, 0.1),
    "seed": NumberInput("Random Seed", 42)
}

# 3) Wrapper to map UI params to RCCModel
def model_wrapper(**kwargs):
    patient_params = {"sex": kwargs.pop("sex"), "age": kwargs.pop("age"), "bmi": kwargs.pop("bmi")}  
    therapy_params = None
    if kwargs.pop("use_treatment"):
        therapy_params = {"therapy_type": kwargs.pop("therapy_type"), "dosage": kwargs.pop("dosage")}  
    return RCCModel(
        patient_params=patient_params,
        therapy_params=therapy_params,
        initial_tumor_cells=kwargs.pop("initial_tumor_cells"),
        initial_immune_cells=kwargs.pop("initial_immune_cells"),
        grid_size=kwargs.pop("grid_size"),
        seed=kwargs.pop("seed")
    )

# 4) Visualization modules
# CanvasGrid: show agents
grid = CanvasGrid(agent_portrayal, 30, 30, 500, 500)

# Charts: include 'Key' matching DataCollector columns exactly
chart_cells = ChartModule([
    {"Label": "Tumor Cells (current)", "Color": "#d73027"},
    {"Label": "Immune Cells (total)", "Color": "#4575b4"},
    {"Label": "Active Immune Cells",  "Color": "#1a9850"},
    {"Label": "Destroyed Tumor Cells","Color": "#000000"},
], data_collector_name="datacollector", canvas_width=500, canvas_height=200)

chart_biomarkers = ChartModule([
    {"Label": "Average Cancer Cell PD-L1", "Color": "#a50026"},
    {"Label": "Average TCell Activation",  "Color": "#2c7bb6"},
    {"Label": "Average TCell Exhaustion",  "Color": "#f46d43"},
], data_collector_name="datacollector", canvas_width=500, canvas_height=200)

chart_survival = ChartModule([
    {"Label": "Overall Survival",           "Color": "#abdda4"},
    {"Label": "Progression-Free Survival", "Color": "#fc8d59"},
    {"Label": "Time Step",                 "Color": "#ffffbf"},
], data_collector_name="datacollector", canvas_width=500, canvas_height=200)

# Legend
legend = LegendElement()

# 5) Server setup
server = ModularServer(
    model_wrapper,
    [grid, chart_cells, chart_biomarkers, chart_survival, legend],
    "Simulazione RCC - Mesa 2.x",
    model_params
)

if __name__ == "__main__":
    server.port = 8521
    # Debug: show exactly what columns exist
    df = model_wrapper(**{
        "initial_tumor_cells": 50,
        "initial_immune_cells": 30,
        "grid_size": 30,
        "sex": "male", "age": 60, "bmi": 25.0,
        "use_treatment": True,
        "therapy_type": "PD1_INHIBITOR", "dosage": 0.5,
        "seed": 42
    }).datacollector.get_model_vars_dataframe()
    print("=== MODEL REPORTER COLUMNS ===")
    print(df.columns.tolist())
    server.launch()
