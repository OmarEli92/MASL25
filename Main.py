from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from Simulation.RCCModel import RCCModel
from LegendModule import LegendElement
def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5,
        "Layer": 0,
        "Color": "gray"
    }

    agent_type = agent.__class__.__name__

    if agent_type == "TumorCell":
        portrayal.update({"Color": "red", "Layer": 0, "r": 0.8})
    elif agent_type == "TCell":
        portrayal.update({"Color": "blue", "Layer": 1, "r": 0.6})
    elif agent_type == "NKCell":
        portrayal.update({"Color": "cyan", "Layer": 1, "r": 0.6})
    elif agent_type == "Macrophage":
        portrayal.update({"Color": "green", "Layer": 1, "r": 0.7})
    elif agent_type == "THelper":
        portrayal.update({"Color": "yellow", "Layer": 1, "r": 0.5})
    elif "ImmuneCell" in agent_type:
        portrayal.update({"Color": "lightblue", "Layer": 1, "r": 0.5})

    return portrayal

model_params = {
    "patient_params": {"sex": "male", "age": 60, "bmi": 25.0},
    "therapy_params": {"therapy_type": "PD1_INHIBITOR", "dosage": 0.5},
    "initial_tumor_cells": 50,
    "initial_immune_cells": 30,
    "grid_size": 30,
    "seed": 42
}

grid = CanvasGrid(agent_portrayal, 30, 30, 500, 500)

chart_cells = ChartModule([
    {"Label": "Tumor Cells (current)", "Color": "Red"},
    {"Label": "Immune Cells (total)", "Color": "Blue"},
    {"Label": "Active Immune Cells", "Color": "Cyan"},
    {"Label": "Destroyed Tumor Cells", "Color": "Black"}
])

chart_biomarkers = ChartModule([
    {"Label": "Average Cancer Cell PD-L1", "Color": "Purple"},
    {"Label": "Average TCell Activation", "Color": "Cyan"},
    {"Label": "Average TCell Exhaustion", "Color": "Magenta"}
])

chart_survival = ChartModule([
    {"Label": "Overall Survival", "Color": "Green"},
    {"Label": "Progression-Free Survival", "Color": "Orange"}
])

legend = LegendElement()
server = ModularServer(
    RCCModel,
    [grid, chart_cells, chart_biomarkers, chart_survival, legend],
    "Simulazione RCC - Mesa 2.1",
    model_params
)

if __name__ == "__main__":
    server.port = 8521
    server.launch()
