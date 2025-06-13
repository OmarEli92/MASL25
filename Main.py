import io
import base64
import matplotlib.pyplot as plt
from mesa.visualization.UserParam import Slider, NumberInput, Choice, Checkbox
from mesa.visualization.modules import CanvasGrid, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from Simulation.RCCModel import RCCModel
from LegendModule import LegendElement

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

class ComprehensiveDashboard(TextElement):    
    
    def render(self, model):
        try:
            tumor_cells = getattr(model, 'get_total_tumor_cells', lambda: 0)()
            immune_cells = getattr(model, 'get_total_immune_cells', lambda: 0)()
            status = getattr(model, 'get_simulation_status', lambda: "RUNNING")()
            step = model.schedule.steps if hasattr(model, 'schedule') else 0
            
            total_cells = tumor_cells + immune_cells
            ratio = tumor_cells / max(immune_cells, 1) if immune_cells > 0 else float('inf')
            
            status_color = {
                "RUNNING": "green",
                "REMISSION": "blue", 
                "TUMOR_VICTORY": "red",
                "MAX_STEPS": "orange"
            }.get(status, "gray")
            
            html = f"""
            <div style="font-family: Arial, sans-serif; margin: 10px;">
                
                <div style="background: linear-gradient(135deg, #85a1f5 0%, #9a75c1 100%);
                            color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; text-align: center;">RCC Simulation Dashboard</h2>
                    <p style="margin: 5px 0 0 0; text-align: center;">
                        Step {step} | Status:
                        <span style="color: {status_color}; font-weight: bold;">{status}</span>
                    </p>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                           gap: 15px; margin-bottom: 20px;">
                    
                    <div style="background: #ffe6e6; border: 2px solid #ff6b6b; 
                               border-radius: 8px; padding: 15px; text-align: center;">
                        <h3 style="margin: 0; color: #c92a2a;">Tumor Cells</h3>
                        <div style="font-size: 24px; font-weight: bold; color: #c92a2a;">{tumor_cells}</div>
                    </div>
                    
                    <div style="background: #e6f3ff; border: 2px solid #4dabf7; 
                               border-radius: 8px; padding: 15px; text-align: center;">
                        <h3 style="margin: 0; color: #1971c2;">Immune Cells</h3>
                        <div style="font-size: 24px; font-weight: bold; color: #1971c2;">{immune_cells}</div>
                    </div>
                    
                    <div style="background: #f0f9f0; border: 2px solid #51cf66; 
                               border-radius: 8px; padding: 15px; text-align: center;">
                        <h3 style="margin: 0; color: #2f9e44;">Total Cells</h3>
                        <div style="font-size: 24px; font-weight: bold; color: #2f9e44;">{total_cells}</div>
                    </div>
                    
                    <div style="background: #fff4e6; border: 2px solid #ff922b; 
                               border-radius: 8px; padding: 15px; text-align: center;">
                        <h3 style="margin: 0; color: #d9480f;">Tumor/Immune Cells Ratio</h3>
                        <div style="font-size: 24px; font-weight: bold; color: {'red' if ratio >= 3.0 else '#d9480f'};">
                            {ratio:.2f}
                        </div>
                    </div>
                </div>
            """
            
            if hasattr(model, 'datacollector'):
                try:
                    df = model.datacollector.get_model_vars_dataframe()      

                    if not df.empty:
                        latest = df.iloc[-1]
                        html += """
                        <div style="background: #f8f9fa; border: 1px solid #dee2e6;
                                border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                            <h3 style="margin-top: 0; color: #495057;">Data Collector Values</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">
                        """

                        for col, val in latest.items():
                            if isinstance(val, (int, float)):
                                name = col.lower()
                                if 'tumor' in name or 'cancer' in name:
                                    bg, text = "#ffebee", "#c62828"
                                elif any(k in name for k in ['immune', 'tcell', 'nk']):
                                    bg, text = "#e3f2fd", "#1565c0"
                                elif any(k in name for k in ['pdl1', 'biomarker']):
                                    bg, text = "#f3e5f5", "#7b1fa2"
                                else:
                                    bg, text = "#f5f5f5", "#424242"

                                formatted = f"{val:.3f}" if isinstance(val, float) else str(val)
                                html += f"""
                                <div style="background: {bg}; padding: 8px; border-radius: 4px;
                                        border-left: 3px solid {text};">
                                    <strong style="color: {text};">{col}:</strong><br>
                                    <span style="font-size: 14px; color: {text};">{formatted}</span>
                                </div>
                                """

                        html += "</div></div>"

                        if len(df) > 1:
                            avg_prefixes = [
                                "average cancer cell pd-l1",
                                "average tcell activation",
                                "average tcell exhaustion"
                            ]
                            all_key_cols = [c for c in df.columns
                                            if any(k in c.lower() for k in
                                                ['tumor', 'immune', 'cancer', 'tcell', 'pdl1'])]
                            avg_cols = [c for c in all_key_cols
                                        if any(c.lower().startswith(k) for k in avg_prefixes)]
                            non_avg_cols = [c for c in all_key_cols if c not in avg_cols]

                            surv_cols = [
                                c for c in df.columns
                                if c.lower().strip() in ["overall survival", "progression-free survival"]
                            ]

                            def make_line_plot(cols, title):
                                fig, ax = plt.subplots()
                                for col in cols:
                                    ax.plot(df.index, df[col], label=col)
                                ax.set_xlabel("Time Step")
                                ax.set_ylabel("Value")
                                ax.set_title(title)
                                ax.legend(fontsize='small', loc='upper left')
                                ax.grid(True)
                                buf = io.BytesIO()
                                plt.tight_layout()
                                fig.savefig(buf, format='png')
                                plt.close(fig)
                                buf.seek(0)
                                return base64.b64encode(buf.read()).decode('utf-8')

                            if non_avg_cols:
                                img1 = make_line_plot(non_avg_cols, "")
                                html += f"""
                                <div style="background: #f8f9fa; border: 1px solid #dee2e6;
                                            border-radius: 8px; padding: 15px; margin-bottom: 15px; text-align: center;">
                                <h3 style="margin-top: 0; color: #495057;">Cells Metrics Over Time<br></h3>
                                <img src="data:image/png;base64,{img1}"
                                    style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />
                                </div>
                                """

                            if avg_cols:
                                img2 = make_line_plot(avg_cols, "")
                                html += f"""
                                <div style="background: #f8f9fa; border: 1px solid #dee2e6;
                                            border-radius: 8px; padding: 15px; margin-bottom: 15px; text-align: center;">
                                <h3 style="margin-top: 0; color: #495057;">Averages Over Time</h3>
                                <img src="data:image/png;base64,{img2}"
                                    style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />
                                </div>
                                """

                            if surv_cols:
                                img3 = make_line_plot(surv_cols, "")
                                html += f"""
                                <div style="background: #f8f9fa; border: 1px solid #dee2e6;
                                            border-radius: 8px; padding: 15px; margin-bottom: 15px; text-align: center;">
                                <h3 style="margin-top: 0; color: #495057;">Survival Metrics Over Time</h3>
                                <img src="data:image/png;base64,{img3}"
                                    style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />
                                </div>
                                """


                except Exception as e:
                    html += f"""
                    <div style="background: #fff3cd; border: 1px solid #ffeaa7;
                            border-radius: 8px; padding: 15px; margin: 10px 0;">
                        <strong>DataCollector Warning:</strong> {e}
                    </div>
                    """

            html += """
            <div style="background: #e9ecef; border: 1px solid #ced4da; 
                       border-radius: 8px; padding: 15px; margin-top: 15px;">
                <h3 style="margin-top: 0; color: #495057;">Termination Conditions</h3>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>REMISSION:</strong> Tumor cells ≤ 5 (Patient recovers)</li>
                    <li><strong>TUMOR VICTORY:</strong> Tumor/Immune ratio ≥ 3.0 (Treatment fails)</li>
                    <li><strong>MAX STEPS:</strong> Simulation reaches 50 steps (Timeout)</li>
                </ul>
            </div>
            
            </div>
            """
            
            return html
            
        except Exception as e:
            return f"""
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; 
                       border-radius: 8px; padding: 15px; color: #721c24;">
                <strong>Dashboard Error:</strong> {e}
            </div>
            """

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

grid = CanvasGrid(agent_portrayal, 30, 30, 500, 500)
dashboard = ComprehensiveDashboard()
legend = LegendElement()

server = ModularServer(
    model_wrapper,
    [grid, legend, dashboard],
    "RCC Simulation",
    model_params
)

if __name__ == "__main__":
    server.port = 8521  
    server.launch()