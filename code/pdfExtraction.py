
import pdfplumber
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import glob
import seaborn as sns

# Imposta encoding UTF-8 per l'output (soluzione per Windows)
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Costruisci percorso relativo al file PDF
base_dir = os.path.dirname(__file__)  # cartella dove si trova il file .py
pdf_path = os.path.join(base_dir, "Santoni24_262_2024_Article_3719-new (4).pdf")

# Estrai testo dal PDF
tables = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        tables.append(page.extract_tables())

# Usa caratteri ASCII-compatibili invece di Unicode
pdf_data = {
    "Group": ["Overall", "Male", "Female", "Clear Cell", "Sarcomatoid", "Age<50", "Age>=70"], 
    "Median_OS_months": [38.7, 40.0, 38.7, 44.2, 34.4, 36.9, 31.4],
    "OS_CI_lower": [32.7, 32.7, 26.4, 35.8, 26.4, 29.0, 26.4],  # Intervalli di confidenza
    "OS_CI_upper": [44.2, 51.6, 41.0, 55.7, 59.0, 51.6, 49.2],
    "Median_PFS_months": [15.7, 15.7, 15.7, None, None, None, None],
    "HR": [None, None, 1.72, None, None, None, None],  # Hazard Ratio
    "p_value": [0.202, None, 0.008, 0.047, 0.001, None, None]
}

pdf_df = pd.DataFrame(pdf_data)
print("Dati di riferimento completi dal PDF:")
print(pdf_df)

# =============================================
# 2. CARICAMENTO E PREPROCESSING DEI CSV
# =============================================
def load_simulations(csv_files):
    sim_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        # Aggiungi colonne derivate (es. rapporto cellule immuni/tumorali)
        df["Immune_Ratio"] = df["Immune Cells (total)"] / (df["Tumor Cells (current)"] + 1e-6)
        # Converti OS e PFS in mesi
        df["OS_months"] = df["Overall Survival"] * 0.4  # 100% = 40 mesi
        df["PFS_months"] = df["Progression-Free Survival"] * 0.157  # 100% = 15.7 mesi
        sim_dfs.append(df)
    return pd.concat(sim_dfs, ignore_index=True)

# Verifica se esistono file CSV prima di caricarli
csv_files = glob.glob("simulation_*.csv")
if not csv_files:
    print("Attenzione: Nessun file simulation_*.csv trovato nella directory corrente")
    # Crea dati mock per il test
    np.random.seed(42)
    mock_data = {
        "Tumor Cells (current)": np.random.normal(1000, 200, 100),
        "Immune Cells (total)": np.random.normal(500, 100, 100),
        "Active Immune Cells": np.random.normal(200, 50, 100),
        "Overall Survival": np.random.normal(75, 15, 100),
        "Progression-Free Survival": np.random.normal(85, 20, 100)
    }
    sim_df = pd.DataFrame(mock_data)
    sim_df["Immune_Ratio"] = sim_df["Immune Cells (total)"] / (sim_df["Tumor Cells (current)"] + 1e-6)
    sim_df["OS_months"] = sim_df["Overall Survival"] * 0.4
    sim_df["PFS_months"] = sim_df["Progression-Free Survival"] * 0.157
    print("Usando dati mock per il test")
else:
    sim_df = load_simulations(csv_files)
    print(f"Caricati {len(csv_files)} file CSV")

# Aggiungi etichette di gruppo alla simulazione (es. mock per test)
sim_df["Group"] = np.random.choice(pdf_df["Group"], size=len(sim_df))  # Sostituisci con dati reali se disponibili

# =============================================
# 3. MODELLO DI ML PER OGNI GRUPPO
# =============================================
features = ["Tumor Cells (current)", "Immune Cells (total)", "Active Immune Cells", "Immune_Ratio"]
results = []

for group in pdf_df["Group"].unique():
    group_data = sim_df[sim_df["Group"] == group]
    if len(group_data) < 10:  # Skip gruppi con pochi dati
        print(f"Saltando gruppo '{group}': troppo pochi dati ({len(group_data)} campioni)")
        continue
    
    X = group_data[features]
    y = group_data["OS_months"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Group": group,
        "MAE": mae,
        "R2": r2,
        "OS_sim_median": group_data["OS_months"].median(),
        "OS_pdf_median": pdf_df[pdf_df["Group"] == group]["Median_OS_months"].values[0]
    })

if results:
    results_df = pd.DataFrame(results)
    print("\nRisultati per gruppo:")
    print(results_df)
else:
    print("\nNessun risultato disponibile - controllare i dati di input")

# =============================================
# 4. VISUALIZZAZIONI AVANZATE
# =============================================
if results:
    # Grafico 1: Confronto mediane OS per gruppo
    plt.figure(figsize=(12, 6))
    
    # Crea subplot per gestire meglio i dati
    x_positions = np.arange(len(pdf_df))
    width = 0.35
    
    plt.bar(x_positions - width/2, pdf_df["Median_OS_months"], width, 
            color="red", alpha=0.7, label="PDF")
    
    # Solo per i gruppi con risultati
    for i, group in enumerate(pdf_df["Group"]):
        if group in results_df["Group"].values:
            sim_median = results_df[results_df["Group"] == group]["OS_sim_median"].values[0]
            plt.bar(i + width/2, sim_median, width, 
                   color="blue", alpha=0.7, label="Simulazione" if i == 0 else "")
    
    # Aggiungi barre di errore per PDF
    plt.errorbar(
        x=x_positions,
        y=pdf_df["Median_OS_months"],
        yerr=[pdf_df["Median_OS_months"] - pdf_df["OS_CI_lower"], 
              pdf_df["OS_CI_upper"] - pdf_df["Median_OS_months"]],
        fmt="none", color="black", capsize=5, label="CI PDF"
    )
    
    plt.title("Confronto Mediana OS: Simulazione vs PDF (con intervalli di confidenza)")
    plt.xlabel("Gruppo")
    plt.ylabel("OS (mesi)")
    plt.xticks(x_positions, pdf_df["Group"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Grafico 2: Distribuzione OS per gruppo (semplificato)
    plt.figure(figsize=(14, 8))
    
    for i, group in enumerate(pdf_df["Group"].unique()):
        group_pdf = pdf_df[pdf_df["Group"] == group]
        group_sim = sim_df[sim_df["Group"] == group]
        
        if len(group_sim) == 0:
            continue
        
        plt.subplot(2, 4, i+1)
        
        # Simulazione (istogramma)
        plt.hist(group_sim["OS_months"], bins=20, alpha=0.6, color="blue", 
                label="Simulazione", density=True)
        
        # PDF (linea verticale con CI)
        median_val = group_pdf["Median_OS_months"].values[0]
        ci_lower = group_pdf["OS_CI_lower"].values[0]
        ci_upper = group_pdf["OS_CI_upper"].values[0]
        
        plt.axvline(x=median_val, color="red", linestyle="--", linewidth=2, label="PDF Mediana")
        plt.axvspan(ci_lower, ci_upper, alpha=0.2, color="red", label="CI PDF")
        
        plt.title(f"{group}")
        plt.xlabel("OS (mesi)")
        plt.ylabel("Densità")
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

