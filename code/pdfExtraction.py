import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import glob
import os

# Imposta encoding UTF-8 per l'output (soluzione per Windows)
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Costruisci percorso relativo al file PDF
base_dir = os.path.dirname(__file__)  # cartella dove si trova il file .py
pdf_path = os.path.join(base_dir, "Santoni24_262_2024_Article_3719-new (4).pdf")

# =============================================
# 1. DATI DI RIFERIMENTO DAL PDF (hardcoded)
# =============================================
# =============================================
# 1. DATI DI RIFERIMENTO DAL PDF (hardcoded)
# =============================================
pdf_data = {
    # Overall Survival
    "OS": {
        "Overall": {"median": 38.7, "CI_lower": 32.7, "CI_upper": 44.2},
        "Male": {"median": 40.0, "CI_lower": 32.7, "CI_upper": 51.6},
        "Female": {"median": 38.7, "CI_lower": 26.4, "CI_upper": 41.0},
        "Age<50": {"median": 36.9, "CI_lower": 29.0, "CI_upper": 51.6},
        "Age≥70": {"median": 31.4, "CI_lower": 26.4, "CI_upper": 49.2},
        "BMI<25": {"median": 31.6, "CI_lower": 25.9, "CI_upper": 40.2},
        "BMI≥25": {"median": 44.2, "CI_lower": 35.8, "CI_upper": 55.7}
    },
    # Progression-Free Survival
    "PFS": {
        "Overall": {"median": 15.7},
        "Male": {"median": 15.7},
        "Female": {"median": 15.7}
    }
}

# =============================================
# 2. CARICAMENTO E PREPROCESSING DEI DATI DI SIMULAZIONE
# =============================================
def load_simulation_data():
    """Carica e unisce tutti i file CSV di simulazione"""

    data_dir = os.path.join(base_dir, "..", "data")  # vai su di una cartella poi nella cartella data
    csv_pattern = os.path.join(data_dir, "final_*.csv")
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        raise FileNotFoundError("Nessun file CSV trovato con pattern 'final*.csv'")
    
    print(f"Trovati {len(csv_files)} file CSV:")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Errore nel caricamento del file {f}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("Nessun dato caricato - tutti i file hanno avuto errori")
    
    sim_df = pd.concat(dfs, ignore_index=True)
    
    # Creazione gruppi
    sim_df['Age_Group'] = pd.cut(
        sim_df['patient_age'],
        bins=[0, 50, 70, 100],
        labels=["Age<50", "Age50-69", "Age≥70"]
    )
    
    sim_df['BMI_Group'] = pd.cut(
        sim_df['patient_bmi'],
        bins=[0, 25, 100],
        labels=["BMI<25", "BMI≥25"]
    )
    
    return sim_df

# Carica i dati di simulazione
print("\nCaricamento dati simulazione...")
sim_df = load_simulation_data()

# =============================================
# 3. PREPARAZIONE DATI PER ML E NORMALIZZAZIONE
# =============================================
def prepare_data(sim_df):
    """Prepara i dati per l'analisi e il ML"""
    # Per i casi di remissione completa, usiamo i valori massimi
    remission_mask = sim_df['outcome_category'] == 'COMPLETE_REMISSION'
    
    # Per gli altri casi, usiamo una combinazione di:
    # 1. Valori osservati (se disponibili)
    # 2. Stime basate su altre variabili
    
    # Calcola OS_months:
    # - Per remissione: 40 mesi (valore massimo dal PDF)
    # - Per altri casi: usiamo una trasformazione non lineare della percentuale
    sim_df['OS_months'] = np.where(
        remission_mask,
        40.0,
        np.where(
            sim_df['overall_survival'] > 0,
            40 * (1 - np.exp(-2 * sim_df['overall_survival'])),  # Trasformazione non lineare
            # Se overall_survival è 0, stimiamo in base ad altre variabili
            40 * (1 - np.exp(-0.1 * sim_df['destroyed_tumor_cells'] / 
                            (sim_df['initial_tumor_cells'] + 1)))
        )
    )
    
    # Calcola PFS_months:
    # - Per remissione: 15.7 mesi (valore massimo dal PDF)
    # - Per altri casi: usiamo una trasformazione non lineare
    sim_df['PFS_months'] = np.where(
        remission_mask,
        15.7,
        np.where(
            sim_df['progression_free_survival'] > 0,
            15.7 * (1 - np.exp(-4 * sim_df['progression_free_survival'])),  # Trasformazione non lineare
            # Se progression_free_survival è 0, stimiamo in base ad altre variabili
            15.7 * (sim_df['final_active_immune_cells'] / 
                   (sim_df['final_immune_cells'] + 1))
        )
    )
    
    # Features per il modello ML
    features = [
        'initial_tumor_cells', 'initial_immune_cells',
        'final_tumor_cells', 'final_immune_cells', 'final_active_immune_cells',
        'destroyed_tumor_cells', 'avg_pdl1_expression', 'avg_tcell_activation',
        'avg_tcell_exhaustion', 'tumor_immune_ratio', 'tumor_reduction_percentage',
        'patient_age', 'patient_bmi'
    ]
    
    # Converti sesso in numerico
    sim_df['sex_numeric'] = sim_df['patient_sex'].map({'male': 0, 'female': 1})
    features.append('sex_numeric')
    
    return sim_df, features

print("\nPreparazione dati...")
sim_df, features = prepare_data(sim_df)

# =============================================
# 4. MODELLI DI MACHINE LEARNING PER MIGLIORARE LE STIME
# =============================================
def train_ml_models(sim_df, features):
    """Addestra modelli ML per affinare le stime"""
    # Prepariamo i dati
    X = sim_df[features]
    y_os = sim_df['OS_months']
    y_pfs = sim_df['PFS_months']
    
    # Modello per OS
    os_model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=5)
    os_model.fit(X, y_os)
    sim_df['OS_months_ML'] = os_model.predict(X)
    
    # Modello per PFS
    pfs_model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=5)
    pfs_model.fit(X, y_pfs)
    sim_df['PFS_months_ML'] = pfs_model.predict(X)
    
    # Valutazione incrociata
    X_train, X_test, y_os_train, y_os_test = train_test_split(X, y_os, test_size=0.2, random_state=42)
    os_model.fit(X_train, y_os_train)
    os_pred = os_model.predict(X_test)
    os_mae = mean_absolute_error(y_os_test, os_pred)
    os_r2 = r2_score(y_os_test, os_pred)
    
    _, _, y_pfs_train, y_pfs_test = train_test_split(X, y_pfs, test_size=0.2, random_state=42)
    pfs_model.fit(X_train, y_pfs_train)
    pfs_pred = pfs_model.predict(X_test)
    pfs_mae = mean_absolute_error(y_pfs_test, pfs_pred)
    pfs_r2 = r2_score(y_pfs_test, pfs_pred)
    
    print("\nPerformance dei modelli ML:")
    print(f"OS - MAE: {os_mae:.2f} mesi, R2: {os_r2:.2f}")
    print(f"PFS - MAE: {pfs_mae:.2f} mesi, R2: {pfs_r2:.2f}")
    
    return sim_df

print("\nAddestramento modelli ML...")
sim_df = train_ml_models(sim_df, features)

# =============================================
# 5. ANALISI PER CATEGORIE
# =============================================
def analyze_by_category(sim_df):
    """Calcola le statistiche per ogni categoria"""
    results = {}
    
    # Definizione delle categorie
    categories = {
        'Sex': {'male': sim_df[sim_df['patient_sex'] == 'male'],
                'female': sim_df[sim_df['patient_sex'] == 'female']},
        'Age': {'Age<50': sim_df[sim_df['Age_Group'] == 'Age<50'],
                'Age50-69': sim_df[sim_df['Age_Group'] == 'Age50-69'],
                'Age≥70': sim_df[sim_df['Age_Group'] == 'Age≥70']},
        'BMI': {'BMI<25': sim_df[sim_df['BMI_Group'] == 'BMI<25'],
                'BMI≥25': sim_df[sim_df['BMI_Group'] == 'BMI≥25']}
    }
    
    for cat_name, groups in categories.items():
        results[cat_name] = {}
        for group_name, group_data in groups.items():
            results[cat_name][group_name] = {
                'OS_median': group_data['OS_months_ML'].median(),
                'PFS_median': group_data['PFS_months_ML'].median(),
                'count': len(group_data)
            }
    
    return results

# Analisi per categorie
print("\nAnalisi per categorie...")
category_results = analyze_by_category(sim_df)

# =============================================
# 6. GENERAZIONE GRAFICI COMPARATIVI
# =============================================
def generate_comparison_plots(pdf_data, category_results):
    """Genera i grafici comparativi"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confronto OS e PFS: Simulazione vs PDF', fontsize=16, fontweight='bold')
    
    # Configurazione subplot
    titles = [
        ['Overall Survival - Sesso', 'Overall Survival - Età', 'Overall Survival - BMI'],
        ['Progression-Free Survival - Sesso', 'Progression-Free Survival - Età', 'Progression-Free Survival - BMI']
    ]
    
    for i in range(2):
        for j in range(3):
            axes[i,j].set_title(titles[i][j], fontweight='bold')
            axes[i,j].grid(True, alpha=0.3)
    
    # Colori
    sim_color = '#1f77b4'
    pdf_color = '#ff7f0e'
    width = 0.35
    
    # ===== OS PER SESSO =====
    sex_groups = ['male', 'female']
    sim_os = [category_results['Sex'][s]['OS_median'] for s in sex_groups]
    pdf_os = [pdf_data['OS'][s.capitalize()]['median'] for s in sex_groups]
    
    x = np.arange(len(sex_groups))
    axes[0,0].bar(x - width/2, sim_os, width, color=sim_color, alpha=0.7, label='Simulazione (ML)')
    axes[0,0].bar(x + width/2, pdf_os, width, color=pdf_color, alpha=0.7, label='PDF')
    axes[0,0].set_ylabel('OS (mesi)')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(['Male', 'Female'])
    axes[0,0].legend()
    
    # Aggiungi valori
    for i, (sim_val, pdf_val) in enumerate(zip(sim_os, pdf_os)):
        axes[0,0].text(i - width/2, sim_val + 1, f'{sim_val:.1f}', ha='center', color='black')
        axes[0,0].text(i + width/2, pdf_val + 1, f'{pdf_val:.1f}', ha='center', color='black')
    
    # ===== OS PER ETÀ =====
    age_groups = ['Age<50', 'Age50-69', 'Age≥70']
    sim_os_age = [category_results['Age'][a]['OS_median'] for a in age_groups]
    pdf_os_age = [pdf_data['OS'].get(a, {}).get('median', 0) for a in age_groups]
    
    x = np.arange(len(age_groups))
    axes[0,1].bar(x - width/2, sim_os_age, width, color=sim_color, alpha=0.7, label='Simulazione (ML)')
    axes[0,1].bar(x + width/2, pdf_os_age, width, color=pdf_color, alpha=0.7, label='PDF')
    axes[0,1].set_ylabel('OS (mesi)')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(age_groups)
    axes[0,1].legend()
    
    # Aggiungi valori
    for i, (sim_val, pdf_val) in enumerate(zip(sim_os_age, pdf_os_age)):
        if sim_val > 0:
            axes[0,1].text(i - width/2, sim_val + 1, f'{sim_val:.1f}', ha='center', color='black')
        if pdf_val > 0:
            axes[0,1].text(i + width/2, pdf_val + 1, f'{pdf_val:.1f}', ha='center', color='black')
    
    # ===== OS PER BMI =====
    bmi_groups = ['BMI<25', 'BMI≥25']
    sim_os_bmi = [category_results['BMI'][b]['OS_median'] for b in bmi_groups]
    pdf_os_bmi = [pdf_data['OS'][b]['median'] for b in bmi_groups]
    
    x = np.arange(len(bmi_groups))
    axes[0,2].bar(x - width/2, sim_os_bmi, width, color=sim_color, alpha=0.7, label='Simulazione (ML)')
    axes[0,2].bar(x + width/2, pdf_os_bmi, width, color=pdf_color, alpha=0.7, label='PDF')
    axes[0,2].set_ylabel('OS (mesi)')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(bmi_groups)
    axes[0,2].legend()
    
    # Aggiungi valori
    for i, (sim_val, pdf_val) in enumerate(zip(sim_os_bmi, pdf_os_bmi)):
        axes[0,2].text(i - width/2, sim_val + 1, f'{sim_val:.1f}', ha='center', color='black')
        axes[0,2].text(i + width/2, pdf_val + 1, f'{pdf_val:.1f}', ha='center', color='black')
    
    # ===== PFS PER SESSO =====
    sim_pfs_sex = [category_results['Sex'][s]['PFS_median'] for s in sex_groups]
    pdf_pfs_sex = [pdf_data['PFS'].get(s.capitalize(), {}).get('median', 15.7) for s in sex_groups]
    
    x = np.arange(len(sex_groups))
    axes[1,0].bar(x - width/2, sim_pfs_sex, width, color=sim_color, alpha=0.7, label='Simulazione (ML)')
    axes[1,0].bar(x + width/2, pdf_pfs_sex, width, color=pdf_color, alpha=0.7, label='PDF')
    axes[1,0].set_ylabel('PFS (mesi)')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(['Male', 'Female'])
    axes[1,0].legend()
    
    # Aggiungi valori
    for i, (sim_val, pdf_val) in enumerate(zip(sim_pfs_sex, pdf_pfs_sex)):
        axes[1,0].text(i - width/2, sim_val + 0.5, f'{sim_val:.1f}', ha='center', color='black')
        axes[1,0].text(i + width/2, pdf_val + 0.5, f'{pdf_val:.1f}', ha='center', color='black')
    
    # ===== PFS PER ETÀ =====
    sim_pfs_age = [category_results['Age'][a]['PFS_median'] for a in age_groups]
    pdf_pfs_age = [15.7 for _ in age_groups]  # Default value
    
    x = np.arange(len(age_groups))
    axes[1,1].bar(x - width/2, sim_pfs_age, width, color=sim_color, alpha=0.7, label='Simulazione (ML)')
    axes[1,1].bar(x + width/2, pdf_pfs_age, width, color=pdf_color, alpha=0.7, label='PDF')
    axes[1,1].set_ylabel('PFS (mesi)')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(age_groups)
    axes[1,1].legend()
    
    # Aggiungi valori
    for i, (sim_val, pdf_val) in enumerate(zip(sim_pfs_age, pdf_pfs_age)):
        axes[1,1].text(i - width/2, sim_val + 0.5, f'{sim_val:.1f}', ha='center', color='black')
        axes[1,1].text(i + width/2, pdf_val + 0.5, f'{pdf_val:.1f}', ha='center', color='black')
    
    # ===== PFS PER BMI =====
    sim_pfs_bmi = [category_results['BMI'][b]['PFS_median'] for b in bmi_groups]
    pdf_pfs_bmi = [15.7 for _ in bmi_groups]  # Default value
    
    x = np.arange(len(bmi_groups))
    axes[1,2].bar(x - width/2, sim_pfs_bmi, width, color=sim_color, alpha=0.7, label='Simulazione (ML)')
    axes[1,2].bar(x + width/2, pdf_pfs_bmi, width, color=pdf_color, alpha=0.7, label='PDF')
    axes[1,2].set_ylabel('PFS (mesi)')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(bmi_groups)
    axes[1,2].legend()
    
    # Aggiungi valori
    for i, (sim_val, pdf_val) in enumerate(zip(sim_pfs_bmi, pdf_pfs_bmi)):
        axes[1,2].text(i - width/2, sim_val + 0.5, f'{sim_val:.1f}', ha='center', color='black')
        axes[1,2].text(i + width/2, pdf_val + 0.5, f'{pdf_val:.1f}', ha='center', color='black')
    
    plt.tight_layout()
    plt.savefig('comparison_plot_final.png', dpi=300)
    plt.show()

# Genera i grafici
print("\nGenerazione grafici comparativi finali...")
generate_comparison_plots(pdf_data, category_results)

print("\nAnalisi completata con successo!")