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

# =============================================
# 1. DATI DI RIFERIMENTO DAL PDF (hardcoded)
# =============================================
pdf_data = {
    "Group": ["Overall", "Male", "Female", "Age<50", "Age≥70", "BMI<25", "BMI≥25"],
    "Median_OS_months": [38.7, 40.0, 38.7, 36.9, 31.4, 31.6, 44.2],
    "OS_CI_lower": [32.7, 32.7, 26.4, 29.0, 26.4, 25.9, 35.8],
    "OS_CI_upper": [44.2, 51.6, 41.0, 51.6, 49.2, 40.2, 55.7],
    "Median_PFS_months": [15.7, 15.7, 15.7, None, None, None, None]
}

pdf_df = pd.DataFrame(pdf_data)
print("Dati di riferimento dal PDF:")
print(pdf_df)

# =============================================
# 2. CARICAMENTO E PREPROCESSING DEI CSV
# =============================================
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    # Converti OS e PFS in mesi
    df["OS_months"] = df["Overall Survival"] * 0.4  # 100% = 40 mesi
    df["PFS_months"] = df["Progression-Free Survival"] * 0.157  # 100% = 15.7 mesi
    
    # Aggiungi colonne per i gruppi (come nel PDF)
    df["Age_Group"] = pd.cut(
        df["Patient Age"],
        bins=[0, 50, 70, 100],
        labels=["Age<50", "Age50-69", "Age≥70"]
    )
    df["BMI_Group"] = pd.cut(
        df["Patient BMI"],
        bins=[0, 25, 100],
        labels=["BMI<25", "BMI≥25"]
    )
    return df

# Carica tutti i CSV - con controllo di esistenza
data_dir = os.path.join(base_dir, "..", "data")  # vai su di una cartella poi nella cartella data
csv_pattern = os.path.join(data_dir, "simulation_*.csv")
csv_files = glob.glob(csv_pattern)

print(f"\nCercando CSV in: {data_dir}")
print(f"Pattern di ricerca: {csv_pattern}")
print(f"File CSV trovati: {csv_files}")

if not csv_files:
    print("ERRORE: Nessun file CSV trovato. Creando dati di esempio...")
    np.random.seed(42)
    n_samples = 1000
    
    sim_df = pd.DataFrame({
        'Patient Sex': np.random.choice(['Male', 'Female'], n_samples),
        'Patient Age': np.random.normal(60, 15, n_samples),
        'Patient BMI': np.random.normal(25, 5, n_samples),
        'Overall Survival': np.random.normal(97, 10, n_samples),  # ~38.7 mesi
        'Progression-Free Survival': np.random.normal(100, 15, n_samples),  # ~15.7 mesi
        'Tumor Cells (current)': np.random.normal(50, 20, n_samples),
        'Immune Cells (total)': np.random.normal(30, 10, n_samples),
        'Active Immune Cells': np.random.normal(20, 8, n_samples)
    })
    
    # Applica preprocessing
    sim_df = load_and_preprocess_example(sim_df)
else:
    # Carica i CSV esistenti
    sim_df = pd.concat([load_and_preprocess(f) for f in csv_files], ignore_index=True)

def load_and_preprocess_example(df):
    # Converti OS e PFS in mesi
    df["OS_months"] = df["Overall Survival"] * 0.4  # 100% = 40 mesi
    df["PFS_months"] = df["Progression-Free Survival"] * 0.157  # 100% = 15.7 mesi
    
    # Aggiungi colonne per i gruppi (come nel PDF)
    df["Age_Group"] = pd.cut(
        df["Patient Age"],
        bins=[0, 50, 70, 100],
        labels=["Age<50", "Age50-69", "Age≥70"]
    )
    df["BMI_Group"] = pd.cut(
        df["Patient BMI"],
        bins=[0, 25, 100],
        labels=["BMI<25", "BMI≥25"]
    )
    return df

print(f"\nDataset caricato: {len(sim_df)} righe")
print(f"Colonne: {list(sim_df.columns)}")

# =============================================
# 3. FUNZIONE PER CREARE GRAFICI A BARRE COMPARATIVI
# =============================================
def create_comparison_charts():
    # Crea una figura con 2 sezioni principali: OS e PFS
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confronto OS e PFS: Simulazione vs PDF', fontsize=16, fontweight='bold')
    
    # ===== SEZIONE OS (riga superiore) =====
    axes[0, 0].set_title('Overall Survival - Sesso', fontweight='bold')
    axes[0, 1].set_title('Overall Survival - Età', fontweight='bold')
    axes[0, 2].set_title('Overall Survival - BMI', fontweight='bold')
    
    # ===== SEZIONE PFS (riga inferiore) =====
    axes[1, 0].set_title('Progression-Free Survival - Sesso', fontweight='bold')
    axes[1, 1].set_title('Progression-Free Survival - Età', fontweight='bold')
    axes[1, 2].set_title('Progression-Free Survival - BMI', fontweight='bold')
    
    # ===== OS PER SESSO =====
    sex_groups = ['Male', 'Female']
    sim_os_sex = []
    pdf_os_sex = []
    
    for sex in sex_groups:
        # Simulazione
        sim_subset = sim_df[sim_df['Patient Sex'] == sex]
        sim_os_sex.append(sim_subset['OS_months'].median())
        
        # PDF
        pdf_val = pdf_df[pdf_df['Group'] == sex]['Median_OS_months'].values
        pdf_os_sex.append(pdf_val[0] if len(pdf_val) > 0 else 0)
    
    x = np.arange(len(sex_groups))
    width = 0.35
    axes[0, 0].bar(x - width/2, sim_os_sex, width, label='Simulazione', color='skyblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, pdf_os_sex, width, label='PDF', color='lightcoral', alpha=0.8)
    axes[0, 0].set_ylabel('OS (mesi)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(sex_groups)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Aggiungi valori esatti sopra le barre
    for i, (sim_val, pdf_val) in enumerate(zip(sim_os_sex, pdf_os_sex)):
        axes[0, 0].text(i - width/2, sim_val + 1, f'{sim_val:.1f}', ha='center', va='bottom')
        axes[0, 0].text(i + width/2, pdf_val + 1, f'{pdf_val:.1f}', ha='center', va='bottom')
    
    # ===== OS PER ETÀ =====
    age_groups = ['Age<50', 'Age50-69', 'Age≥70']  # Ora includiamo anche Age50-69
    sim_os_age = []
    pdf_os_age = []
    
    for age in age_groups:
        # Simulazione
        sim_subset = sim_df[sim_df['Age_Group'] == age]
        sim_os_age.append(sim_subset['OS_months'].median())
        
        # PDF (solo per Age<50 e Age≥70)
        if age == 'Age<50':
            pdf_val = pdf_df[pdf_df['Group'] == 'Age<50']['Median_OS_months'].values[0]
        elif age == 'Age≥70':
            pdf_val = pdf_df[pdf_df['Group'] == 'Age≥70']['Median_OS_months'].values[0]
        else:
            pdf_val = np.nan  # Non presente nel PDF
        pdf_os_age.append(pdf_val)
    
    x = np.arange(len(age_groups))
    width = 0.35
    axes[0, 1].bar(x - width/2, sim_os_age, width, label='Simulazione', color='skyblue', alpha=0.8)
    # Mostra solo le barre PDF per i gruppi presenti nel PDF
    pdf_mask = ~np.isnan(pdf_os_age)
    axes[0, 1].bar(x[pdf_mask] + width/2, np.array(pdf_os_age)[pdf_mask], width, label='PDF', color='lightcoral', alpha=0.8)
    axes[0, 1].set_ylabel('OS (mesi)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(age_groups)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Aggiungi valori esatti sopra le barre
    for i, (sim_val, pdf_val) in enumerate(zip(sim_os_age, pdf_os_age)):
        axes[0, 1].text(i - width/2, sim_val + 1, f'{sim_val:.1f}', ha='center', va='bottom')
        if not np.isnan(pdf_val):
            axes[0, 1].text(i + width/2, pdf_val + 1, f'{pdf_val:.1f}', ha='center', va='bottom')
    
    # ===== OS PER BMI =====
    bmi_groups = ['BMI<25', 'BMI≥25']
    sim_os_bmi = []
    pdf_os_bmi = []
    
    for bmi in bmi_groups:
        # Simulazione
        sim_subset = sim_df[sim_df['BMI_Group'] == bmi]
        sim_os_bmi.append(sim_subset['OS_months'].median())
        
        # PDF
        pdf_val = pdf_df[pdf_df['Group'] == bmi]['Median_OS_months'].values
        pdf_os_bmi.append(pdf_val[0] if len(pdf_val) > 0 else 0)
    
    x = np.arange(len(bmi_groups))
    axes[0, 2].bar(x - width/2, sim_os_bmi, width, label='Simulazione', color='skyblue', alpha=0.8)
    axes[0, 2].bar(x + width/2, pdf_os_bmi, width, label='PDF', color='lightcoral', alpha=0.8)
    axes[0, 2].set_ylabel('OS (mesi)')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(bmi_groups)
    axes[0, 2].legend()
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Aggiungi valori esatti sopra le barre
    for i, (sim_val, pdf_val) in enumerate(zip(sim_os_bmi, pdf_os_bmi)):
        axes[0, 2].text(i - width/2, sim_val + 1, f'{sim_val:.1f}', ha='center', va='bottom')
        axes[0, 2].text(i + width/2, pdf_val + 1, f'{pdf_val:.1f}', ha='center', va='bottom')
    
    # ===== PFS PER SESSO =====
    sim_pfs_sex = []
    pdf_pfs_sex = []
    
    for sex in sex_groups:
        # Simulazione
        sim_subset = sim_df[sim_df['Patient Sex'] == sex]
        sim_pfs_sex.append(sim_subset['PFS_months'].median())
        
        # PDF (solo Overall ha PFS nel PDF)
        pdf_val = pdf_df[pdf_df['Group'] == 'Overall']['Median_PFS_months'].values
        pdf_pfs_sex.append(pdf_val[0] if len(pdf_val) > 0 and pd.notna(pdf_val[0]) else 15.7)
    
    x = np.arange(len(sex_groups))
    axes[1, 0].bar(x - width/2, sim_pfs_sex, width, label='Simulazione', color='lightgreen', alpha=0.8)
    axes[1, 0].bar(x + width/2, pdf_pfs_sex, width, label='PDF', color='orange', alpha=0.8)
    axes[1, 0].set_ylabel('PFS (mesi)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(sex_groups)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Aggiungi valori esatti sopra le barre
    for i, (sim_val, pdf_val) in enumerate(zip(sim_pfs_sex, pdf_pfs_sex)):
        axes[1, 0].text(i - width/2, sim_val + 0.5, f'{sim_val:.1f}', ha='center', va='bottom')
        axes[1, 0].text(i + width/2, pdf_val + 0.5, f'{pdf_val:.1f}', ha='center', va='bottom')
    
    # ===== PFS PER ETÀ =====
    age_groups = ['Age<50', 'Age50-69', 'Age≥70']  # Ora includiamo anche Age50-69
    sim_pfs_age = []
    pdf_pfs_age = []
    
    for age in age_groups:
        # Simulazione
        sim_subset = sim_df[sim_df['Age_Group'] == age]
        sim_pfs_age.append(sim_subset['PFS_months'].median())
        
        # PDF (usa Overall come riferimento)
        pdf_pfs_age.append(15.7)
    
    x = np.arange(len(age_groups))
    axes[1, 1].bar(x - width/2, sim_pfs_age, width, label='Simulazione', color='lightgreen', alpha=0.8)
    axes[1, 1].bar(x + width/2, pdf_pfs_age, width, label='PDF', color='orange', alpha=0.8)
    axes[1, 1].set_ylabel('PFS (mesi)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(age_groups)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Aggiungi valori esatti sopra le barre
    for i, (sim_val, pdf_val) in enumerate(zip(sim_pfs_age, pdf_pfs_age)):
        axes[1, 1].text(i - width/2, sim_val + 0.5, f'{sim_val:.1f}', ha='center', va='bottom')
        axes[1, 1].text(i + width/2, pdf_val + 0.5, f'{pdf_val:.1f}', ha='center', va='bottom')
    
    # ===== PFS PER BMI =====
    sim_pfs_bmi = []
    pdf_pfs_bmi = []
    
    for bmi in bmi_groups:
        # Simulazione
        sim_subset = sim_df[sim_df['BMI_Group'] == bmi]
        sim_pfs_bmi.append(sim_subset['PFS_months'].median())
        
        # PDF (usa Overall come riferimento)
        pdf_pfs_bmi.append(15.7)
    
    x = np.arange(len(bmi_groups))
    axes[1, 2].bar(x - width/2, sim_pfs_bmi, width, label='Simulazione', color='lightgreen', alpha=0.8)
    axes[1, 2].bar(x + width/2, pdf_pfs_bmi, width, label='PDF', color='orange', alpha=0.8)
    axes[1, 2].set_ylabel('PFS (mesi)')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(bmi_groups)
    axes[1, 2].legend()
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    # Aggiungi valori esatti sopra le barre
    for i, (sim_val, pdf_val) in enumerate(zip(sim_pfs_bmi, pdf_pfs_bmi)):
        axes[1, 2].text(i - width/2, sim_val + 0.5, f'{sim_val:.1f}', ha='center', va='bottom')
        axes[1, 2].text(i + width/2, pdf_val + 0.5, f'{pdf_val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Mostra i grafici
create_comparison_charts()

# =============================================
# 4. CALCOLA LE DIFFERENZE TRA SIMULAZIONE E PDF
# =============================================
def calculate_differences():
    print("\n" + "="*60)
    print("ANALISI DELLE DIFFERENZE TRA SIMULAZIONE E PDF")
    print("="*60)
    
    differences = []
    
    # Per sesso
    for sex in ['Male', 'Female']:
        sim_subset = sim_df[sim_df['Patient Sex'] == sex]
        sim_os = sim_subset['OS_months'].median()
        sim_pfs = sim_subset['PFS_months'].median()
        
        pdf_os = pdf_df[pdf_df['Group'] == sex]['Median_OS_months'].values[0]
        pdf_pfs = 15.7  # Overall PFS dal PDF
        
        differences.append({
            'Category': 'Sesso',
            'Group': sex,
            'OS_Sim': sim_os,
            'OS_PDF': pdf_os,
            'OS_Diff': sim_os - pdf_os,
            'PFS_Sim': sim_pfs,
            'PFS_PDF': pdf_pfs,
            'PFS_Diff': sim_pfs - pdf_pfs
        })
    
    # Per età
    for age_group in ['Age<50', 'Age50-69', 'Age≥70']:  # Ora includiamo anche Age50-69
        sim_subset = sim_df[sim_df['Age_Group'] == age_group]
        sim_os = sim_subset['OS_months'].median()
        sim_pfs = sim_subset['PFS_months'].median()
        
        # Solo per Age<50 e Age≥70 abbiamo dati PDF
        if age_group == 'Age<50':
            pdf_os = pdf_df[pdf_df['Group'] == 'Age<50']['Median_OS_months'].values[0]
        elif age_group == 'Age≥70':
            pdf_os = pdf_df[pdf_df['Group'] == 'Age≥70']['Median_OS_months'].values[0]
        else:
            pdf_os = np.nan  # Non presente nel PDF
        
        pdf_pfs = 15.7
        
        differences.append({
            'Category': 'Età',
            'Group': age_group,
            'OS_Sim': sim_os,
            'OS_PDF': pdf_os,
            'OS_Diff': sim_os - pdf_os if not np.isnan(pdf_os) else np.nan,
            'PFS_Sim': sim_pfs,
            'PFS_PDF': pdf_pfs,
            'PFS_Diff': sim_pfs - pdf_pfs
        })
    
    # Per BMI
    for bmi in ['BMI<25', 'BMI≥25']:
        sim_subset = sim_df[sim_df['BMI_Group'] == bmi]
        sim_os = sim_subset['OS_months'].median()
        sim_pfs = sim_subset['PFS_months'].median()
        
        pdf_os = pdf_df[pdf_df['Group'] == bmi]['Median_OS_months'].values[0]
        pdf_pfs = 15.7
        
        differences.append({
            'Category': 'BMI',
            'Group': bmi,
            'OS_Sim': sim_os,
            'OS_PDF': pdf_os,
            'OS_Diff': sim_os - pdf_os,
            'PFS_Sim': sim_pfs,
            'PFS_PDF': pdf_pfs,
            'PFS_Diff': sim_pfs - pdf_pfs
        })
    
    diff_df = pd.DataFrame(differences)
    print(diff_df.to_string(index=False))
    
    return diff_df

# Calcola e mostra le differenze
diff_results = calculate_differences()

# =============================================
# 5. MODELLO DI ML PER OGNI SOTTOGRUPPO
# =============================================
def train_models_by_groups():
    print("\n" + "="*60)
    print("TRAINING MODELLI ML PER SOTTOGRUPPI")
    print("="*60)
    
    features = ["Tumor Cells (current)", "Immune Cells (total)", "Active Immune Cells"]
    results = []
    
    # Per sesso
    for sex in ['Male', 'Female']:
        group_data = sim_df[sim_df['Patient Sex'] == sex]
        if len(group_data) >= 20:
            result = train_model_for_group(group_data, features, f"Sex_{sex}")
            results.append(result)
    
    # Per età
    for age_group in ['Age<50', 'Age50-69', 'Age≥70']:  # Ora includiamo anche Age50-69
        group_data = sim_df[sim_df['Age_Group'] == age_group]
        if len(group_data) >= 20:
            result = train_model_for_group(group_data, features, f"Age_{age_group}")
            results.append(result)
    
    # Per BMI
    for bmi in ['BMI<25', 'BMI≥25']:
        group_data = sim_df[sim_df['BMI_Group'] == bmi]
        if len(group_data) >= 20:
            result = train_model_for_group(group_data, features, f"BMI_{bmi}")
            results.append(result)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df

def train_model_for_group(group_data, features, group_name):
    # Training per OS
    X = group_data[features]
    y_os = group_data["OS_months"]
    y_pfs = group_data["PFS_months"]
    
    X_train, X_test, y_os_train, y_os_test = train_test_split(X, y_os, test_size=0.3, random_state=42)
    _, _, y_pfs_train, y_pfs_test = train_test_split(X, y_pfs, test_size=0.3, random_state=42)
    
    # Modello OS
    model_os = RandomForestRegressor(n_estimators=100, random_state=42)
    model_os.fit(X_train, y_os_train)
    y_os_pred = model_os.predict(X_test)
    os_mae = mean_absolute_error(y_os_test, y_os_pred)
    os_r2 = r2_score(y_os_test, y_os_pred)
    
    # Modello PFS
    model_pfs = RandomForestRegressor(n_estimators=100, random_state=42)
    model_pfs.fit(X_train, y_pfs_train)
    y_pfs_pred = model_pfs.predict(X_test)
    pfs_mae = mean_absolute_error(y_pfs_test, y_pfs_pred)
    pfs_r2 = r2_score(y_pfs_test, y_pfs_pred)
    
    return {
        "Group": group_name,
        "N_samples": len(group_data),
        "OS_MAE": round(os_mae, 2),
        "OS_R2": round(os_r2, 3),
        "PFS_MAE": round(pfs_mae, 2),
        "PFS_R2": round(pfs_r2, 3),
        "OS_median": round(group_data["OS_months"].median(), 1),
        "PFS_median": round(group_data["PFS_months"].median(), 1)
    }

# Allena i modelli
ml_results = train_models_by_groups()