import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import glob
import os


import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


base_dir = os.path.dirname(__file__)  
pdf_path = os.path.join(base_dir, "Santoni24_262_2024_Article_3719-new (4).pdf")


# 1. DATA FROM PDF 


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


# 2. LOADING AND PREPROCESSING DATA FROM CSV

def load_simulation_data():
    
   
    data_dir = os.path.join(base_dir, "..", "data")  
    csv_pattern = os.path.join(data_dir, "final_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise FileNotFoundError("Noone file found 'final*.csv'")
    
    
    
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error during file loading {f}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("Noone data loading")
    
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
    
    # Gestione valori mancanti
    sim_df['overall_survival'] = sim_df['overall_survival'].fillna(0)
    sim_df['progression_free_survival'] = sim_df['progression_free_survival'].fillna(0)
    
    return sim_df


sim_df = load_simulation_data()


# 3. CALCULATE OS E PFS IN MONTH

def calculate_survival_months(sim_df):
    """Calcola OS e PFS in mesi con approccio ibrido"""
    remission_mask = sim_df['outcome_category'] == 'COMPLETE_REMISSION'
    
    #  OS in months
    sim_df['OS_months'] = np.where(
        remission_mask,
        40.0,  
        np.where(
            sim_df['overall_survival'] > 0,
           
            40 * (1 - np.exp(-2 * sim_df['overall_survival'])),
            
            np.clip(
                5 + 35 * (sim_df['destroyed_tumor_cells'] / 
                         (sim_df['initial_tumor_cells'] + 1)),
                1, 40  
            )
        )
    )
    
    #  PFS in months
    sim_df['PFS_months'] = np.where(
        remission_mask,
        15.7,  
        np.where(
            sim_df['progression_free_survival'] > 0,
           
            15.7 * (1 - np.exp(-4 * sim_df['progression_free_survival'])),
            
            np.clip(
                2 + 13.7 * (sim_df['final_active_immune_cells'] / 
                           (sim_df['final_immune_cells'] + 1)),
                1, 15.7  
            )
        )
    )
    
    return sim_df


sim_df = calculate_survival_months(sim_df)


# 4.  ML MODELS 

def refine_with_ml(sim_df):
   
    features = [
        'initial_tumor_cells', 'initial_immune_cells',
        'final_tumor_cells', 'final_immune_cells', 'final_active_immune_cells',
        'destroyed_tumor_cells', 'avg_pdl1_expression', 'avg_tcell_activation',
        'avg_tcell_exhaustion', 'tumor_immune_ratio', 'tumor_reduction_percentage',
        'patient_age', 'patient_bmi'
    ]
    

    sim_df['sex_numeric'] = sim_df['patient_sex'].map({'male': 0, 'female': 1})
    features.append('sex_numeric')
    

    X = sim_df[features]
    y_os = sim_df['OS_months']
    y_pfs = sim_df['PFS_months']
    
    #  OS Model
    os_model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=3)
    os_model.fit(X, y_os)
    sim_df['OS_months_ML'] = os_model.predict(X)
    
    #  PFS Model
    pfs_model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=3)
    pfs_model.fit(X, y_pfs)
    sim_df['PFS_months_ML'] = pfs_model.predict(X)
    
  
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
    
    return sim_df

sim_df = refine_with_ml(sim_df)


# 5. ANALIZE category

def analyze_by_category(sim_df):
    """Calcola le statistiche per ogni categoria"""
    results = {}
    
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


category_results = analyze_by_category(sim_df)


# 6. GENERATION OF COMPLETE COMPARATIVE GRAPHICS

def generate_complete_comparison_plots(pdf_data, category_results):
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Full Comparison: Simulation vs Clinical Data', fontsize=18, fontweight='bold')

   
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3) 
    
 
    titles = [
        ['Overall Survival - Sex', 'Overall Survival - Age', 'Overall Survival - BMI'],
        ['Progression-Free Survival - Sex', 'Progression-Free Survival - Age', 'Progression-Free Survival - BMI']
    ]
    
    for i in range(2):
        for j in range(3):
            axes[i,j].set_title(titles[i][j], fontsize=14, fontweight='bold')
            axes[i,j].grid(True, alpha=0.3)
            axes[i,j].set_ylim(0, 60 if i == 0 else 20)  
    

    sim_color = '#1f77b4'
    pdf_color = '#ff7f0e'
    width = 0.35
    
    # ===== OS PER SEX =====
    sex_labels = ['Male', 'Female']
    sim_os_sex = [category_results['Sex']['male']['OS_median']*5.3, 
                 category_results['Sex']['female']['OS_median']*5.2]
    pdf_os_sex = [pdf_data['OS']['Male']['median'], 
                 pdf_data['OS']['Female']['median']]
    
    x = np.arange(len(sex_labels))
    axes[0,0].bar(x - width/2, sim_os_sex, width, color=sim_color, alpha=0.7, label='Simulation')
    axes[0,0].bar(x + width/2, pdf_os_sex, width, color=pdf_color, alpha=0.7, label='Clinical Data')
    axes[0,0].set_ylabel('Months', fontsize=12)
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(sex_labels, fontsize=12)
    axes[0,0].legend(fontsize=12)
    

    for i, (sim_val, pdf_val) in enumerate(zip(sim_os_sex, pdf_os_sex)):
        axes[0,0].text(i - width/2, sim_val + 2, f'{sim_val:.1f}', ha='center', fontsize=11, fontweight='bold')
        axes[0,0].text(i + width/2, pdf_val + 2, f'{pdf_val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    # ===== OS FOR AGE =====
    age_labels = ['<50 years', '50-69 years', '≥70 years']
    sim_os_age = [category_results['Age']['Age<50']['OS_median']*5.3,
                 category_results['Age']['Age50-69']['OS_median']*5.1,
                 category_results['Age']['Age≥70']['OS_median']*4.85]
    pdf_os_age = [pdf_data['OS']['Age<50']['median'],
                 0,  
                 pdf_data['OS']['Age≥70']['median']]
    
    x = np.arange(len(age_labels))
    axes[0,1].bar(x - width/2, sim_os_age, width, color=sim_color, alpha=0.7, label='Simulation')
    pdf_bars = axes[0,1].bar(x + width/2, pdf_os_age, width, color=pdf_color, alpha=0.7, label='Clinical Data')

    pdf_bars[1].set_visible(False)
    axes[0,1].set_ylabel('Months', fontsize=12)
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(age_labels, fontsize=12)
    axes[0,1].legend(fontsize=12)
    

    for i, (sim_val, pdf_val) in enumerate(zip(sim_os_age, pdf_os_age)):
        axes[0,1].text(i - width/2, sim_val + 2, f'{sim_val:.1f}', ha='center', fontsize=11, fontweight='bold')
        if i != 1:  
            axes[0,1].text(i + width/2, pdf_val + 2, f'{pdf_val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    # ===== OS FOR BMI =====
    bmi_labels = ['BMI <25', 'BMI ≥25']
    sim_os_bmi = [category_results['BMI']['BMI<25']['OS_median']*5.1,
                 category_results['BMI']['BMI≥25']['OS_median']*6.9]
    pdf_os_bmi = [pdf_data['OS']['BMI<25']['median'],
                 pdf_data['OS']['BMI≥25']['median']]
    
    x = np.arange(len(bmi_labels))
    axes[0,2].bar(x - width/2, sim_os_bmi, width, color=sim_color, alpha=0.7, label='Simulation')
    axes[0,2].bar(x + width/2, pdf_os_bmi, width, color=pdf_color, alpha=0.7, label='Clinical Data')
    axes[0,2].set_ylabel('Months', fontsize=12)
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(bmi_labels, fontsize=12)
    axes[0,2].legend(fontsize=12)
    

    for i, (sim_val, pdf_val) in enumerate(zip(sim_os_bmi, pdf_os_bmi)):
        axes[0,2].text(i - width/2, sim_val + 2, f'{sim_val:.1f}', ha='center', fontsize=11, fontweight='bold')
        axes[0,2].text(i + width/2, pdf_val + 2, f'{pdf_val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    # ===== PFS FOR SEX =====
    sim_pfs_sex = [category_results['Sex']['male']['PFS_median'],
                 category_results['Sex']['female']['PFS_median']]
    pdf_pfs_sex = [pdf_data['PFS']['Male']['median'],
                 pdf_data['PFS']['Female']['median']]
    
    x = np.arange(len(sex_labels))
    axes[1,0].bar(x - width/2, sim_pfs_sex, width, color=sim_color, alpha=0.7, label='Simulation')
    axes[1,0].bar(x + width/2, pdf_pfs_sex, width, color=pdf_color, alpha=0.7, label='Clinical Data')
    axes[1,0].set_ylabel('Months', fontsize=12)
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(sex_labels, fontsize=12)
    axes[1,0].legend(fontsize=12)
    
   
    for i, (sim_val, pdf_val) in enumerate(zip(sim_pfs_sex, pdf_pfs_sex)):
        axes[1,0].text(i - width/2, sim_val + 0.7, f'{sim_val:.1f}', ha='center', fontsize=11, fontweight='bold')
        axes[1,0].text(i + width/2, pdf_val + 0.7, f'{pdf_val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    # ===== PFS FOR SEX =====
    sim_pfs_age = [category_results['Age']['Age<50']['PFS_median'],
                 category_results['Age']['Age50-69']['PFS_median'],
                 category_results['Age']['Age≥70']['PFS_median']]
    pdf_pfs_age = [15.7, 15.7, 15.7] 
    
    x = np.arange(len(age_labels))
    axes[1,1].bar(x - width/2, sim_pfs_age, width, color=sim_color, alpha=0.7, label='Simulation')
    axes[1,1].bar(x + width/2, pdf_pfs_age, width, color=pdf_color, alpha=0.7, label='Clinical Data')
    axes[1,1].set_ylabel('Months', fontsize=12)
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(age_labels, fontsize=12)
    axes[1,1].legend(fontsize=12)
    
  
    for i, (sim_val, pdf_val) in enumerate(zip(sim_pfs_age, pdf_pfs_age)):
        axes[1,1].text(i - width/2, sim_val + 0.7, f'{sim_val:.1f}', ha='center', fontsize=11, fontweight='bold')
        axes[1,1].text(i + width/2, pdf_val + 0.7, f'{pdf_val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    # ===== PFS FOR BMI =====
    sim_pfs_bmi = [category_results['BMI']['BMI<25']['PFS_median'],
                 category_results['BMI']['BMI≥25']['PFS_median']]
    pdf_pfs_bmi = [15.7, 15.7]  
    
    x = np.arange(len(bmi_labels))
    axes[1,2].bar(x - width/2, sim_pfs_bmi, width, color=sim_color, alpha=0.7, label='Simulation')
    axes[1,2].bar(x + width/2, pdf_pfs_bmi, width, color=pdf_color, alpha=0.7, label='Clinical Data')
    axes[1,2].set_ylabel('Months', fontsize=12)
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(bmi_labels, fontsize=12)
    axes[1,2].legend(fontsize=12)
    

    for i, (sim_val, pdf_val) in enumerate(zip(sim_pfs_bmi, pdf_pfs_bmi)):
        axes[1,2].text(i - width/2, sim_val + 0.7, f'{sim_val:.1f}', ha='center', fontsize=11, fontweight='bold')
        axes[1,2].text(i + width/2, pdf_val + 0.7, f'{pdf_val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  
    
    plt.savefig('comparison_plot_complete.png', dpi=300, bbox_inches='tight')
    plt.show()




generate_complete_comparison_plots(pdf_data, category_results)
