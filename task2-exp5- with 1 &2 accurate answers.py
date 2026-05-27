import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# ==========================================
# --- STEP 1: DIRECTORIES & PATH SETUP ---
# ==========================================
print("[SYSTEM] Setting up file paths and loading all data components...")
base_path_w4 = Path(r"C:\Users\gsudh\Documents\depaul\courses\capstone\week4")
base_path_w5 = Path(r"C:\Users\gsudh\Documents\depaul\courses\capstone\week5")

task1_main_file = base_path_w4 / "task1_gecs_classification_final.csv"
task1_lookup_file = base_path_w4 / "Empty cell info morningstar_gecs_industries.csv"

task2_main_file = base_path_w5 / "task2_subindustry_classification_final.xlsx"
mapping_file = base_path_w5 / "GECS_Activities2026.xlsx"

try:
    # Task 1 Dataset Imports
    df_t1 = pd.read_csv(task1_main_file)
    df_lookup = pd.read_csv(task1_lookup_file)
    
    # Task 2 Dataset Imports
    df_t2_main = pd.read_excel(task2_main_file)
    df_map = pd.read_excel(mapping_file)
    
    # Strip whitespace from headers immediately
    df_t1.columns = df_t1.columns.str.strip()
    df_lookup.columns = df_lookup.columns.str.strip()
    df_t2_main.columns = df_t2_main.columns.str.strip()
    df_map.columns = df_map.columns.str.strip()

except Exception as e:
    print(f"CRITICAL FILE SYSTEM ERROR: Could not load target analytics files.\n{e}")
    sys.exit()

# ==========================================
# --- STEP 2: TASK 1 DICTIONARIES & TRANSFORMS ---
# ==========================================
industry_name_map_t1 = dict(zip(df_lookup['GECS_Code'].astype(str), df_lookup['Industry_Name']))
definition_map_t1 = dict(zip(df_lookup['GECS_Code'].astype(str), df_lookup['Description'].fillna('No official definition.')))

df_t1['MstarGlobal'] = df_t1['MstarGlobal'].astype(str)
df_t1['SegmentDescription'] = df_t1['SegmentDescription'].fillna(df_t1['MstarGlobal'].map(definition_map_t1))

df_t1['master_text'] = (df_t1['SegmentName'].fillna('') + " " + 
                         df_t1['LongProfile'].fillna('') + " " + 
                         df_t1['SegmentDescription'].fillna('')).str.lower()
df_t1['log_rev_transformed'] = np.log1p(df_t1['total_revenue_company_as_of'].clip(lower=0))

le_t1 = LabelEncoder()
y_t1 = le_t1.fit_transform(df_t1['MstarGlobal'])
X_t1 = df_t1[['master_text', 'log_rev_transformed', 'Revenue', 'revenue_share']]

X_train_t1, X_test_t1, y_train_t1, y_test_t1 = train_test_split(
    X_t1, y_t1, test_size=0.2, stratify=y_t1, random_state=42
)

# ==========================================
# --- STEP 3: TASK 2 MERGE & HIERARCHY BUILDING ---
# ==========================================
print("[SYSTEM] Merging relational datasets and generating hierarchical dictionaries...")
if 'CompanyId' in df_t2_main.columns and 'CompanyId' in df_t1.columns:
    df_t2_main['CompanyId'] = df_t2_main['CompanyId'].astype(str).str.strip()
    df_t1['CompanyId'] = df_t1['CompanyId'].astype(str).str.strip()
    df_t2 = pd.merge(df_t2_main, df_t1[['CompanyId', 'Revenue']], on='CompanyId', how='left')
else:
    print("CRITICAL STRUCTURE ERROR: 'CompanyId' identifier field missing from source indices.")
    sys.exit()

def get_map_dict(df, id_idx, name_idx):
    temp = df.iloc[:, [id_idx, name_idx]].dropna().astype(str)
    temp.columns = ['id', 'name']
    return temp.drop_duplicates('id').set_index('id')['name'].to_dict()

sector_map = get_map_dict(df_map, 0, 1)
group_map = get_map_dict(df_map, 2, 3)
industry_map = get_map_dict(df_map, 4, 5)

activity_df = df_map.iloc[:, [6, 7, 8]].dropna().astype(str)
activity_df.columns = ['id', 'name', 'definition']
activity_map = activity_df.drop_duplicates('id').set_index('id').to_dict('index')

# Feature engineering with strict data leakage mitigation 
df_t2['training_text'] = (
    df_t2['SegmentName'].fillna('').astype(str) + " " + 
    df_t2['SegmentDescription'].fillna('').astype(str) + " " + 
    df_t2['Industry'].fillna('').astype(str) + " " +
    df_t2['Revenue'].fillna('0').astype(str)
).str.lower()

df_t2['testing_text'] = (
    df_t2['SegmentName'].fillna('').astype(str) + " " + 
    df_t2['SegmentDescription'].fillna('').astype(str) + " " +
    df_t2['Revenue'].fillna('0').astype(str)
).str.lower()

# Minimum filter strategy to ensure structural stratification safety
counts_t2 = df_t2['SubIndustry'].value_counts()
df_t2_filtered = df_t2[df_t2['SubIndustry'].isin(counts_t2[counts_t2 >= 3].index)].copy()

X_train_t2, X_test_t2, y_train_t2, y_test_t2 = train_test_split(
    df_t2_filtered['training_text'], 
    df_t2_filtered['SubIndustry'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df_t2_filtered['SubIndustry']
)

# ==========================================
# --- STEP 4: TRAINING PIPELINE INFRASTRUCTURE ---
# ==========================================
business_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + \
                      ['company', 'incorporated', 'limited', 'services', 'solutions', 'operations', 'segment']

preprocessor_t1 = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=10000, stop_words=business_stop_words, ngram_range=(1, 3)), 'master_text'),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ]), ['log_rev_transformed', 'Revenue', 'revenue_share'])
    ]
)

pipeline_t1 = Pipeline([
    ('preprocessor', preprocessor_t1),
    ('classifier', LinearSVC(class_weight='balanced', random_state=42, max_iter=3000, C=0.8))
])

pipeline_t2 = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=40000, stop_words='english')),
    ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=7000))
])

# Execute Model Fittings
print("\n[SYSTEM] Training Task 1 Model (Macro GECS Industries)...")
pipeline_t1.fit(X_train_t1, y_train_t1)

print("[SYSTEM] Training Task 2 Model (Hierarchical Sub-Category Deep Dive)...")
pipeline_t2.fit(X_train_t2, y_train_t2)

# ==========================================
# --- STEP 5: INTEGRATED ANALYTICS DISPLAY ---
# ==========================================
# --- TASK 1 CALCULATIONS ---
y_pred_t1 = pipeline_t1.predict(X_test_t1)
report_t1 = classification_report(y_test_t1, y_pred_t1, target_names=le_t1.classes_.astype(str), output_dict=True)
report_df_t1 = pd.DataFrame(report_t1).transpose()

industry_stats_t1 = report_df_t1.iloc[:-3, :].copy()
industry_stats_t1['Industry Name'] = industry_stats_t1.index.map(lambda x: industry_name_map_t1.get(str(x), "Unknown"))

top_10_t1 = industry_stats_t1.sort_values(by='f1-score', ascending=False).head(10)
bottom_10_t1 = industry_stats_t1.sort_values(by='f1-score', ascending=True).head(10)

print("\n" + "="*85)
print("PART 1: TASK 1 - MACRO LEVEL DESCRIPTIVE PERFORMANCE")
print("="*85)

print("\n--- TOP 10 BEST PREDICTED MACRO INDUSTRIES ---")
print(f"{'GECS CODE':<12} | {'INDUSTRY NAME':<35} | {'F1-SCORE':<10} | {'SAMPLES'}")
print("-" * 85)
for code, row in top_10_t1.iterrows():
    print(f"{str(code):<12} | {row['Industry Name'][:35]:<35} | {row['f1-score']:<10.4f} | {int(row['support'])}")

print("\n--- BOTTOM 10 WORST PREDICTED MACRO INDUSTRIES ---")
print(f"{'GECS CODE':<12} | {'INDUSTRY NAME':<35} | {'F1-SCORE':<10} | {'SAMPLES'}")
print("-" * 85)
for code, row in bottom_10_t1.iterrows():
    print(f"{str(code):<12} | {row['Industry Name'][:35]:<35} | {row['f1-score']:<10.4f} | {int(row['support'])}")


# --- TASK 2 CALCULATIONS ---
print("\n" + "="*85)
print("PART 2: TASK 2 - SUB-CATEGORIZATION PERFORMANCE DEEP-DIVE")
print("="*85)
y_pred_t2 = pipeline_t2.predict(X_test_t2)

# Generate Task 2 dict for the final summary breakout
report_t2 = classification_report(y_test_t2, y_pred_t2, zero_division=0, output_dict=True)
print(classification_report(y_test_t2, y_pred_t2, zero_division=0))


# --- FINAL EXECUTION SUMMARY BLOCK ---
print("\n" + "="*85)
print("FINAL MODEL PERFORMANCE SUMMARY (F1-SCORES)")
print("="*85)

print("\n--- TASK 1: MACRO LEVEL GECS ---")
print(f"  • Macro Average F1-Score:  {report_t1['macro avg']['f1-score']:.4f}")
print(f"  • Weighted Average F1-Score: {report_t1['weighted avg']['f1-score']:.4f}")
print(f"  • Overall Accuracy:         {report_t1['accuracy']:.4f}")

print("\n--- TASK 2: SUB-INDUSTRY / ACTIVITY LEVEL ---")
print(f"  • Macro Average F1-Score:  {report_t2['macro avg']['f1-score']:.4f}")
print(f"  • Weighted Average F1-Score: {report_t2['weighted avg']['f1-score']:.4f}")
print(f"  • Overall Accuracy:         {report_t2['accuracy']:.4f}")

print("\n--- GLOBAL MULTI-TASK INTEGRATION METRICS ---")
print("-" * 85)
# Combined tracking metric calculation
combined_macro_f1 = (report_t1['macro avg']['f1-score'] + report_t2['macro avg']['f1-score']) / 2
combined_weighted_f1 = (report_t1['weighted avg']['f1-score'] + report_t2['weighted avg']['f1-score']) / 2

print(f"  • Global System Macro F1 (Avg of Tasks):     {combined_macro_f1:.4f}")
print(f"  • Global System Weighted F1 (Avg of Tasks):  {combined_weighted_f1:.4f}")
print(f"  • Total Records Evaluated across Pipelines:  {len(df_t1) + len(df_t2_filtered)}")
print("="*85 + "\n")

# ==========================================
# --- STEP 6: UNIFIED INTERACTIVE INTERFACE ---
# ==========================================
def run_integrated_engine():
    print("\n" + "╔" + "═"*75 + "╗")
    print("║" + " "*14 + "UNIFIED GECS MACRO & HIERARCHICAL RECO ENGINE" + " "*16 + "║")
    print("╚" + "═"*75 + "╝")
    print("Type 'exit' to cleanly close out operations.\n")

    while True:
        print("─"*77)
        desc = input("▶ ENTER NEW DESCRIPTION: ")
        if desc.lower() == 'exit': 
            print("[SYSTEM] Safely terminating deployment framework. Goodbye.")
            break
        
        try:
            revenue = float(input("▶ ENTER REVENUE ($): "))
        except ValueError:
            print("!! VALUE ERROR: Input numeric variables for tracking validation.")
            continue

        master_text = desc.lower()
        log_rev = np.log1p(max(0, revenue))
        
        # --- TASK 1 GENERATION PROCESS ---
        custom_input_t1 = pd.DataFrame({
            'master_text': [master_text],
            'log_rev_transformed': [log_rev],
            'Revenue': [revenue],
            'revenue_share': [1.0] 
        })
        
        decision_scores_t1 = pipeline_t1.decision_function(custom_input_t1)[0]
        T = 0.12 
        scaled_scores = np.clip(decision_scores_t1 / T, -20, 20)
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probabilities_t1 = (exp_scores / np.sum(exp_scores)) * 100
        
        top_3_indices = np.argsort(decision_scores_t1)[-3:][::-1]
        top_3_codes = le_t1.inverse_transform(top_3_indices)
        top_3_probs = probabilities_t1[top_3_indices]

        # --- Display Task 1 Predictions ---
        print("\n┌" + "─"*74 + "┐")
        print(f"│ {'RANK':<6} | {'GECS CODE':<12} | {'TASK 1 MACRO INDUSTRY NAME':<35} | {'CERTAINTY %':<12} │")
        print("├" + "─"*74 + "┤")
        for i, (code, prob) in enumerate(zip(top_3_codes, top_3_probs)):
            industry_name = industry_name_map_t1.get(str(code), "UNKNOWN").upper()
            short_name = (industry_name[:32] + "...") if len(industry_name) > 35 else industry_name
            rank_str = f"#{i+1}{' *' if i == 0 else '  '}"
            print(f"│ {rank_str:<6} | {code:<12} | {short_name:<35} | {prob:>11.1f}% │")
        print("└" + "─"*74 + "┘")

        # --- TASK 2 DEEP HIERARCHICAL PREDICTION ---
        pred_id_t2 = str(pipeline_t2.predict([master_text])[0]).strip()
        s_id, g_id, i_id, a_id = pred_id_t2[:3], pred_id_t2[:5], pred_id_t2[:8], pred_id_t2
        act_info = activity_map.get(a_id, {})

        print("┌" + "─"*74 + "┐")
        print(f"│ {'LEVEL':<15} | {'GECS CODE (SLICED)':<20} | {'MAPPED VALUE DESCRIPTIONS':<33} │")
        print("├" + "─"*74 + "┤")
        print(f"│ {'SECTOR':<15} | {s_id:<20} | {str(sector_map.get(s_id, 'NOT FOUND'))[:33]:<33} │")
        print(f"│ {'INDUSTRY GROUP':<15} | {g_id:<20} | {str(group_map.get(g_id, 'NOT FOUND'))[:33]:<33} │")
        print(f"│ {'INDUSTRY':<15} | {i_id:<20} | {str(industry_map.get(i_id, 'NOT FOUND'))[:33]:<33} │")
        print(f"│ {'SUB-ACTIVITY':<15} | {a_id:<20} | {str(act_info.get('name', 'NOT FOUND'))[:33]:<33} │")
        print("├" + "─"*74 + "┤")
        print(f"│ {'DEFINITION:':<74} │")
        
        raw_def = str(act_info.get('definition', 'No definition available.'))
        wrapped_def = [raw_def[i:i+70] for i in range(0, len(raw_def), 70)]
        for line in wrapped_def:
            print(f"│   {line:<71} │")
        print("└" + "─"*74 + "┘\n")

if __name__ == "__main__":
    run_integrated_engine()