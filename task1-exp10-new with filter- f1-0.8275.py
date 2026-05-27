import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# --- STEP 1: LOAD DATA ---
print("System: Loading datasets for Task 1...")
try:
    # Task 1 Main Data
    df = pd.read_csv(r'C:\Users\gsudh\OneDrive\Documents\capstone\capstone\week7\task1_gecs_classification_final.csv')
    # Lookup file for Industry Names and Blank Descriptions
    df_lookup = pd.read_csv(r'C:\Users\gsudh\OneDrive\Documents\capstone\capstone\week7\Empty cell info morningstar_gecs_industries.csv')
    
    # Clean headers
    df_lookup.columns = df_lookup.columns.str.strip()
    industry_name_map = dict(zip(df_lookup['GECS_Code'].astype(str), df_lookup['Industry_Name']))
    definition_map = dict(zip(df_lookup['GECS_Code'].astype(str), df_lookup['Description'].fillna('No official definition.')))

except Exception as e:
    print(f"CRITICAL ERROR: Could not load data files. \n{e}")
    import sys
    sys.exit()

# --- STEP 1b: FILTER FOR LARGEST REVENUE SHARE SEGMENTS ---
# Convert the filter column to string/uppercase to cleanly catch 'TRUE', 'True', or boolean True values
if 'is_largest_share_segment' in df.columns:
    initial_row_count = len(df)
    df = df[df['is_largest_share_segment'].astype(str).str.upper() == 'TRUE']
    filtered_row_count = len(df)
    print(f"✓ Filtering Guard: Retained {filtered_row_count} rows out of {initial_row_count} where 'is_largest_share_segment' is True.")
else:
    print("⚠ WARNING: 'is_largest_share_segment' column not found. Skipping filter step.")

# --- STEP 2: KNOWLEDGE-BASED IMPUTATION ---
df['MstarGlobal'] = df['MstarGlobal'].astype(str)
df['SegmentDescription'] = df['SegmentDescription'].fillna(df['MstarGlobal'].map(definition_map))

# --- STEP 3: FEATURE ENGINEERING (TEXT ONLY) ---
df['master_text'] = (df['SegmentName'].fillna('') + " " + 
                     df['LongProfile'].fillna('') + " " + 
                     df['SegmentDescription'].fillna('')).str.lower()

# --- STEP 3b: STRATIFICATION GUARD FOR RARE CLASSES ---
X_raw = df['master_text']
y_raw = df['MstarGlobal']

# Count occurrences of each industry class
class_counts = y_raw.value_counts()
valid_classes = class_counts[class_counts >= 2].index

# Filter down to classes that have at least 2 samples to allow safe stratification split
mask = y_raw.isin(valid_classes)
X_filtered = X_raw[mask]
y_filtered = y_raw[mask]

dropped_rows = len(y_raw) - len(y_filtered)
if dropped_rows > 0:
    print(f"ℹ Stratification Guard: Dropped {dropped_rows} rows belonging to single-member industries.")
    print(f"  Active dataset size: {len(y_filtered)} rows across {len(valid_classes)} distinct industries.")

le = LabelEncoder()
y = le.fit_transform(y_filtered)
X = X_filtered

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- STEP 4: TEXT-ONLY PIPELINE ---
business_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + \
                      ['company', 'incorporated', 'limited', 'services', 'solutions', 'operations', 'segment']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words=business_stop_words, ngram_range=(1, 3))),
    ('classifier', LinearSVC(class_weight='balanced', random_state=42, max_iter=3000, C=0.8))
])

# --- STEP 5: TRAINING & ANALYTICS GENERATION ---
print(f"\n[SYSTEM] Training Task 1 Model (Text-Only - {len(le.classes_)} Industries)...")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Generate report for descriptive analytics
report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str), output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()

# Filter for just the industry classes
industry_stats = report_df.iloc[:-3, :].copy()
industry_stats['Industry Name'] = industry_stats.index.map(lambda x: industry_name_map.get(str(x), "Unknown"))

# --- STEP 6: PRINT DESCRIPTIVE ANALYTICS ---
top_10 = industry_stats.sort_values(by='f1-score', ascending=False).head(10)
bottom_10 = industry_stats.sort_values(by='f1-score', ascending=True).head(10)

print("\n" + "="*85)
print("PART 1: DESCRIPTIVE ANALYTICS - MODEL PERFORMANCE (TEXT ONLY)")
print("="*85)

print("\n--- TOP 10 BEST PREDICTED INDUSTRIES (Highest F1-Score) ---")
print(f"{'GECS CODE':<12} | {'INDUSTRY NAME':<35} | {'F1-SCORE':<10} | {'SAMPLES'}")
print("-" * 85)
for code, row in top_10.iterrows():
    print(f"{str(code):<12} | {row['Industry Name'][:35]:<35} | {row['f1-score']:<10.4f} | {int(row['support'])}")

print("\n--- BOTTOM 10 WORST PREDICTED INDUSTRIES (Lowest F1-Score) ---")
print(f"{'GECS CODE':<12} | {'INDUSTRY NAME':<35} | {'F1-SCORE':<10} | {'SAMPLES'}")
print("-" * 85)
for code, row in bottom_10.iterrows():
    print(f"{str(code):<12} | {row['Industry Name'][:35]:<35} | {row['f1-score']:<10.4f} | {int(row['support'])}")

print("\n" + "="*85)
print("GLOBAL PERFORMANCE SUMMARY")
print("-" * 85)
print(f"Total Records Extracted (True Share): {len(df)}")
print(f"Unique GECS Industries Trained:       {len(le.classes_)}")
print(f"Overall Model Accuracy:              {report['accuracy']:.4f}")
print(f"Macro F1-Score (Avg):                {report['macro avg']['f1-score']:.4f}")
print("="*85)

# --- STEP 7: INTERACTIVE TOP-3 ADVISORY ENGINE ---
def run_interactive_engine():
    print("\n" + "╔" + "═"*75 + "╗")
    print("║" + " "*16 + "MORNINGSTAR GECS TOP-3 RECOMMENDATION ENGINE" + " "*15 + "║")
    print("╚" + "═"*75 + "╝")
    print("Type 'exit' to quit.\n")

    while True:
        print("─"*40)
        desc = input("▶ ENTER DESCRIPTION: ")
        if desc.lower() == 'exit': break
        if not desc.strip():
            print("!! ERROR: Please enter a valid description.")
            continue

        master_text = desc.lower()
        
        # Prepare input as a series or list matching text format expected by pipeline
        custom_input = [master_text]
        
        # Get raw decision scores
        decision_scores = pipeline.decision_function(custom_input)[0]
        
        # --- CALIBRATED PREDICTION CERTAINTY ---
        T = 0.12 
        scaled_scores = decision_scores / T
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probabilities = (exp_scores / np.sum(exp_scores)) * 100
        
        # Get Top 3 indices
        top_3_indices = np.argsort(decision_scores)[-3:][::-1]
        top_3_codes = le.inverse_transform(top_3_indices)
        top_3_probs = probabilities[top_3_indices]

        # Standardized box layout to ensure pipes align perfectly
        print("\n" + "┌" + "─"*74 + "┐")
        print(f"│ {'RANK':<6} | {'GECS CODE':<12} | {'INDUSTRY NAME':<35} | {'CERTAINTY %':<12} │")
        print("├" + "─"*74 + "┤")
        
        for i, (code, prob) in enumerate(zip(top_3_codes, top_3_probs)):
            industry_name = industry_name_map.get(str(code), "UNKNOWN").upper()
            # Clean up the name for the fixed-width column
            short_name = (industry_name[:32] + "...") if len(industry_name) > 35 else industry_name
            
            rank_tag = f"#{i+1}"
            star = " *" if i == 0 else "  "
            rank_str = f"{rank_tag}{star}"
            
            # Print with precise spacing
            print(f"│ {rank_str:<6} | {code:<12} | {short_name:<35} | {prob:>11.1f}% │")
                
        print("└" + "─"*74 + "┘\n")

if __name__ == "__main__":
    run_interactive_engine()