import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# --- STEP 1: LOAD DATA ---
df = pd.read_csv(r'C:\Users\gsudh\Documents\depaul\courses\capstone\week4\task1_gecs_classification_final.csv')
df_blanks = pd.read_csv(r'C:\Users\gsudh\Documents\depaul\courses\capstone\week4\Empty cell info morningstar_gecs_industries.csv')

# --- STEP 2: KNOWLEDGE-BASED IMPUTATION ---
df['MstarGlobal'] = df['MstarGlobal'].astype(str)
df_blanks['GECS_Code'] = df_blanks['GECS_Code'].astype(str)
definition_map = dict(zip(df_blanks['GECS_Code'], df_blanks['Description']))

# Fill empty segment descriptions with official PDF definitions
df['SegmentDescription'] = df['SegmentDescription'].fillna(df['MstarGlobal'].map(definition_map))

# --- STEP 3: FEATURE ENGINEERING ---
df['master_text'] = (df['SegmentName'].fillna('') + " " + 
                     df['LongProfile'].fillna('') + " " + 
                     df['SegmentDescription'].fillna(''))

# Log transform revenue to normalize scale
df['log_total_rev'] = np.log1p(df['total_revenue_company_as_of'].clip(lower=0))

le = LabelEncoder()
y = le.fit_transform(df['MstarGlobal'])
X = df[['master_text', 'log_total_rev', 'Revenue', 'revenue_share']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- STEP 4: ENHANCED HYBRID PIPELINE ---
# Adding custom business stop words to remove "noise"
business_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + \
                      ['company', 'incorporated', 'limited', 'services', 'solutions', 'operations', 'segment']

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            max_features=10000, 
            stop_words=business_stop_words, 
            ngram_range=(1, 3), # Captures phrases like "Real Estate Trust"
            use_idf=True, 
            smooth_idf=True
        ), 'master_text'),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ]), ['log_total_rev', 'Revenue', 'revenue_share'])
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LinearSVC(class_weight='balanced', random_state=42, max_iter=3000, C=0.8))
])

# --- STEP 5: TRAINING & EVALUATION ---
print("\n[SYSTEM] Training model for 145 GECS industries...")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str), output_dict=True)
report_df = pd.DataFrame(report).transpose()

industry_perf = report_df.iloc[:-3, :].copy()
summary_performance = report_df.tail(3).copy()
sorted_top_20 = industry_perf.sort_values(by='f1-score', ascending=False).head(20)

# --- DISPLAY METRICS ---
print("\n" + "="*60)
print("TOP 20 INDUSTRIES BY F1-SCORE")
print("="*60)
print(sorted_top_20[['f1-score', 'precision', 'recall', 'support']])

print("\n" + "="*60)
print("GLOBAL MODEL SCORECARD")
print("="*60)
print(f"OVERALL ACCURACY:      {report_df.loc['accuracy', 'f1-score']:.4f}")
print(f"MACRO AVG F1-SCORE:    {report_df.loc['macro avg', 'f1-score']:.4f}  <-- GOAL: 0.75")
print(f"WEIGHTED AVG F1-SCORE: {report_df.loc['weighted avg', 'f1-score']:.4f}")
print("="*60)

# --- STEP 6: PRE-STEP FOR UI ---
try:
    df_lookup = pd.read_csv(r'C:\Users\gsudh\Documents\depaul\courses\capstone\week3\Empty cell info morningstar_gecs_industries.csv')
    df_lookup.columns = df_lookup.columns.str.strip()
    industry_name_map = dict(zip(df_lookup['GECS_Code'].astype(str), df_lookup['Industry_Name']))
except Exception as e:
    print(f"Warning: Industry Name lookup failed: {e}")
    industry_name_map = {}

# --- STEP 7: INTERACTIVE "MORNINGSTAR TERMINAL" UI ---
def run_interactive_prediction():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*13 + "MORNINGSTAR GECS INTELLIGENCE ENGINE" + " "*11 + "║")
    print("╚" + "═"*58 + "╝")
    print("Type 'exit' in Segment Name to quit.\n")

    while True:
        print("─"*35)
        name = input("▶ SEGMENT NAME: ")
        if name.lower() == 'exit': break
        
        profile = input("▶ LONG PROFILE: ")
        description = input("▶ DESCRIPTION:  ")
        
        try:
            total_rev = float(input("▶ TOTAL REVENUE ($): "))
            rev_share = float(input("▶ REVENUE SHARE (0-1): "))
            segment_rev = total_rev * rev_share
        except ValueError:
            print("!! ERROR: Please enter valid numeric values for revenue.")
            continue

        # Prepare manual input
        master_text = f"{name} {profile} {description}"
        log_rev = np.log1p(max(0, total_rev))
        
        custom_input = pd.DataFrame({
            'master_text': [master_text],
            'log_total_rev': [log_rev],
            'Revenue': [segment_rev],
            'revenue_share': [rev_share]
        })
        
        # Predict
        pred_num = pipeline.predict(custom_input)
        gecs_code = le.inverse_transform(pred_num)[0]
        industry_name = industry_name_map.get(str(gecs_code), "UNKNOWN INDUSTRY")
        
        # Confidence Score Calculation
        decision_scores = pipeline.decision_function(custom_input)
        confidence = np.max(decision_scores)

        # UI Results Box
        print("\n" + "┌" + "─"*50 + "┐")
        print(f"│ PREDICTED GECS: {gecs_code:<32} │")
        print(f"│ INDUSTRY TYPE:  {industry_name.upper():<32} │")
        print(f"│ CONFIDENCE:     {confidence:>32.2f} │")
        print("└" + "─"*50 + "┘\n")

if __name__ == "__main__":
    run_interactive_prediction()