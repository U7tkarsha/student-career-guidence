from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# Load your dataset here
df = pd.read_csv("student_stream_profession.csv")  # Replace with actual path

# Profession grouping
profession_map = {
    'Software Engineer': 'Engineering',
    'Data Scientist': 'Engineering',
    'Doctor': 'Medical',
    'Pharmacist': 'Medical',
    'CA': 'Business',
    'Accountant': 'Business',
    'Designer': 'Creative',
    'Psychologist': 'Creative',
    'Journalist': 'Creative'
}

df = df[df['profession'].isin(profession_map.keys())]
df['profession_group'] = df['profession'].map(profession_map)

# Label Encoding
le_prof = LabelEncoder()
df['profession_encoded'] = le_prof.fit_transform(df['profession_group'])
le_stream = LabelEncoder()
df['stream_encoded'] = le_stream.fit_transform(df['stream'])
le_family = LabelEncoder()
df['family_background_encoded'] = le_family.fit_transform(df['family_background'])

# Features
features = ['grade_10_marks', 'interest_math', 'interest_bio', 'interest_econ',
            'coding', 'drawing', 'sports', 'stream_encoded', 'family_background_encoded']
X = df[features]
y = df['profession_encoded']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le_prof.classes_)
print("=== Profession Prediction (Grouped) ===")
print("Accuracy:", accuracy)
print()
# print("Classification Report:\n", report) 
# Save model and encoders
joblib.dump(model, 'model.pkl')
joblib.dump(le_prof, 'le_prof.pkl')
joblib.dump(le_stream, 'le_stream.pkl')
joblib.dump(le_family, 'le_family.pkl')