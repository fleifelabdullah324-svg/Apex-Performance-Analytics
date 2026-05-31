import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Loading the fixed dataset (94 columns)
print("Loading Dataset...")
df = pd.read_csv('../data/processed/The_Flawless_Tuning_ERP_V20_Fixed.csv')
df.columns = df.columns.str.strip()

# 2. Handling Missing Values
print("Handling missing values...")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Standard')
    else:
        df[col] = df[col].fillna(df[col].mean())

# 3. Separating Features and Target
X = df.drop('Safety_Status', axis=1)
y = df['Safety_Status']

# 4. Applying Label Encoding for text columns
print("Applying Label Encoding...")
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])

# 5. Applying Min-Max Normalization
print("Applying Min-Max Normalization...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train/Test Split (80/20) with Stratification
print("Splitting Data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 7. Training Random Forest Classifier
print("Training Random Forest Classifier...")
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15)
rf.fit(X_train, y_train)

# 8. Evaluating the Model
print("Evaluating Model...")
y_pred = rf.predict(X_test)

# 9. Printing Final Results
print("="*50)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("="*50)
print("Confusion Matrix (Safe, Risky, Critical Failure):")
print(confusion_matrix(y_test, y_pred, labels=['Safe', 'Risky', 'Critical Failure']))
print("="*50)
print("Classification Report:")
print(classification_report(y_test, y_pred))  
