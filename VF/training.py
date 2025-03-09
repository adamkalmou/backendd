import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load the CSV data with proper delimiter
csv_file = "C:/Users/ADAM/Desktop/Nouveau dossier (9)/projet pfe vf/VF/a.csv"
df = pd.read_csv(csv_file, delimiter=';')  # Use ';' as the separator

# Clean the longitude and latitude columns (replace commas with dots, then convert to float)
df['LONG_D'] = df['LONG_D'].str.replace(',', '.').astype(float)
df['LAT_D'] = df['LAT_D'].str.replace(',', '.').astype(float)

# Clean environmental columns (Salinite, Temp, DO, pH), replace commas with dots and convert to float
df['Salinite'] = df['Salinite'].str.replace(',', '.').astype(float)
df['Temp'] = df['Temp'].str.replace(',', '.').astype(float)
df['DO'] = df['DO'].str.replace(',', '.').astype(float)
df['pH'] = df['pH'].str.replace(',', '.').astype(float)

# Extract relevant columns for features and target variable
features = ['LAT_D', 'LONG_D', 'Salinite', 'Temp', 'DO', 'pH']
target = 'Pres_Esp2'

X = df[features]  # Feature columns
y = df[target]    # Target column (Pres_Esp1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model using RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Save the model for later use
import joblib
joblib.dump(model, 'species_prediction_model_ESP2.pkl')
