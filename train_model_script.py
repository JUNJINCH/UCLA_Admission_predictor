from src.data_loader import load_data
from src.model_trainer import train_model
import os

# Load data
df = load_data("data/Admission.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

# Debug: Show columns
print("Columns in the dataset:")
print(df.columns.tolist())

# Handle different column name variations
target_col = None
for col in df.columns:
    if "admit" in col.lower():
        target_col = col
        break

if not target_col:
    raise ValueError("Could not find admission chance column. Available columns: " + str(df.columns.tolist()))

# Create binary target variable
df["Admitted"] = (df[target_col] >= 0.75).astype(int)

# Prepare features (drop non-predictive columns)
X = df.drop(columns=[target_col, "Admitted", "Serial_No"])
y = df["Admitted"]

# Train model
os.makedirs("models", exist_ok=True)
model, cm = train_model(X, y, "models/admission_model.pkl")

print("Model trained. Confusion Matrix:")
print(cm)