import pandas as pd
df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df.info()
df.describe()

# Drop kolom tidak relevan seperti ID dan ZipCode
df = df.drop(columns=["ID", "ZIP Code"])

# Tangani missing value (jika ada)
df = df.dropna()

# Normalisasi
from sklearn.preprocessing import StandardScaler
X = df.drop(columns=['Personal Loan'])  # fitur tanpa label
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
