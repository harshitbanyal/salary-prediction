# ==========================================
# 💼 Salary Prediction Project
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ------------------------------
# Create Dataset (NO INTERNET)
# ------------------------------
data = {
    "experience": [1,2,3,4,5,6,7,8,9,10],
    "salary": [15000,20000,25000,30000,35000,40000,50000,60000,70000,80000]
}

df = pd.DataFrame(data)

print("📊 Dataset:\n")
print(df)

# ------------------------------
# Visualization
# ------------------------------
plt.scatter(df["experience"], df["salary"])
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()

# ------------------------------
# Features & Target
# ------------------------------
X = df[["experience"]]
y = df["salary"]

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Model Training
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Prediction & Accuracy
# ------------------------------
y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)
print(f"\n✅ Model Accuracy (R2 Score): {score*100:.2f}%")

# ------------------------------
# Plot Regression Line
# ------------------------------
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Regression Line")
plt.show()

# ------------------------------
# Manual Prediction
# ------------------------------
exp = float(input("\nEnter years of experience: "))

predicted_salary = model.predict([[exp]])

print(f"\n💰 Predicted Salary: ₹{predicted_salary[0]:.2f}")