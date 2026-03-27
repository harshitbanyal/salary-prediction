import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# same dataset
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([15000,20000,25000,30000,35000,40000,50000,60000,70000,80000])

model = LinearRegression()
model.fit(X, y)

st.title("💼 Salary Prediction App")

exp = st.slider("Experience (years)", 1, 10, 2)

if st.button("Predict Salary"):
    salary = model.predict([[exp]])
    st.success(f"Predicted Salary: ₹{salary[0]:.2f}")
