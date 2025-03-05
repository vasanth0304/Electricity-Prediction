import gradio as gd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
df = pd.read_csv(r"C:\Users\ashik\OneDrive\Desktop\energy project\test_energy_data.csv")
x_text = df[["Building Type"]]
x_numeric = df[["Square Footage", "Number of Occupants", "Average Temperature"]]
y = df["Energy Consumption"].astype(float)
tfidf_vectorizer = TfidfVectorizer()
x_text_transformed = tfidf_vectorizer.fit_transform(x_text["Building Type"])

x_combined = hstack([x_text_transformed, x_numeric])
x_train, x_test, y_train, y_test = train_test_split(x_combined, y, test_size=0.3, random_state=9)
model = LinearRegression()
model.fit(x_train, y_train)
def calc(building, sq_foot, no_of_occup, temp):
    building_transformed = tfidf_vectorizer.transform([building])
    numeric_input = np.array([[float(sq_foot), float(no_of_occup), float(temp)]])
    combined_input = hstack([building_transformed, numeric_input])
    predicted_energy = model.predict(combined_input)
    return f"${predicted_energy[0]:,.2f}"

inputs = [
    gd.Textbox(label="Building Type"),
    gd.Textbox(label="Square Footage"),
    gd.Textbox(label="Number of Occupants"),
    gd.Textbox(label="Average Temperature")
]
interface = gd.Interface(
    fn=calc,
    inputs=inputs,
    outputs=gd.Textbox(label="Predicted Energy Consumption")
)
interface.launch()
