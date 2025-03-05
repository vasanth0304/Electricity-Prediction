# Electricity-Prediction

This project predicts the energy consumption of buildings based on factors such as building type, square footage, number of occupants, and average temperature. The model is built using machine learning techniques with `scikit-learn` and is deployed with `Gradio` for an interactive user interface.

## Features

- **Machine Learning Model**: Uses `LinearRegression` to predict energy consumption.
- **Text & Numeric Feature Handling**: Uses `TfidfVectorizer` for text-based building types and standard numeric features.
- **Interactive UI**: Provides a simple Gradio-based web interface for predictions
Ensure you have Python installed along with the required libraries:
pip install numpy pandas scikit-learn gradio scipy
4. The Gradio interface will launch, allowing you to enter building details and get energy consumption predictions.

## File Structure

- `energy_proj.py` - The main script containing the machine learning model and UI.
- `test_energy_data.csv` - The dataset used for training and testing.

## Acknowledgments

- `scikit-learn` for machine learning functionalities.
- `Gradio` for interactive UI.
- `pandas` and `numpy` for data processing.

## Future Improvements

- Support for additional building parameters.
- Integration with live weather data.
- Deployment as a web app.
