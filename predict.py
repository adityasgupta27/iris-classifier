import joblib
import pandas as pd

model = joblib.load("iris_model.joblib")


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }])

    prediction = model.predict(features)

    return prediction[0]