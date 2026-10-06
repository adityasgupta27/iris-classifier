import streamlit as st

from predict import predict_species


st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸"
)

st.title("Iris Flower Classifier")
st.write("Enter the flower measurements to predict its species.")


sepal_length = st.number_input(
    "Sepal length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)


if st.button("Predict"):
    prediction = predict_species(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    st.success(f"Predicted species: {prediction}")