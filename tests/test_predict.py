from predict import predict_species


def test_setosa_prediction():
    prediction = predict_species(5.1, 3.5, 1.4, 0.2)
    assert prediction == "Iris-setosa"


def test_virginica_prediction():
    prediction = predict_species(6.7, 3.1, 5.6, 2.4)
    assert prediction == "Iris-virginica"


def test_prediction_returns_valid_species():
    prediction = predict_species(5.9, 3.0, 5.1, 1.8)

    valid_species = {
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    }

    assert prediction in valid_species