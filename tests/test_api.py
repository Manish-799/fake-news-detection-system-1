import pytest

from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert data["documentation"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["vectorizer_loaded"] is True
    assert data["metrics_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "Logistic Regression"
    assert data["accuracy"] > 0
    assert data["roc_auc"] > 0
    assert data["train_rows"] > 0
    assert data["test_rows"] > 0
    assert data["vocabulary_size"] > 0
    assert 0 <= data["uncertain_threshold"] <= 1


def test_predict_rejects_blank_text():
    response = client.post(
        "/predict",
        json={
            "text": "   ",
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["detail"] == "Article text cannot be blank."


def test_predict_rejects_unknown_features():
    response = client.post(
        "/predict",
        json={
            "text": "qzxvplmokn vbnmqwerty zxcasdfghjkl",
        },
    )

    assert response.status_code == 422


def test_predict_returns_valid_response():
    article = (
        "The government said the president met officials "
        "at the White House on Monday to discuss policy."
    )

    response = client.post(
        "/predict",
        json={
            "text": article,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["verdict"] in {
        "LIKELY_REAL",
        "LIKELY_FAKE",
        "UNCERTAIN",
    }

    assert data["model_prediction"] in {
        "REAL",
        "FAKE",
    }

    assert data["predicted_label"] in {
        0,
        1,
    }


def test_prediction_probabilities_are_valid():
    article = (
        "The government said the president met officials "
        "at the White House on Monday to discuss policy."
    )

    response = client.post(
        "/predict",
        json={
            "text": article,
        },
    )

    assert response.status_code == 200

    data = response.json()

    fake_probability = data["probabilities"]["fake"]
    real_probability = data["probabilities"]["real"]

    assert 0 <= fake_probability <= 1
    assert 0 <= real_probability <= 1

    assert fake_probability + real_probability == pytest.approx(
        1.0
    )


def test_feature_contribution_directions():
    article = (
        "The government said the president met officials "
        "at the White House on Monday to discuss policy."
    )

    response = client.post(
        "/predict",
        json={
            "text": article,
        },
    )

    assert response.status_code == 200

    data = response.json()

    fake_indicators = data["explanation"]["fake_indicators"]
    real_indicators = data["explanation"]["real_indicators"]

    for feature in fake_indicators:
        assert feature["contribution"] < 0

    for feature in real_indicators:
        assert feature["contribution"] > 0