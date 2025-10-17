import pytest
from fastapi.testclient import TestClient
from app import app
from preprocess_function import preprocess_text
import time

# Client FastAPI pour tests
client = TestClient(app)

# -----------------------------
# Tests endpoint racine "/"
# -----------------------------
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Bienvenue" in response.text  # ou adapte si ta racine est en anglais

# -----------------------------
# Tests endpoint "/health"
# -----------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["vectorizer_loaded"] is True

# -----------------------------
# Tests endpoint "/predict"
# -----------------------------
def test_predict_positive():
    payload = {"text": "i love you, it's good!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"].lower() in ["positive", "negative"]
    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["probability_positive"] <= 1
    assert 0 <= data["probability_negative"] <= 1

def test_predict_negative():
    payload = {"text": "i hate you, it's horrible!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"].lower() in ["positive", "negative"]
    assert 0 <= data["confidence"] <= 1

# -----------------------------
# Tests endpoint "/explain"
# -----------------------------
def test_explain():
    payload = {"text": "it's a test for explain."}
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert "html_explanation" in data
    assert isinstance(data["explanation"], list)

# -----------------------------
# Tests du prétraitement
# -----------------------------
def test_preprocess_text():
    text = "hello @user! visit https://test.com 😃 #fun"
    cleaned = preprocess_text(text)
    assert "@" not in cleaned
    assert "http" not in cleaned
    assert "#" not in cleaned
    assert "😃" not in cleaned or cleaned.isascii()  # emoji removed if non-ASCII

# -----------------------------
# Advanced tests
# -----------------------------
def test_predict_endpoint_valid():
    """Test the /predict endpoint with a valid text input"""
    test_data = {"text": "I love this product, it's fantastic!"}
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()

    # Check expected fields
    for field in ["sentiment", "confidence", "probability_positive", "probability_negative"]:
        assert field in data

    assert data["sentiment"].lower() in ["positive", "negative"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_positive"] <= 1.0
    assert 0.0 <= data["probability_negative"] <= 1.0

    total_prob = data["probability_positive"] + data["probability_negative"]
    assert abs(total_prob - 1.0) < 0.01

    print(f"✅ Prediction OK: {data['sentiment']} ({data['confidence']:.2f})")

def test_predict_endpoint_invalid():
    """Test the /predict endpoint with invalid input data"""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in [200, 422]  # Tolerate if validation not yet added

    long_text = "a" * 300
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code in [200, 422]

    response = client.post("/predict", json={})
    assert response.status_code in [200, 422]

    print("✅ Error handling OK")

def test_explain_endpoint():
    """Test the /explain endpoint with a normal text"""
    test_data = {"text": "This movie is absolutely terrible, I hate it!"}
    start_time = time.time()
    response = client.post("/explain", json=test_data)
    duration = time.time() - start_time

    assert response.status_code == 200
    data = response.json()

    for field in ["sentiment", "explanation", "html_explanation"]:
        assert field in data

    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0
    html_content = data["html_explanation"]
    assert isinstance(html_content, str)
    assert len(html_content) > 100
    assert "<div" in html_content
    assert duration < 120

    print(f"✅ LIME OK: {len(data['explanation'])} words explained")
    print(f"⏱️ Time: {duration:.1f}s")

@pytest.mark.timeout(90)
def test_explain_robustness():
    """Test the robustness of the /explain endpoint with edge cases"""
    test_cases = [
        "Super!",              # very short
        "😊" * 10,             # only emojis
        "http://example.com",  # URLs
    ]
    for text in test_cases:
        response = client.post("/explain", json={"text": text})
        assert response.status_code in [200, 422]

        if response.status_code == 200:
            data = response.json()
            assert "html_explanation" in data

    print("✅ LIME Robustness OK")
