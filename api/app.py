# app.py
import os
import re
import string
import json
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from preprocess_function import preprocess_text
from fastapi.responses import HTMLResponse
from lime.lime_text import LimeTextExplainer
# Charger emojis et stopwords depuis JSON
ARTIFACTS_DIR = "api_artifacts"

with open(os.path.join(ARTIFACTS_DIR, "emoji_dict.json"), encoding="utf-8") as f:
    emojis = json.load(f)

with open(os.path.join(ARTIFACTS_DIR, "stopwords_list.json"), encoding="utf-8") as f:
    stopwords_list = json.load(f)

# Charger modèle et vectorizer
try:
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "sentiment_model.joblib"))
    vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib"))
except Exception as e:
    print(f"Erreur chargement modèle/vectorizer: {e}")
    model = None
    vectorizer = None

# FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Request model
class TweetRequest(BaseModel):
    text: str

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }


# Predict endpoint
@app.post("/predict")
def predict(request: TweetRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    try:
        cleaned_text = preprocess_text(request.text, emojis=emojis, stopwords_list=stopwords_list)
        X = vectorizer.transform([cleaned_text])
        proba = model.predict_proba(X)[0]
        # Dans /predict
        sentiment = "positive" if proba[1] >= proba[0] else "negative"

        confidence = max(proba)
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probability_positive": proba[1],
            "probability_negative": proba[0]
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Explain endpoint (LIME)
from lime.lime_text import LimeTextExplainer

@app.post("/explain")
def explain(request: TweetRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        cleaned_text = preprocess_text(request.text, emojis=emojis, stopwords_list=stopwords_list)

        # ✅ Handle very short or empty text
        if len(cleaned_text.strip().split()) < 2:
            raise HTTPException(status_code=422, detail="Text too short for explanation")

        class_names = ["negative", "positive"]
        explainer = LimeTextExplainer(class_names=class_names)

        # Wrapper for LIME to preprocess each sample
        def predict_proba_lime(texts):
            X = vectorizer.transform([preprocess_text(t, emojis=emojis, stopwords_list=stopwords_list) for t in texts])
            return model.predict_proba(X)

        exp = explainer.explain_instance(cleaned_text, predict_proba_lime, num_features=10)
        html_exp = exp.as_html()
        sentiment = "positive" if model.predict(vectorizer.transform([cleaned_text]))[0] == 1 else "negative"

        return {
            "sentiment": sentiment,
            "explanation": exp.as_list(),
            "html_explanation": html_exp
        }
    except HTTPException:
        # Reraise HTTP errors (like 422 for short text)
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    
@app.get("/", response_class=HTMLResponse)
def root():
    html_content = """
    <html>
        <head>
            <title>Sentiment Analysis API</title>
            <style>
                body { font-family: Arial, sans-serif; background-color: #f0f8f5; color: #333; padding: 40px; }
                h1 { color: #20a08d; }
                p { font-size: 18px; }
                ul { font-size: 16px; }
                code { background-color: #e0f7f2; padding: 2px 4px; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>Bienvenue sur l'API Sentiment Analysis 🌟</h1>
            <p>Cette API vous permet de :</p>
            <ul>
                <li>Obtenir la prédiction de sentiment pour un texte via <code>/predict</code></li>
                <li>Obtenir une explication LIME du modèle via <code>/explain</code></li>
                <li>Vérifier l'état de santé de l'API via <code>/health</code></li>
            </ul>
            <p>La documentation interactive est disponible sur <a href="/docs">Swagger UI</a>.</p>
            <p>Amusez-vous bien et explorez vos textes avec notre modèle ML ! 😊</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)