from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)


model = pickle.load(open('model_final.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_final.pkl', 'rb'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\b(reuters|ap|associated press|afp)\b", "", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_news_smart(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    if vec.sum() == 0:
        return "Real", 0.5, [0.5, 0.5]
    
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    fake_prob, real_prob = probabilities[0], probabilities[1]
    confidence = probabilities.max()
    
    # Smart decision making
    if confidence >= 0.55:
        label = "Real" if prediction == 1 else "Fake"
        return label, confidence, probabilities.tolist()
    elif real_prob >= 0.45:
        return "Real", real_prob, probabilities.tolist()
    else:
        return "Real", real_prob, probabilities.tolist()

@app.route('/')
def home():
    return "Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data.get('text', '')

    if not news_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
       
        prediction, confidence, probabilities = predict_news_smart(news_text)
        
        return jsonify({
            'prediction': prediction,  
            'confidence': round(confidence, 3),
            'probabilities': {
                'fake': round(probabilities[0], 3),
                'real': round(probabilities[1], 3)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)