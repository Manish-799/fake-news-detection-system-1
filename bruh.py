import pickle
import re
import numpy as np


print("Loading model and vectorizer...")
model = pickle.load(open('model_final.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_final.pkl', 'rb'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\b(reuters|ap|associated press|afp)\b", "", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def debug_prediction(text):
    print("\n" + "="*50)
    print(f"Testing: '{text}'")
    print("="*50)
    
    
    cleaned = clean_text(text)
    print(f"Cleaned text: '{cleaned}'")
    
   
    vec = vectorizer.transform([cleaned])
    print(f"Vector sum (non-zero features): {vec.sum()}")
    print(f"Vector shape: {vec.shape}")
    
    if vec.sum() == 0:
        print("NO FEATURES FOUND - text is out of vocabulary")
        return
    
    
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    fake_prob, real_prob = probabilities[0], probabilities[1]
    
    print(f"Raw prediction: {prediction} (0=Fake, 1=Real)")
    print(f"Probabilities - Fake: {fake_prob:.3f}, Real: {real_prob:.3f}")
    print(f"Confidence: {probabilities.max():.3f}")
    
   
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = vec.nonzero()[1]
    
    print(f"\nTop features found in this text:")
    for idx in nonzero_indices[:10]: 
        feature_name = feature_names[idx]
        feature_value = vec[0, idx]
        print(f"  '{feature_name}': {feature_value:.4f}")
    
    
    confidence = probabilities.max()
    if confidence >= 0.55:
        label = "Real" if prediction == 1 else "Fake"
        print(f"FINAL PREDICTION: {label} (High confidence)")
    elif real_prob >= 0.45:
        print(f"FINAL PREDICTION: Real (Leaning toward Real)")
    else:
        print(f"FINAL PREDICTION: Real (Default safety)")
    
    return prediction, probabilities

test_texts = [
    "Scientists discover new method for clean energy",
    "The president signed a new economic bill today",
    "Federal reserve announces interest rate changes",
    "SHOCKING: This one trick will make you rich", 
    "CLICK HERE TO WIN FREE IPHONE",
    "Study shows benefits of daily exercise",
    "Breaking news from the white house"
]

print("DEBUGGING MODEL PREDICTIONS")
print("="*60)

for text in test_texts:
    debug_prediction(text)


print("\n" + "="*60)
print("MODEL ANALYSIS")
print("="*60)

feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

print("Top 10 REAL indicators (positive coefficients):")
real_indices = coef.argsort()[-10:][::-1]
for idx in real_indices:
    print(f"  '{feature_names[idx]}': {coef[idx]:.3f}")

print("\nTop 10 FAKE indicators (negative coefficients):")
fake_indices = coef.argsort()[:10]
for idx in fake_indices:
    print(f"  '{feature_names[idx]}': {coef[idx]:.3f}")

print(f"\nModel intercept: {model.intercept_[0]:.3f}")