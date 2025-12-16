import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import re
import numpy as np

#Loading Data
true = pd.read_csv(r'C:\CSE\fake_news_detector\dataset\True.csv')
fake = pd.read_csv(r'C:\CSE\fake_news_detector\dataset\Fake.csv')

true['label'] = 1
fake['label'] = 0

true['content'] = true['title'].astype(str)
fake['content'] = fake['title'].astype(str)

data = pd.concat([true[['content', 'label']], fake[['content', 'label']]], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\b(reuters|ap|associated press|afp)\b", "", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

data['content'] = data['content'].apply(clean_text)
data = data[data['content'].str.strip() != '']

print("Dataset balance:")
print(data['label'].value_counts())

X = data['content']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.8,      
    min_df=5,        
    max_features=8000, 
    ngram_range=(1, 2), 
    sublinear_tf=True  
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    C=0.1,           
    solver='lbfgs',  
    random_state=42
)
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)
pred_proba = model.predict_proba(X_test_vec)
acc = accuracy_score(y_test, pred)

print(f"\nModel Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, pred))


print(f"\nConfidence Statistics:")
print(f"Average confidence: {pred_proba.max(axis=1).mean():.3f}")
print(f"Min confidence: {pred_proba.max(axis=1).min():.3f}")
print(f"Max confidence: {pred_proba.max(axis=1).max():.3f}")


pickle.dump(model, open('model_final.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer_final.pkl', 'wb'))


def predict_news_smart(text, model, vectorizer, confidence_threshold=0.55, real_bias_threshold=0.45):
    """
    Predicts fake news with smart handling of uncertain cases:
    - High confidence: Use model prediction
    - Medium confidence: If leaning toward Real, accept it
    - Low confidence: Default to Real (safer assumption)
    """
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    if vec.sum() == 0:
        return "Real (no features)", 0.5, [0.5, 0.5]
    
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    fake_prob, real_prob = probabilities[0], probabilities[1]
    confidence = probabilities.max()
    
    
    if confidence >= confidence_threshold:
        label = "Real" if prediction == 1 else "Fake"
        return label, confidence, probabilities
    elif real_prob >= real_bias_threshold:
        return "Real (leaning)", real_prob, probabilities
    else:
        return "Real (default)", real_prob, probabilities


def predict_news_simple(text, model, vectorizer, confidence_threshold=0.52):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    if vec.sum() == 0:
        return "Real", 0.5, [0.5, 0.5]
    
    probabilities = model.predict_proba(vec)[0]
    fake_prob, real_prob = probabilities[0], probabilities[1]
    confidence = probabilities.max()
    
    
    if real_prob > fake_prob or confidence < confidence_threshold:
        return "Real", real_prob, probabilities
    else:
        return "Fake", fake_prob, probabilities


test_examples = [
    # Should be REAL
    ("The president signed a new economic bill today", "Real"),
    ("Federal reserve announces interest rate changes", "Real"), 
    ("Scientists discover new method for clean energy", "Real"),
    ("Company reports strong quarterly results", "Real"),
    ("Study shows benefits of daily exercise", "Real"),
    ("New research finds benefits of Mediterranean diet", "Real"),
    ("City council approves new infrastructure project", "Real"),
    
    # Should be FAKE
    ("SHOCKING: This one trick will make you rich", "Fake"),
    ("CLICK HERE TO WIN FREE IPHONE", "Fake"),
    ("BREAKING: Celebrity scandal revealed", "Fake"),
    ("Doctors hate this one weight loss secret", "Fake"),
    ("You won't believe what happened next", "Fake"),
    ("This video will change your life forever", "Fake"),
    ("They don't want you to know this secret", "Fake"),
    
    # Ambiguous/Neutral
    ("Breaking news from the white house", "Ambiguous"),
    ("Important announcement about health", "Ambiguous"),
    ("Latest updates on the situation", "Ambiguous")
]

print("\n" + "="*70)
print("SMART PREDICTION TESTING (Uncertain -> Real)")
print("="*70)

print("\n--- SMART VERSION ---")
correct_smart = 0
total_smart = 0

for text, expected in test_examples:
    prediction, confidence, probabilities = predict_news_smart(text, model, vectorizer)
    
    if expected != "Ambiguous":
        total_smart += 1
        if (expected == "Real" and "Real" in prediction) or (expected == "Fake" and prediction == "Fake"):
            correct_smart += 1
    
    print(f"Text: '{text}'")
    print(f"Expected: {expected:12} Predicted: {prediction:20}")
    print(f"Confidence: {confidence:.3f} Probs: [Fake: {probabilities[0]:.3f}, Real: {probabilities[1]:.3f}]")
    print("-" * 70)

if total_smart > 0:
    print(f"SMART Version Test Accuracy: {correct_smart/total_smart*100:.1f}%")

print("\n--- SIMPLE VERSION (Lower Threshold) ---")
correct_simple = 0
total_simple = 0

for text, expected in test_examples:
    prediction, confidence, probabilities = predict_news_simple(text, model, vectorizer)
    
    if expected != "Ambiguous":
        total_simple += 1
        if (expected == "Real" and prediction == "Real") or (expected == "Fake" and prediction == "Fake"):
            correct_simple += 1
    
    print(f"Text: '{text}'")
    print(f"Expected: {expected:12} Predicted: {prediction:20}")
    print(f"Confidence: {confidence:.3f} Probs: [Fake: {probabilities[0]:.3f}, Real: {probabilities[1]:.3f}]")
    print("-" * 70)

if total_simple > 0:
    print(f"SIMPLE Version Test Accuracy: {correct_simple/total_simple*100:.1f}%")


print("\n" + "="*50)
print("FEATURE ANALYSIS")
print("="*50)

feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]


print("\nTop 10 REAL news indicators:")
real_indices = coef.argsort()[-10:][::-1]
for idx in real_indices:
    print(f"  {feature_names[idx]}: {coef[idx]:.3f}")


print("\nTop 10 FAKE news indicators:")
fake_indices = coef.argsort()[:10]
for idx in fake_indices:
    print(f"  {feature_names[idx]}: {coef[idx]:.3f}")

