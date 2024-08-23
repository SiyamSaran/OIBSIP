import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# Load the data
data = pd.read_csv("E:\\spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Feature extraction
class TextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Check if input is a list, convert to DataFrame
        if isinstance(X, list):
            X = pd.Series(X)
        return pd.DataFrame(X.apply(self.extract_features).tolist())
    
    def extract_features(self, text):
        features = {
            'text_length': len(text),
            'special_chars': len(re.findall(r'[!$%]', text)),
            'upper_words': sum(1 for word in text.split() if word.isupper()),
            'digit_count': sum(char.isdigit() for char in text),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if len(text.split()) > 0 else 0
        }
        return features

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Pipeline
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', vectorizer),
        ('text_features', TextFeatures())
    ])),
    ('scaler', StandardScaler(with_mean=False)),
    ('model', LogisticRegression(max_iter=1000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Apply pipeline to training thedata
X_train_transformed = pipeline.named_steps['features'].fit_transform(X_train)
X_train_scaled = pipeline.named_steps['scaler'].fit_transform(X_train_transformed)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Fit the model using resampled data
pipeline.named_steps['model'].fit(X_resampled, y_resampled)

# Transform the test data
X_test_transformed = pipeline.named_steps['features'].transform(X_test)
X_test_scaled = pipeline.named_steps['scaler'].transform(X_test_transformed)

# Predictions
y_pred = pipeline.named_steps['model'].predict(X_test_scaled)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Cross-validation
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
cv_scores = cross_val_score(pipeline, data['text'], data['label'], cv=cv, scoring='f1')

print(f'Cross-validated F1 Score: {np.mean(cv_scores):.2f}')

# Spam Prediction Function
def predict_spam(email_text):
    if not email_text.strip():
        return "Invalid input: Empty text"
    email_features = pipeline.named_steps['features'].transform([email_text])
    email_scaled = pipeline.named_steps['scaler'].transform(email_features)
    prediction = pipeline.named_steps['model'].predict(email_scaled)
    return 'Spam' if prediction == 1 else 'Not Spam'

#Get email text input from the user
email_text = input("Enter the email text:")
print("The email is:", predict_spam(email_text))
