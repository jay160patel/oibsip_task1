import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data = pd.read_csv(r'C:\Users\patel\OneDrive\Desktop\DataAnalytics\Task1\SentimentAnalysis\TwitterData.csv')
data.dropna(inplace=True)

X = data['clean_text']
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train_tfidf, y_train)
y_pred = logreg_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.countplot(x='category', data=data, hue='category', palette='viridis', legend=False)
plt.title('Sentiment Distribution in the Dataset')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.show()
cm = confusion_matrix(y_test, y_pred, labels=logreg_model.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=logreg_model.classes_, yticklabels=logreg_model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.show()

report = classification_report(y_test, y_pred, output_dict=True)
metrics = pd.DataFrame(report).transpose()
metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(8, 5), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()
