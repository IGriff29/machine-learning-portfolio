from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def main():
    texts = [
        "appointment confirmed for tomorrow", "your care plan has been updated", "medication reminder is ready",
        "system error while loading dashboard", "model service timeout detected", "database connection failed",
        "new coaching message available", "weekly health check completed", "application log contains exception",
        "server latency crossed threshold", "follow up visit scheduled", "patient education content published",
    ]
    labels = ["care","care","care","technical","technical","technical","care","care","technical","technical","care","care"]
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=.33, random_state=42, stratify=labels)
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2))), ("nb", MultinomialNB())])
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

if __name__ == "__main__": main()
