from flask import Flask,render_template,url_for,request

app = Flask(__name__)

# Data
import pandas as pd
with open("text.txt", encoding='utf-8') as f:
    texts = [line.strip() for line in f.readlines()]

with open("label.txt", encoding='utf-8') as f:
    categories = [line.strip() for line in f.readlines()]

data = pd.DataFrame({"sentiment":categories, "text":texts})
data.to_csv('data01.csv',index=False)

# Text data preprocessing
import string
from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_stopwords

thai_stopwords = list(thai_stopwords())


def text_cleaning(text):
    clean_text = word_tokenize(text)
    clean_text = " ".join(w for w in clean_text)

    # ลบเครื่องหมายวรรคตอน (Punctuation)
    clean_text = "".join((w for w in clean_text if w not in list(string.punctuation)))
    # ลบเครื่องหมายวรรคตอนไทย
    clean_text = "".join(w for w in clean_text
                         if w not in ("ๆ", "ฯ", "ฯลฯ", "ฯเปฯ", "๏", "๚", "๚ะ", "๛", "๚ะ๛", "๙", "+"))
    # ลบตัวเลข
    clean_text = "".join(w for w in clean_text if not w.isnumeric())

    # ตัดคำ
    clean_text = " ".join(w for w in clean_text.split()
                          if w not in thai_stopwords)

    return clean_text


data['text_tokens'] = data['text'].apply(text_cleaning)

# Feature - TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(analyzer=lambda x:x.split(' '))
x = tfidf_vect.fit_transform(data['text_tokens'])
tfidf_vect.vocabulary_

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(data['sentiment'])

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=101)

# Model building
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(dual=True, solver='liblinear', max_iter=10000)
classifier.fit (X_train, y_train)

from  sklearn.metrics import accuracy_score
y_preds = classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_preds)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        text = (message)
        text_tokens = text_cleaning(text)
        text_features = tfidf_vect.transform(pd.Series([text_tokens]))
        text_predictions = classifier.predict(text_features)
    return render_template('result.html',prediction = text_predictions)


if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=4000)
    app.run()