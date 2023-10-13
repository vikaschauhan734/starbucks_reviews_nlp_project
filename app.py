from flask import Flask, render_template, request
import numpy as np
import requests
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api

app= Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
lemma = WordNetLemmatizer()
wv = api.load("word2vec-google-news-300")

def vectorize(sentence):
  words = sentence.split()
  words_vecs = []
  for word in words:
    try:
      words_vecs.append(wv[word])
    except KeyError:
      words_vecs.append(np.zeros(300))
  return np.array(words_vecs).mean(axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        review = str(request.form['review'])
        # Text Preprocessing
        words = re.findall("[a-zA-Z]*", review)
        review = " ".join(words)
        # Lowering the review
        review = review.lower()
        # Spliting all the words
        review = review.split()
        # Removing Stop Words
        review = [word for word in review if word not in set(stopwords.words('english'))]
        # Lemmatization
        review = [lemma.lemmatize(word) for word in review]
        # Joining all remaining words
        review = " ".join(review)
        
        vectorized = vectorize(review).reshape(1, -1)

        prediction = model.predict(vectorized)[0]
    return render_template('prediction.html',result=prediction)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)
