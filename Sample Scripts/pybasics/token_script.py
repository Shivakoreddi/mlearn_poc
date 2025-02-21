import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, PunktTokenizer,word_tokenize,TreebankWordTokenizer

corpus = """Hello, Welcome to AI Learnings.
 Good luck! This is test of NL Processing using NLTK."""

#
documents = sent_tokenize(corpus)
#
# type(documents)

# sent_detector = PunktTokenizer()
#
# documents = sent_detector.tokenize(corpus.strip())

for sentence in documents:
    print(sentence)

words = word_tokenize(corpus)
# for word in words:
#     print(word)


tokenizer = TreebankWordTokenizer()

print(tokenizer.tokenize(corpus))



