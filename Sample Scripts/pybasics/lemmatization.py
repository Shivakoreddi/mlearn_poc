##its like stemming ,output we get is called lemma, which is a root word
##for example - goes root is go
##fairly root is fair, eaten root is eat
# import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

'''
POS - Noun-n
verb-v
adjective-a
adverb-r
'''
print(lemmatizer.lemmatize('going',pos='v'))
