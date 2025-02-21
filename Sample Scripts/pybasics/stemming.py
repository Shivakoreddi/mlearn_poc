##Stemming

##Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the
##roots of words known as lemma.

from nltk.stem import PorterStemmer,RegexpStemmer,SnowballStemmer
words = ['eating','eat','eaten','goes','go','gone','think']

stemming = PorterStemmer()

for word in words:
    print(word+"--->"+stemming.stem(word))

##problem with stemming is ,sometimes stemming doesnt give meaningful stem from keywords
print(stemming.stem('congratulations'))


##RegexpStemmer Class

reg_stemmer = RegexpStemmer('ing$|s$|e$|able$',min=4)

print(reg_stemmer.stem('eating'))



##Snowballstemmer

snowball = SnowballStemmer('english')

for word in words:
    print(word+"---->"+snowball.stem(word))

print(stemming.stem('fairly'))
print(snowball.stem('fairly'))