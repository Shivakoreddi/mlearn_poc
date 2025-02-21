
##stopwords

from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords


import nltk
# nltk.download('stopwords')


paragraph = """In a small village nestled between rolling hills, lived a young girl named Lily. She had a curious mind and a heart full of dreams. Every evening, she would sit by the old oak tree, gazing at the stars, wondering what lay beyond the horizon.
One day, while exploring the forest, Lily stumbled upon a hidden path. Intrigued, she followed it and discovered a sparkling stream. As she knelt to drink from the crystal-clear water, a tiny, glowing creature appeared. It was a fairy named Elara.
Elara told Lily about a magical realm where dreams came true. Excited, Lily asked if she could visit. The fairy agreed, but only if Lily promised to return before sunset. With a sprinkle of fairy dust, they were transported to a land of wonder.
Lily marveled at the vibrant flowers, talking animals, and floating islands. She danced with the fairies, laughed with the elves, and felt a joy she had never known. But as the sun began to set, Elara reminded her of the promise.
Reluctantly, Lily returned home, her heart brimming with memories. From that day on, she knew that magic existed, not just in fairy tales, but in the world around her."""


stopwords.words('english')

stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)

##apply stopwords and filter and then apply stemming

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words) ##Converting all the words into sentences
print(sentences)


lemmatizer = WordNetLemmatizer()


for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words) ##Converting all the words into sentences
print(sentences)