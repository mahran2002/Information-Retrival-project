from nltk.tokenize import regexp_tokenize, word_tokenize

import nltk


def TokenizationForQuery(Query):
    listLower = []
    word_tokens = regexp_tokenize(Query, "\s|[\.,;'_/]", gaps=True)
    for word in word_tokens:
        word.split("_")
        listLower.append(word.lower())
    return listLower


def TokenizationForFile(filename):
    file = open(filename, "r+")
    stri = file.read()
    listLower = []

    word_tokens = word_tokenize(stri)
    for word in word_tokens:
        listLower.append(word.lower())
    return listLower
    file.close()


def StopWord(tokenizationList):
    stopwords = nltk.corpus.stopwords.words("english")
    var = [w for w in tokenizationList if not w in stopwords]
    filtered_sentence = []
    for w in tokenizationList:
        if w not in stopwords:
            filtered_sentence.append(w)
    return filtered_sentence
