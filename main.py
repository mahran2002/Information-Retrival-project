import math
import operator

import nltk
import nltk.corpus
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# function to print taple
def print_taple(header, body, space_in_the_firs_cel):
    if space_in_the_firs_cel == True:
        if header[0] != " ":
            header.insert(0, " ")
    taple = PrettyTable()
    index = 0
    for col in body:
        taple.add_column(header[index], col)
        index += 1
    print(taple)


def readFromFile(path):
    file = open(path, "r+")
    content = file.read()
    return content


def listofalltext():
    listofstrings = []
    for file in files:
        listofstrings.append(readFromFile(file))
    return listofstrings


files = [

    "docs/1.txt",
    "docs/2.txt",
    "docs/3.txt",
    "docs/4.txt",
    "docs/5.txt",
    "docs/6.txt",
    "docs/7.txt",
    "docs/8.txt",
    "docs/9.txt",
    "docs/10.txt",
]


def main():
    files = [
        "docs/1.txt",
        "docs/2.txt",
        "docs/3.txt",
        "docs/4.txt",
        "docs/5.txt",
        "docs/6.txt",
        "docs/7.txt",
        "docs/8.txt",
        "docs/9.txt",
        "docs/10.txt",
    ]
    filesnames = [
        "docs/1",
        "docs/2",
        "docs/3",
        "docs/4",
        "docs/5",
        "docs/6",
        "docs/7",
        "docs/8",
        "docs/9",
        "docs/10",
    ]
    toknize = []
    all_data = ""
    stop_words = set(stopwords.words('english'))
    header = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
    body = []

    # toknization
    # read all file and stored in all data
    for file in files:
        f = open(file, "r")
        all_data += f.read() + " "
    # make all data lower case
    all_data = all_data.lower()
    # toknize all data

    toknizer = RegexpTokenizer("[\w',]+")
    toknize = toknizer.tokenize(all_data)

    print("Tokens :-")
    print(toknize)
    print("----------------------------------------------------------------------------------------")

    # remove stop words
    for term in toknize:
        for sw in stop_words:
            if term == sw:
                if term == "in" or term == "to" or term == "where":
                    break
                else:
                    toknize.remove(term)

    # sorted toknize and rmove repeat words and remove ' and , in last index in word

    toknize = list(dict.fromkeys(toknize))
    for i, word in enumerate(toknize):
        index = len(word)
        if word[index - 1] == "." or word[index - 1] == ",":
            toknize[i] = word[:-1]

    # positional index
    postional_index = {}

    for word in toknize:
        for file in files:
            # read files and split them
            f = open(file, "r")
            content = f.read()
            content = content.lower()
            content_list = content.split()
            for index, word_in_file in enumerate(content_list):
                # index is the position of word in file
                i = len(word_in_file)
                if word_in_file[i - 1] == "." or word_in_file[i - 1] == ",":
                    word_in_file = word_in_file[:-1]
                if word == word_in_file:
                    if (len(postional_index) == 0):

                        postional_index[word] = {file: [index]}
                    else:
                        if word in postional_index:
                            if file in postional_index[word]:
                                postional_index[word][file].append(index)
                            else:
                                postional_index[word][file] = [index]
                        else:
                            postional_index[word] = {file: [index]}
    print("postional index :-")
    print(postional_index)
    print("----------------------------------------------------------------------------------------")

    list_of_terms = []
    for term in postional_index:
        list_of_terms.append(term)
    body.append(list_of_terms)
    # vector space model for document
    vsm = {}
    for file in files:
        vsm[file] = {}
        for term in postional_index:
            if file in postional_index[term]:
                idf = math.log10(len(files) / len(postional_index[term]))
                tf = len(postional_index[term][file])
                tf_weight = 1 + math.log(tf)
                tf_idf = tf_weight * idf
                vsm[file][term] = [idf, tf, tf_weight, tf_idf]
            else:
                vsm[file][term] = [0, 0, 0, 0]
    for file in vsm:
        list_of_tf = []
        for term in list_of_terms:
            list_of_tf.append(vsm[file][term][1])
        body.append(list_of_tf)
    header = ["term", "doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]

    print("\ntf (term frequency) matrix")
    header = ["term", "doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
    print(pd.DataFrame(body, index=header).T)
    print("----------------------------------------------------------------------------------------")

    # print tf-weight
    body.clear()
    body.append(list_of_terms)
    for file in vsm:
        list_of_tf_weight = []
        for term in list_of_terms:
            list_of_tf_weight.append(vsm[file][term][2])
        body.append(list_of_tf_weight)

    print("\n w tf(1 + log tf) matrix :-")
    print(pd.DataFrame(body, index=header).T)
    print("----------------------------------------------------------------------------------------")

    # print df and idf

    header.clear()
    body.clear()
    body.append(list_of_terms)
    header = ["term", "df", "idf"]
    list_of_df = []
    list_of_idf = []
    for term in postional_index:
        list_of_df.append(len(postional_index[term]))
        list_of_idf.append(math.log10(len(files) / len(postional_index[term])))
    body.append(list_of_df)
    body.append(list_of_idf)

    print("\n df & idf  matrix:- ")
    print(pd.DataFrame(body, index=header).T)
    print("----------------------------------------------------------------------------------------")

    # print tf * idf

    header.clear()
    body.clear()
    header = ["term", "doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
    body.append(list_of_terms)

    for file in vsm:
        list_of_tf_idf = []
        for term in list_of_terms:
            list_of_tf_idf.append(vsm[file][term][3])
        body.append(list_of_tf_idf)

    print("\n tf * idf matrix:-")
    print(pd.DataFrame(body, index=header).T)
    print("----------------------------------------------------------------------------------------")

    header.clear()
    body.clear()

    # Euclidean length and unit query for document
    euclidean_length = []
    for file in vsm:
        tf_idf_sqr_sum = 0
        for term in vsm[file]:
            tf_idf_sqr_sum += vsm[file][term][3] * vsm[file][term][3]
        lenth = math.sqrt(tf_idf_sqr_sum)
        euclidean_length.append(lenth)
        for term in vsm[file]:
            if sum(vsm[file][term]) == 0:
                vsm[file][term].append(0)
            else:
                vsm[file][term].append(vsm[file][term][3] / lenth)

    header = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
    print("\n length matrix:-")
    print(pd.DataFrame(euclidean_length, index=header))
    print("----------------------------------------------------------------------------------------")

    # print normalized tfidf
    body.clear()

    body.append(list_of_terms)

    for file in vsm:
        list_of_normalize = []
        for term in list_of_terms:
            list_of_normalize.append(vsm[file][term][4])
        body.append(list_of_normalize)

    print("\n Normalized tf.idf matrix")
    header = ["term", "doc1", "do2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]

    print(pd.DataFrame(body, index=header).T)
    print("----------------------------------------------------------------------------------------")

    vector_corpus = listofalltext()
    vectorizer = TfidfVectorizer(analyzer='word', use_idf=True, stop_words=nltk.corpus.stopwords.words("english"))
    tfidf = vectorizer.fit_transform(vector_corpus)
    feature_names = sorted(vectorizer.get_feature_names_out())
    tfidf_matrix = pd.DataFrame(tfidf.toarray(), columns=feature_names, index=filesnames).T
    ################# cosine similarity #################
    print("Cosine similarity for all Docs :-")
    cs = cosine_similarity(pd.DataFrame(tfidf.toarray()))

    print(pd.DataFrame(cs, columns=filesnames, index=filesnames))
    print("")
    print("----------------------------------------------------------------------------------------")

    ################# cosine similarity #################

    option = int(input("Enter Option : 1 => Phrase Query or 0=> Exit : "))
    term = ""
    while option != 0:
        if option == 1:

            # phrase query
            solution = {}
            query = input("Please Enter Your Query : ")
            query_list = nltk.word_tokenize(query)

            query_list = [word.lower() for word in query_list]
            for word in query_list:
                if word == "," or word == ".":
                    query_list.remove(word)
            flag = False
            for word in query_list:
                for term in postional_index:
                    if word == term:
                        flag = True
                        break
                    else:
                        flag = False
                if flag:
                    if len(solution) == 0:
                        solution[word] = {}
                        for file in postional_index[word]:
                            solution[word][file] = []
                            for index in postional_index[word][file]:
                                solution[word][file].append(index)
                    else:
                        if word in solution:
                            break
                        else:
                            solution[word] = {}
                            for file in postional_index[word]:
                                solution[word][file] = []
                                for index in postional_index[word][file]:
                                    solution[word][file].append(index)


                else:
                    print("Sorry Not Found")
            for term in solution:
                for file in solution[term]:
                    print(f"{term} is found in doc{file}")

            # vector space model for query
            query_vsm = {}
            for query in query_list:
                query_vsm[word] = []
                idf = math.log10(len(files) / len(postional_index[term]))
                tf = 0
                for word in query_list:
                    if word == query:
                        tf += 1
                tf_weight = 1 + math.log(tf)
                tf_idf = tf_weight * idf
                query_vsm[query] = [idf, tf, tf_weight, tf_idf]

            for word in query_vsm:
                tf_idf_sqr_sum = 0
                for query in query_vsm:
                    tf_idf_sqr_sum += pow(query_vsm[query][3], 2)
                lenth = math.sqrt(tf_idf_sqr_sum)
                query_vsm[word].append(query_vsm[word][3] / lenth)

            print(query_vsm)

            similarity_dec = {}
            for file in files:
                similarity = 0
                for word in query_vsm:
                    if word in vsm[file]:
                        similarity += query_vsm[word][4] * vsm[file][word][4]
                similarity_dec[file] = similarity

            sorted_similarity = sorted(similarity_dec.items(), key=operator.itemgetter(1), reverse=True)
            print(sorted_similarity)
            for index in range(0, len(sorted_similarity)):
                if sorted_similarity[index][1] != 0:
                    print(
                        f"\nthe similarity between the query and doc{sorted_similarity[index][0]} is {sorted_similarity[index][1]}")

        option = int(input("\nEnter Option : 1 => Phrase Query or 0=> Exit : "))


main()

#%%

#%%
