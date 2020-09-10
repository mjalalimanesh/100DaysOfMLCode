from string import punctuation, digits
import numpy as np
import utils
from pegasos import pegasos

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()
#pragma: coderesponse end


#pragma: coderesponse template
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    with open('stopwords' + '.txt') as f:        
        list_of_stopwords = f.readlines()
    list_of_stopwords = list(map(str.strip, list_of_stopwords))
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in list_of_stopwords:
                dictionary[word] = len(dictionary)
    return dictionary
#pragma: coderesponse end


#pragma: coderesponse template
def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = word_list.count(word)
    return feature_matrix
#pragma: coderesponse end

train_data = utils.load_data('reviews_train.tsv')
train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
dictionary = bag_of_words(train_texts)
train_bow_features = extract_bow_feature_vectors(train_texts, dictionary)
theta, theta0 = pegasos(train_bow_features, train_labels, T=25, L=0.01)
wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(theta, wordlist)
print(" *** Most Positive Word Features ***")
print(sorted_word_features[0:20])
print(" *** Most Negative Word Features ***")
print(sorted_word_features[-20:-1])
