from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import multiprocessing
import time
import nltk
import platform
import spacy


def worker_nltk(a):

    start, length = a
    nltk.word_tokenize(" ".join(data[start:length]))

def worker_spacy(a):

    nlp = spacy.load("en_core_web_sm")
    start, length = a
    dat = " ".join(data[start:length])
    nlp.make_doc(dat)

def worker_keras(a):

    start, length = a
    dat = " ".join(data[start:length])
    x = text_to_word_sequence(dat)

def experiment_multi(data, seq, worker):
    
    tmp = time.time()
    pool.map(worker, seq, chunksize=1)
    print("Multiprocessing time:", time.time()-tmp)

def experiment_single_nltk(data):

    tmp = time.time()
    data = " ".join(data)
    results = nltk.word_tokenize(data)
    print("Regular time:", time.time()-tmp)

def experiment_single_spacy(data):

    tmp = time.time()
    nlp = spacy.load("en_core_web_sm")
    data = " ".join(data)
    docs = nlp.make_doc(data)
    print("Regular time:", time.time()-tmp)

def experiment_single_keras(data):
    
    tmp = time.time()
    data = " ".join(data)
    tokens = text_to_word_sequence(data)
    print("Regular time:", time.time()-tmp)



if __name__ == '__main__':
    
    # Change file to your own txt file that you want to run an experiment on
    with open("/../100m.txt", "r") as infile:
        data = infile.readlines()

        multiprocessing.freeze_support()
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        
        chunk = (len(data)+cores-1)//cores
        seq = [(i, i+chunk) for i in range(0, len(data), chunk)]

        # Runs all experiments sequentially, comment out other experiments for running single experiments
        # NLTK
        experiment_multi(data, seq, worker_nltk)
        experiment_single_nltk(data)

        # spaCy
        experiment_multi(data, seq, worker_spacy)
        experiment_single_spacy(data)

        # Keras
        experiment_multi(data, seq, worker_keras)
        experiment_single_keras(data)

    



