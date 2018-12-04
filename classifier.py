import numpy as np
import pickle
import time
import csv
from sklearn import svm


def calculate_avg_emb(tweet, vocab, embed):
    words = tweet.split()
    embeds = []
    for word in words:
        v = vocab.get(word, None)
        if v != None:
            embeds.append(embed[v])
    avg_emb = sum(np.array(embeds))/len(embeds) if len(embeds) != 0 else []
    return avg_emb


def main():
    
    print("\nLoading embeddings...")
    embed = np.load('embeddings.npy')
    print(len(embed), "embeddings loaded\n")

    print("Loading vocab...")
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    print(len(vocab), "vocab lines loaded\n")
    
    print("Loading dataset...")
    fp = open("twitter-datasets/train_pos.txt", "r")
    train_pos = fp.readlines()
    print(len(train_pos), "positive tweets loaded")
    fp.close()
    fn = open("twitter-datasets/train_neg.txt", "r")
    train_neg = fn.readlines()
    print(len(train_neg), "negative tweets loaded\n")
    fn.close()
    
    half_num_samples = 1000  # Total number of tweets used
    train_pos = train_pos[:half_num_samples]
    train_neg = train_neg[:half_num_samples]
    train = train_pos + train_neg
    
    embs = []  # Will contain embeddings (average) for each tweet
    for tweet in train:
        avg_emb = calculate_avg_emb(tweet, vocab, embed)
        if avg_emb != []:
            embs.append(avg_emb)
    
    
    print("\nTraining...\n")
    
    clf = svm.SVC(gamma=0.001, C=100.)
    
    X = embs
    y = np.ones(half_num_samples, dtype=int)
    y = np.append(y, -y)
    print(y)
    clf.fit(X, y)
    
    
    print("\nTesting...\n")
    
    fp = open("twitter-datasets/test_data.txt", "r")
    test_data = fp.readlines()
    print(len(test_data), "test tweets loaded\n")
    fp.close()
    
    localtime = time.asctime(time.localtime(time.time()))
    
    fp = open("twitter-datasets/submission" + localtime + ".csv", "w")
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    
    for tweet in test_data:
        i, t = tweet.split(",", maxsplit=1)  # Splitting the index from the tweet text
        avg_emb = calculate_avg_emb(t, vocab, embed)
        prediction = 1  # Default value for tweets we can't analyse
        if avg_emb != []:
            prediction = clf.predict([avg_emb])[0]
        writer.writerow({'Id': str(i), 'Prediction': str(prediction)})
    fp.close()
    
if __name__ == '__main__':
    main()
