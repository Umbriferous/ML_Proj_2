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
    
    # Params
    
    training_samples_per_class = 10000
    testing_samples_per_class = 5000
    
    # Loading data
    
    print("\nLoading embeddings...")
    embed = np.load('embeddings.npy')
    print(len(embed), "embeddings loaded\n")

    print("Loading vocab...")
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    print(len(vocab), "vocab lines loaded\n")
    
    print("Loading dataset...")
    fp = open("twitter-datasets/train_pos_full.txt", "r")
    tweets_pos = fp.readlines()
    print(len(tweets_pos), "positive tweets loaded")
    fp.close()
    fn = open("twitter-datasets/train_neg_full.txt", "r")
    tweets_neg = fn.readlines()
    print(len(tweets_neg), "negative tweets loaded\n")
    fn.close()
    
    # Generating embeddings for tweets
    
    embs = []
    for tweet in tweets_pos[:training_samples_per_class]:
        avg_emb = calculate_avg_emb(tweet, vocab, embed)
        if avg_emb != []:
            embs.append(avg_emb)
            
    num_pos = len(embs)
    
    for tweet in tweets_neg[:training_samples_per_class]:
        avg_emb = calculate_avg_emb(tweet, vocab, embed)
        if avg_emb != []:
            embs.append(avg_emb)
    
    num_neg = len(embs) - num_pos
    
    # Training the model
    
    print("Training on", len(embs), "samples...\n")

    clf = svm.SVC(gamma=0.001, C=100., verbose=1)
    
    X = embs
    y = np.append(np.ones(num_pos, dtype=int),-np.ones(num_neg, dtype=int))
    clf.fit(X, y)
    
    # Testing (on unused samples)
    
    test_data = tweets_pos[:testing_samples_per_class] + tweets_neg[:testing_samples_per_class]
    correct_predictions = 0
    for i, tweet in enumerate(test_data):
        avg_emb = calculate_avg_emb(tweet, vocab, embed)
        prediction = 1  # Default value for tweets we can't analyse
        if avg_emb != []:
            prediction = clf.predict([avg_emb])[0]
        correct_answer = int(i < testing_samples_per_class) * 2 - 1
        if prediction == correct_answer:
            correct_predictions += 1
    print("\n\nPredicted accuracy: " + str(correct_predictions/(2 * testing_samples_per_class)))
    
    
    # Generating submission for test_data
    
    print("\nLoading test_data.txt...\n")
    
    fp = open("twitter-datasets/test_data.txt", "r")
    test_data = fp.readlines()
    print(len(test_data), "test tweets loaded\n")
    fp.close()
    
    localtime = time.asctime(time.localtime(time.time()))
    fp = open("twitter-datasets/submission " + localtime[4:-5] + ".csv", "w")
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    
    print("Generating predictions...\n")
    
    for tweet in test_data:
        i, t = tweet.split(",", maxsplit=1)  # Splitting the index from the tweet text
        avg_emb = calculate_avg_emb(t, vocab, embed)
        prediction = 1  # Default value for tweets we can't analyse
        if avg_emb != []:
            prediction = clf.predict([avg_emb])[0]
        writer.writerow({'Id': str(i), 'Prediction': str(prediction)})
    fp.close()
    
    print("Done.")
    
if __name__ == '__main__':
    main()
