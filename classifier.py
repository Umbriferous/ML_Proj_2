import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier

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
    
    num_samples = 3  # Half of the total number of tweets considered
    train_pos = train_pos[:num_samples]
    train_neg = train_neg[:num_samples]
    train = train_pos + train_neg
    
    embs = []
    for i, tweet in enumerate(train):
        print(tweet)
        words = tweet.split()
        
        embeds = []
        for word in words:
            v = vocab.get(word, 0)
            if v != 0:
                embeds.append(embed[v])
        print(len(words), "words,", len(embeds), "embeds\n")
        avg_emb = sum(np.array(embeds))/len(embeds)
        print(avg_emb, "\n\n")
        
        embs.append(avg_emb)
    
    
    print("\nTraining...\n")
    
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    
    X = embs
    print(len(X))
    y = np.ones(num_samples)
    y = np.append(y, -y)
    print(len(y))
    clf.fit(X, y)
    
    print(clf.predict([embs[-1]]))  # Testing predictions
    
            
if __name__ == '__main__':
    main()
