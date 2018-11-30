import numpy as np
import pickle


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
    
    num_samples = 3
    train_pos = train_pos[:num_samples]
    train_neg = train_neg[:num_samples]
    train = train_pos + train_neg
    
    embs_pos, embs_neg = [], []
    for i, tweet in enumerate(train):
        print("~TWEET~:", tweet)
        words = tweet.split()
        
        embeds = []
        for word in words:
            v = vocab.get(word, 0)
            if v != 0:
                embeds.append(embed[v])
        print(len(words), "words,", len(embeds), "embeds\n")
        avg_emb = sum(np.array(embeds))/len(embeds)
        print(avg_emb, "\n\n")
        
        if(i < num_samples):
            embs_pos.append(avg_emb)
        else:
            embs_neg.append(avg_emb)

    print("Positive embeddings:", embs_pos)
    print("Negative embeddings:", embs_neg)
            
if __name__ == '__main__':
    main()
