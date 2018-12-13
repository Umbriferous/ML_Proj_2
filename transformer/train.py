import argparse
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import time
import csv
from sklearn.utils import shuffle

from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from opt import OpenAIAdam
from utils import (encode_dataset, iter_data, ResultLogger, make_path)
from loss import MultipleChoiceLossCompute

def transform_tweet(X1):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 1, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 1, n_ctx), dtype=np.float32)

    for i, x1 in enumerate(X1):
        x12 = [start_token] + x1[:max_len] + [clf_token]
        l12 = len(x12)
        xmb[i, 0, :l12, 0] = x12
        mmb[i, 0, :l12] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb

"""
def iter_apply(Xs, Ms, Ys):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            clf_logits *= n
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost
"""


def iter_predict(Xs, Ms):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits


"""
def log(save_dir, desc):
    global best_score
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))
"""

"""
def predict(dataset, submission_dir):
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))
"""


def run_epoch():
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        #print("clf", clf_logits)
        #print("ymb", ymb)
        l = compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        #print("loss:", l)
        n_updates += 1
        # if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            # log(save_dir, desc)


argmax = lambda x: np.argmax(x, 1)


def generate_encodings(tweets, split_index=False):
    encods = []
    for tweet in tweets:
        if split_index:
            _, tweet = tweet.split(",", maxsplit=1)
        encoding = []
        for word in tweet:
            v = voc.get(word, None)
            if v is not None:
                encoding.append(v)
        if encoding:
            encods.append(encoding)
    return encods


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)  # 3
    parser.add_argument('--n_batch', type=int, default=14)  # 8
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=140)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    submit = args.submit
    dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)

    # Loading Data
    with open('../vocab.pkl', 'rb') as f:
        voc = pickle.load(f)
    fp = open("../twitter-datasets/train_pos.txt", "r")
    tweets_pos = fp.readlines()
    fp.close()
    fn = open("../twitter-datasets/train_neg.txt", "r")
    tweets_neg = fn.readlines()
    fn.close()

    training = 0
    if training:

        # Generating encodings
        num_samples_per_class = 100000
        encodings = generate_encodings(tweets_pos[:num_samples_per_class])
        num_pos = len(encodings)
        encodings = encodings + generate_encodings(tweets_neg[:num_samples_per_class])
        num_neg = len(encodings) - num_pos

        trX1 = encodings
        trY = np.append(np.zeros(num_pos, dtype=int), np.ones(num_neg, dtype=int))
        n_vocab = len(voc)
        start_token = n_vocab
        clf_token = n_vocab + 1
        n_special = 2
        max_len = 140
        n_ctx = max_len + 2

        vocab = n_vocab + n_special + n_ctx
        trX, trM = transform_tweet(trX1)
        n_train = len(trY)

        n_batch_train = args.n_batch * max(n_gpu, 1)
        n_updates_total = (n_train // n_batch_train) * args.n_iter

        print("updates total", n_updates_total)

        dh_model = DoubleHeadModel(args, clf_token, ('classification', 2), vocab, n_ctx)

        criterion = nn.CrossEntropyLoss(reduce=False)
        model_opt = OpenAIAdam(dh_model.parameters(),
                               lr=args.lr,
                               schedule=args.lr_schedule,
                               warmup=args.lr_warmup,
                               t_total=n_updates_total,
                               b1=args.b1,
                               b2=args.b2,
                               e=args.e,
                               l2=args.l2,
                               vector_l2=args.vector_l2,
                               max_grad_norm=args.max_grad_norm)
        compute_loss_fct = MultipleChoiceLossCompute(criterion,
                                                     criterion,
                                                     args.lm_coef,
                                                     model_opt)

        # load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)

        dh_model.to(device)
        dh_model = nn.DataParallel(dh_model)

        n_updates = 0
        n_epochs = 0
        trYt = trY
        best_score = 0

        for i in range(args.n_iter):
            print("running epoch", i)
            run_epoch()
            n_epochs += 1
            # log(save_dir, desc)

        torch.save(dh_model.state_dict(), 'model_state')

    else:
        n_vocab = len(voc)
        max_len = 140
        n_special = 2
        n_ctx = max_len + 2
        vocab = n_vocab + n_special + n_ctx
        n_batch_train = args.n_batch * max(n_gpu, 1)
        start_token = n_vocab
        clf_token = n_vocab + 1
        dh_model = DoubleHeadModel(args, clf_token, ('classification', 2), vocab, n_ctx)
        dh_model.to(device)
        dh_model = nn.DataParallel(dh_model)

        dh_model.load_state_dict(torch.load('model_state'))

    # Estimation

    dh_model.eval()
    testing_samples_per_class = 100
    test_data = tweets_pos[:testing_samples_per_class] + tweets_neg[:testing_samples_per_class]
    correct_predictions = 0
    teX1 = generate_encodings(test_data)
    teX, teM = transform_tweet(teX1)
    logits = iter_predict(teX, teM)
    prediction = argmax(logits)
    print(prediction)
    answers = np.append(np.zeros(testing_samples_per_class, dtype=int), np.ones(testing_samples_per_class, dtype=int))
    num_mistakes = sum(abs(prediction - answers))
    print("Estimated accuracy:", (1 - num_mistakes/(testing_samples_per_class*2)))

    # Final predictions

    fp = open("../twitter-datasets/test_data.txt", "r")
    test_data = fp.readlines()
    print(len(test_data), "test tweets loaded\n")
    fp.close()

    if not os.path.exists("../out"):
        os.makedirs("../out")

    localtime = time.asctime(time.localtime(time.time())).replace(':', '.')
    fp = open("../out/submission " + localtime[4:-5] + ".csv", "w")
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()

    print("Generating predictions...")

    teX1 = generate_encodings(test_data, split_index=True)
    teX, teM = transform_tweet(teX1)
    logits = iter_predict(teX, teM)
    prediction = argmax(logits)*(-2) + 1
    for i, p in enumerate(prediction):
        writer.writerow({'Id': str(i+1), 'Prediction': str(p)})
    fp.close()
    print("Done.")

