import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from datasets import _news

def news(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, labels = _news(os.path.join(data_dir, 'newsgroup_test.csv'))
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('Newsgroup Valid Accuracy: %.2f'%(valid_accuracy))
    print('Newsgroup Test Accuracy:  %.2f'%(test_accuracy))
