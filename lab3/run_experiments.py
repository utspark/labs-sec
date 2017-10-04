import argparse
import subprocess32
import string
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import LabelBinarizer
from collections import defaultdict
from os import path
import xgboost as xgb

def runCommand(experiment, shell=False):
    print ("Running command: " + experiment + "\n")
    if not shell:
        experiment = experiment.split()
    try:
        subprocess32.call(experiment, shell=shell)
    except:
        return

def gen_traces(args):
    keywords = ['CMOS', 'FPGA', 'branch', 'anomaly', 'detection', 'prefetch', 'memory', 'floating point', 'malware', 'iot']
    for i in xrange(10):
        for keyword in keywords:
            event = 'cpu/umask=0x80,event=0xB0' #OFFCORE_REQUESTS.ALL_REQUESTS
            runCommand("mkdir -p output/" + str(i))
            # mySqlCommand = ("mysql -e 'use employees; select AVG(salary) from employees e join salaries s on e.emp_no = s.emp_no join titles t on e.emp_no = t.emp_no where e.first_name REGEXP " +
            #                 '"' +
            #                 letter +
            #                 '.*";' + "'")

            mySqlCommand = ("mysql -e 'use patent; select * from PUBLICATION where Abstract REGEXP(" + '"' + keyword + '")' + "'")
            command = ('pcm/pcm-core.x 0.015 -e ' +
                       event +
                       ' -csv=output/' +
                       str(i) + '/' + keyword +
                       '.trace -- ' +
                       mySqlCommand + '>' +
                       'output/' + str(i) + '/' + keyword + 'query.out')
            runCommand(command, True)
            #runCommand("yes | pcm/pcm-core.x 0.015 -e"  + event, True)

# From list of files, create dictionary of labels (filename) with Event0 counter values
# Assumes:  Output columns of the form 'Core,IPC,Instructions,Cycles,Event0,...'
#           Label is filename
def get_data(filenames):
    matrix = defaultdict(list)
    get_event0 = lambda x : float(x.split(',')[4])
    for filename in filenames:
        base = path.basename(filename)
        # Create timeseries features with 'Event0' counter
        with open(filename) as f:
            matrix[base].append( map(get_event0, re.findall('(\*.*)', f.read())) )
    minLen = min(map(lambda(k,v): min(map(len, v)), matrix.iteritems()))
    for base in matrix:
        for i in xrange(len(matrix[base])):
            matrix[base][i] = matrix[base][i][0:minLen]
    return matrix

# Creates pandas dataframe from dictionary
def make_fmat(data):
    mat = [ [k] + iv for k,v in data.iteritems() for iv in v ]
    return pd.DataFrame(mat).fillna(0).rename(index=str, columns={0:'label'})

# Performs k-fold cross validation and computes F-score for each fold
# Also reports aggregate AUC with 'micro'-averaging
# Takes optional argument of classifier to swap out classifier
# If classifier uses different decision function, specifiy with 'decision'
def kfold_classify(params, X, y, n_folds, classifier=RFC, decision='predict'):
    print 'Fitting %d folds' % n_folds

    # shuffle and split training and test sets for each fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = classifier( **params )
        clf.fit(X_train, y_train)
        y_pred = getattr(clf, decision)(X_test)

        print 'f1 score: %f' % f1_score(y_test, y_pred, average='micro')

        # Compute roc for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print 'Multiclass, %d-fold AUC: %f' % (n_folds, roc_auc['micro'])

def classify(args):
    # Get data from input files
    df = make_fmat( get_data(args.infiles) )
    
    # Get classification data and sanitize
    # for multiclass problem
    # TODO: You can post-process the features in "X"
    #       or add features (more counters, average
    #       value, sum of values, etc.) 
    fmat = df.as_matrix()
    X = fmat[:,1:]
    y = fmat[:,0]
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    n_classes = y_bin.shape[1]

    # We instantiate random forest with these parameters
    # TODO: You can change the classifier and/or
    #       post-process the time-series data to
    #       improve classification
    # Note: If you change the classifier, parameters should
    #       be changed accordingly
    params = {
            'n_estimators':200,
            'max_depth':10,
            'random_state':0,
            'class_weight':'balanced_subsample',
    }

    # Perform 5-fold cross validation and report accuracy of each fold
    kfold_classify(params, X, y_bin, n_folds=5)

    #Plot mean trace for each query
    for label in set(y):
        data = np.mean(X[y==label], axis=0)
        #plt.plot(xrange(len(data)), data, label=label, markersize=1)
        plt.plot(xrange(100), data[0:100], label=label, markersize=1)
    plt.ylabel("Memory Requests", fontsize=8)
    plt.xlabel("Interval", fontsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.tick_params(axis='x', labelsize=8)
    plt.legend(prop={'size': 5}, bbox_to_anchor=(1,1))
    plt.savefig('output/traces.pdf', format='pdf')

def main():
    parser = argparse.ArgumentParser(description=
                                     'Run utilization privacy experiments. \
                                     You must run this script from \
                                     the root gem5 directory.',
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--app', action='store',
                        help='Name of the app to run (medical, pagerank)')

    parser.add_argument('--func', action='store', default='gen_traces',
                        help='Function to run \
                        (gen_traces, classify, ...)')

    parser.add_argument('--infiles', type=str, nargs='+', required=False,
                        help='Input files containing training counter values.\
                                File name corresponds to the data\'s label.')

    args = parser.parse_args()

    if args.func in globals() and callable(globals()[args.func]):
        globals()[args.func](args)
    else:
        raise KeyError('No such function: ' + args.func)

if (__name__ == "__main__"):
    main()
