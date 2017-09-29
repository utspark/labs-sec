import argparse
import subprocess
import string
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from collections import defaultdict
from os import path

def runCommand(experiment, shell=False):
    print ("Running command: " + experiment + "\n")
    if not shell:
        experiment = experiment.split()
    subprocess.call(experiment, shell=shell)

def gen_traces(args):
    for i in xrange(10):
        for number,letter in enumerate(string.ascii_uppercase, 1):
            event = 'cpu/umask=0x80,event=0xB0' #OFFCORE_REQUESTS.ALL_REQUESTS
            runCommand("mkdir -p output/" + str(i))
            mySqlCommand = ("mysql -e 'use employees; select AVG(salary) from employees e join salaries s on e.emp_no = s.emp_no join titles t on e.emp_no = t.emp_no where e.first_name REGEXP " +
                            '"' +
                            letter +
                            '.*";' + "'")
            command = ('pcm/pcm-core.x 0.015 -e ' +
                       event +
                       ' -csv=output/' +
                       str(i) + '/' + letter +
                       '.trace -- ' +
                       mySqlCommand + '>' +
                       'output/' + str(i) + '/' + letter + 'query.out')
            runCommand(command, True)

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
    return matrix

# Creates pandas dataframe from dictionary
def make_fmat(data):
    mat = [ [k] + iv for k,v in data.iteritems() for iv in v ]
    return pd.DataFrame(mat).fillna(0).rename(index=str, columns={0:'label'})

def classify(args):
    # Get data from input files
    df = make_fmat( get_data(args.infiles) )

    # Instantiate classifier with parameters
    # TODO: You can change the classifier and/or
    #       post-process the time-series data to
    #       improve classification
    clf = RFC(  n_estimators=10,
                max_depth=6,
                random_state=0,
                class_weight='balanced_subsample'
             )

    # Get classifier AUC score with 10 folds
    fmat = df.as_matrix()
    X = fmat[:,1:]
    y = fmat[:,0]
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print '10-fold score: %f +/- %f' % (scores.mean(), scores.std()*2)

    #Plot mean trace for each query
    for label in set(y):
        data = np.mean(X[y==label], axis=0)
        plt.plot(xrange(len(data)), data, label=label, markersize=1)
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
