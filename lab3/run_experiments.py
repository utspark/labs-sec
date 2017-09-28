import argparse
import subprocess
import string

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

    args = parser.parse_args()

    if args.func in globals() and callable(globals()[args.func]):
        globals()[args.func](args)
    else:
        raise KeyError('No such function: ' + args.func)

if (__name__ == "__main__"):
    main()
