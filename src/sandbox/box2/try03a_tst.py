"""
Usage:
    try03a_tst.py (succ|fail)
"""
import docopt
import time


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    time.sleep(1)
    if args['succ']:
        print "random message 1"
        print "successful"
        print "random message 2"

    if args['fail']:
        print "failed"
        exit(-1)
