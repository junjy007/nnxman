from subprocess import PIPE, Popen
import os

if __name__ == '__main__':
    print "Calling Popen ..."
    ps = Popen(['python', 'try03a_tst.py', 'succ'], stdout=PIPE, stderr=PIPE,
               env=os.environ.copy())
    print "Commnicating ..."
    so, se = ps.communicate()
    print "-- stdout"
    print so
    print "-- stderr"
    print se


    if 'success' in so:
        print "Success found"