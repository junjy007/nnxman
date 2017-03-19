import threading


a = 10
print "a {}".format(a)
def func():
    a = 20
    print "thread: hi"

threading.Thread(target=func)
print "a {}".format(a)
