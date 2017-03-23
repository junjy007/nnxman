from multiprocessing import Process, Queue
import  time

def f(q):
    time.sleep(5)
    q.put("hi")
    time.sleep(5)
    q.put("bye")

msg_queue = Queue()
p = Process(target=f, args=(msg_queue, ))
p.start()
while True:
    print "reading message ..."
    try:
        m = msg_queue.get(timeout=1)
        print "get {}".format(m)
        if m == "bye":
            break
    except:
        pass

p.join()
