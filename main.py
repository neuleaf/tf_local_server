import datetime
import pyinotify
import logging
import os
import time
import threading
from queue import Queue
import segment

logging.basicConfig(level=logging.INFO, filename='./monitor.log')
logging.info("Starting monitor...")


class MyEventHandler(pyinotify.ProcessEvent):
    def __init__(self, queue):
        super(MyEventHandler, self).__init__()
        self.q = queue

    def process_IN_ACCESS(self, event):
        logging.info("ACCESS event : %s  %s" % (os.path.join(event.path, event.name), datetime.datetime.now()))

    def process_IN_ATTRIB(self, event):
        logging.info("IN_ATTRIB event : %s  %s" % (os.path.join(event.path, event.name), datetime.datetime.now()))

    def process_IN_CLOSE_NOWRITE(self, event):
        logging.info("CLOSE_NOWRITE event : %s  %s" % (os.path.join(event.path, event.name), datetime.datetime.now()))

    def process_IN_CLOSE_WRITE(self, event):
        self.q.put(event.name)
        logging.info("CLOSE_WRITE event : %s  %s" % (os.path.join(event.path, event.name), datetime.datetime.now()))

    def process_IN_CREATE(self, event):
        logging.info("CREATE event : %s  %s" % (os.path.join(event.path, event.name), datetime.datetime.now()))

    def process_IN_DELETE(self, event):
        logging.info("DELETE event : %s  %s" % (os.path.join(event.path, event.name), datetime.datetime.now()))

    def process_IN_MODIFY(self, event):

        logging.info("MODIFY event : %s  %s" % (os.path.join(event.path, event.name), datetime.datetime.now()))

    def process_IN_OPEN(self, event):
        logging.info("OPEN event : %s  %s" % (os.path.join(event.path, event.name), datetime.datetime.now()))


def enqueues(watch_path, queue):
    # watch manager
    wm = pyinotify.WatchManager()
    wm.add_watch(watch_path, pyinotify.ALL_EVENTS, rec=True)
    # event handler
    eh = MyEventHandler(queue)

    # notifier
    notifier = pyinotify.Notifier(wm, eh)
    notifier.loop()


def dequeues(watch_path, queue):
    while True:
        while not queue.empty():
            img_path = os.path.join(watch_path, queue.get())
            logging.info("Predict %s" % img_path)
            segment.predict(img_path)
        time.sleep(1)


def main(watch_path):
    queue = Queue(maxsize=2000)

    threads = []
    t1 = threading.Thread(target=enqueues, args=(watch_path, queue))
    threads.append(t1)
    t2 = threading.Thread(target=dequeues, args=(watch_path, queue))
    threads.append(t2)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logging.info("Process exit!")


if __name__ == '__main__':
    main('./images')
