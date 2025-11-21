from queue import Queue

embedding_queue = Queue(maxsize=20000)
STOP_SIGNAL = object()
