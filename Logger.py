from time import time


class Logger:
    def __init__(self, title):
        self.start_time = time()
        self.title = title
        
    def log(self, message):
        elapsedTime = round(time() - self.start_time,4)


        print("{} -- {} -- {}".format(self.title, elapsedTime, message))