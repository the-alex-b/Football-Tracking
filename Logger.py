from time import time


class Logger:
    def __init__(self, title):
        self.start_time = time()
        self.title = title
        
    def log(self, message):
        elapsedTime = round(time() - self.start_time,4)


        print("{} -- {} -- {}".format(self.title, elapsedTime, message))


    def print_average(self, i):
        runtime_per_frame = round((time() - self.start_time)/i,3)
        try:        
            fps = round((1/runtime_per_frame),3)
        except:
            fps = 'Unlimited'
        print("Average runtime per frame = {}. This comes down to {} fps".format(runtime_per_frame, fps))