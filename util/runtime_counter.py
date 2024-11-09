import time

def get_start_time():
    return time.time()

def get_end_time():
    return time.time()

def get_time_difference(t1, t2):
    return abs(t1 - t2)

def print_time(t_seconds, status, desc):
    print(f"{desc} {status}: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_seconds))}")