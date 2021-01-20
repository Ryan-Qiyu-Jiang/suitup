
import time
counter = 0
t1 = 0
t2 = 0

def frame_count():
    global counter, t1, t2
    print('frame total', time.time()-t2)
    t2 = time.time()

    if counter == 0:
        t1 = time.time()
    elif time.time()-t1 >= 1:
        print(counter, 'frames a sec')
        counter=0
        t1 = time.time()
    counter += 1



stopwatch_start = time.time()
avg_time = {}
observations = {}

def timer(msg=''):
  global stopwatch_start
  res = time.time() - stopwatch_start
  if msg in observations:
    observations[msg] +=1
    n = observations[msg]
    avg_time[msg] = avg_time[msg]*(n-1)/n + res/n
  else:
     observations[msg] = 1
     avg_time[msg] = res
  print(msg)
  stopwatch_start = time.time()
  return res