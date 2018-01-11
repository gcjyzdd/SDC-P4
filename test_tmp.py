import numpy as np
import _thread

def testList():
    stack = list()
    stack.append([1,2])
    stack.append([2,3])
    stack.append([3,4])

    print(stack.pop())  #3
    print(stack.pop())  #2
    print(stack.pop())  #1


    stack = list()
    stack.append(1)
    stack.append(2)
    stack.append(3)

    print(stack.pop(0))  #3
    print(stack.pop(0))  #2
    print(stack.pop(0))  #1

def test2():
    a = []
    print(len(a))

    a.append(np.array([1.2,0.3,0.1]))
    print(len(a))

    a.append(np.array([0.8, 1.7, 1.9]))

    print(a)
    print(np.array(a))
    print((a[0]+a[1])/2)
    print(np.mean(np.array(a), axis=0))

def test3():
    print(np.max(np.array([1.2,2.3])))

# Define a function for the thread
def print_time( threadName, delay):
    print(threadName, ': ',delay+1)
    return delay+1

def test4():
    try:
        _thread.start_new_thread(print_time, ("Thread-1", 2,))
        _thread.start_new_thread(print_time, ("Thread-2", 4,))

    except:
        print("Error: unable to start thread")

    print_time("Thread-3", 5)

test4()

def foo(bar, baz):
  print( 'hello {0}'.format(bar))
  return 'foo' + baz

def test5():
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=1)

    async_result = pool.apply_async(foo, ('world', 'foo')) # tuple of args for foo

    # do some other stuff in the main process
    foo('world2', 'hi')
    return_val = async_result.get()
    print(return_val)

for i in range(5):
    test5()