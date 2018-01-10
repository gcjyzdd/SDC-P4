import numpy as np

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

test2()
