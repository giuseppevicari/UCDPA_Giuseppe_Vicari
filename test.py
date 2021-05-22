import timeit
import sys
import matplotlib.pyplot as plt

dict = {}

def q(n):
    if n==1 or n==2:
        return 1
    else:
        return(q(n-q(n-1))+q(n-q(n-2)))

def qb(n):
    global dict
    if n==1 or n==2:
        return 1
    elif n in dict.keys():
        return dict[n]

    else:
        dict[n]= qb(n - qb(n - 1)) + qb(n - qb(n - 2))
        return(qb(n-qb(n-1))+qb(n-qb(n-2)))

def runq():
    for i in range(1,35):
        print(i, q(i))
    return

#print(timeit.timeit(runq, number=1))

def runqb(n):
    x=[]
    y=[]
    for i in range(1,n+1):
        x.append(i)
        y.append(qb(i))
    return x, y

#print(timeit.timeit(runqb, number=1))
x, y = runqb(1000000)
plt.plot(x, y)
plt.show()
