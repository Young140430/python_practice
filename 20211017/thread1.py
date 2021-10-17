import threading
def fun1():
    for i in range(20):
        print(threading.current_thread().getName(),i)
def fun2():
    for i in range(65,91):
        print(threading.current_thread().getName(),chr(i))
thread1=threading.Thread(target=fun1,name="线程1")
thread2=threading.Thread(target=fun2,name="线程2")
thread1.start()
thread2.start()