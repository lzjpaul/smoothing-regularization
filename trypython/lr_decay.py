if __name__ == '__main__':
    lr = 0.01
    for i in range(100000):
        lr = lr - lr * (1e-4)
        print "iter, lr: ", i, lr
