
import numpy as np

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)



# Expects tuples of (state, next_state, action, reward, done)
class ReplayBufferNew(object):
    """
        dims: dims of feeded sample, which is (st, sst, at, rt, done)

        Please also check the output of sample as your policy requirement
    """
    def __init__(self, max_size=1e6, dims):
        self.max_size = max_size
        self.storage = np.zeros((max_size, dims))
        self.ptr = 0

    def add(self, data):
        index = self.ptr % self.max_size  # replace the old memory with new memory
        self.data[index, :] = np.reshape(np.array(data),[1,-1])
        self.ptr += 1

    def sample(self, batch_size):
        assert self.ptr >= self.max_size, 'Memory has not been fulfilled'
        indices = np.random.choice(self.max_size, size=batch_size)
        return self.storage[indices, :]
        # return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)