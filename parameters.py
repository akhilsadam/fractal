class Parameters:
    def __init__(self):
        self.train = False
        self.q = 4 # number of layers
        self.receptive_field = 50
        self.w = 10
        self.lr = 0.01
        self.epochs = 1000
        self.test_q = 4