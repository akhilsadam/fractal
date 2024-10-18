class Parameters:
    def __init__(self):
        self.train = False
        self.q = 4 # number of layers
        self.receptive_field = 50
        self.w = 10
        self.lr = 0.025
        self.epochs = 2000
        self.test_q = 4