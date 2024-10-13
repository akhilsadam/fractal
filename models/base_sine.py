import torch


class param:
    train = True
    q = 20
    lr = 1e-4
    epochs = 100000
    
scale_down = lambda x, a, b: (1/a) * torch.exp(-a * x) + b 
scale_up = lambda x, a, b: (1/a) * torch.exp(a * x) + b 
# Implicit function model
class Weierstrass(torch.nn.Module):
    def __init__(self, w=10):
        super(Weierstrass, self).__init__()
        
        self.path = 'base_sine.pth'

        self.param = param
        
        self.fc1 = torch.nn.Linear(1, w)
        
        self.fc2 = torch.nn.Linear(w, w)
        self.fc3 = torch.nn.Linear(w, w)
        
        self.fc4 = torch.nn.Linear(w, 1)
        # self.f_out = torch.nn.Linear(w, 1)

        # scale/shift layer
        # self.qc = torch.nn.Linear(w, w)
        
                
        # 1/sqrt(w) initialization
        torch.nn.init.normal_(self.fc1.weight, std=1.0 / w**0.5)
        torch.nn.init.eye_(self.fc2.weight)
        torch.nn.init.eye_(self.fc3.weight)
        torch.nn.init.normal_(self.fc4.weight, std=1.0 / w**0.5)
        # torch.nn.init.normal_(self.f_out.weight, std=1.0 / w**0.5)
        # torch.nn.init.eye_(self.qc.weight)
        [torch.nn.init.constant_(layer.bias, 0) for layer in [self.fc1, self.fc2, self.fc3, self.fc4]]
        
        # self.ps = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([1.0])) for _ in range(self.q)])
        self.p_a = torch.nn.Parameter(torch.tensor([1.0]))
        self.p_b = torch.nn.Parameter(torch.tensor([2.0]))
        self.ps = lambda x: scale_down(x, self.p_a, self.p_b)
        # self.qs = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([10.0])) for _ in range(self.q)])
        self.q_a = torch.nn.Parameter(torch.tensor([1.0]))
        self.q_b = torch.nn.Parameter(torch.tensor([2.0]))
        self.qs = lambda x: scale_up(x, self.q_a, self.q_b)      
         
    def _fwd(self, u, x):
        y = self.fc2(torch.frac(x+0.5)-0.5)
        z = torch.sin(self.fc3(u))
        return (z * y)
    
    def q_fwd(self, u, x, q, st=0):
        for i in range(q):
            j = i + st
            u = u + self._fwd(self.ps(j) * u, self.qs(j) * x) # scale changing is important
        return u
    
    def enc(self, x):
        return self.fc1(x)
    
    def dec(self, x):
        return self.fc4(x)
    
    def forward(self, x, q):
        x = self.enc(x)
        u = self.q_fwd(x, x, q)
        u = self.dec(u)
        return u

    def loss(self, x, y, criterion, q):
        # l = 1.0
        # for _ in range(self.q):
        #     x = self._fwd(x)
        #     l = l + criterion(x, y)
        # return l
        main = criterion(self(x, q), y)
        
        # x = self.enc(x)
        # u = x
        # yhat_prev = x
        # consistency = 0
        # for j in range(q):
        #     u = self._fwd(self.ps(j) * u, self.qs(j) * x)
        #     yhat = self.dec(u)
        #     consistency += torch.mean(torch.relu((yhat-y)**2 - (yhat_prev-y)**2))**2
        #     yhat_prev = yhat
        #     # need to force monotonicity...need better constraint, and then maybe a adjoint / const.opt?
        
        # main = criterion(yhat, y)
        
        return main #+ 0.1 * consistency
    #### Loss needs a way to do the following:
    # Penalize a noisy output, but keep fractal-like structure (which is not smooth)
    # So wildly oscillating outputs are good, but only as long as they are close to the ground truth
    # not sure how to do this, but maybe a consistency loss?
    # i.e. make sure each layer is close to the previous layer, but also close to the ground truth
    # and also always improving the prediction (monotonicity)?
