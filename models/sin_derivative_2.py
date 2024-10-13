import torch
from util import device
# anomaly check
# torch.autograd.set_detect_anomaly(True)
class param:
    # train = False
    train = True
    q = 4
    lr = 0.001
    epochs = 2000

class Poly1D(torch.nn.Module):
    def __init__(self, n=3, vals=[1.0, 1.0, 1.0]):
        super(Poly1D, self).__init__()
        
        self.n = n+1
        for i in range(self.n):
            setattr(self, f'p_{i}', torch.nn.Parameter(torch.tensor([vals[i]])))
        
    def forward(self, x):
        result = 0
        for i in range(self.n):
            result += getattr(self, f'p_{i}') * x**i
        return result

nkern = torch.tensor([1.0, -2.0, 1.0]).to(device=device)
    
# Implicit function model
class Weierstrass(torch.nn.Module):
    def __init__(self, w=4):
        super(Weierstrass, self).__init__()
        
        self.path = 'sin_deriv2.pth'
        self.param = param
        
        self.fc1 = torch.nn.Linear(1, w)
        
        # self.fc2 = torch.nn.Conv1d(w, w, 3)
        self.fc2 = torch.nn.Linear(w, w)
        # self.fc3 = torch.nn.Linear(w, w)
        
        self.fc4 = torch.nn.Linear(w, 1)
        # self.f_out = torch.nn.Linear(w, 1)

        # scale/shift layer
        # self.qc = torch.nn.Linear(w, w)
        
                
        # 1/sqrt(w) initialization
        torch.nn.init.normal_(self.fc1.weight, std=1.0 / w**0.5)
        torch.nn.init.eye_(self.fc2.weight)
        # torch.nn.init.eye_(self.fc3.weight)
        torch.nn.init.normal_(self.fc4.weight, std=1.0 / w**0.5)
        # torch.nn.init.normal_(self.f_out.weight, std=1.0 / w**0.5)
        # torch.nn.init.eye_(self.qc.weight)
        [torch.nn.init.constant_(layer.bias, 0) for layer in [self.fc1, self.fc2, self.fc4]]
        

        # self.ps = Poly1D(2,[-0.175,-0.006,0.173])

        # self.qs = Poly1D(2,[9.3,0.75,0.0])
        self.psi = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([0.05])) for _ in range(self.param.q)])
        self.ps = lambda x: self.psi[x]

        self.qsi = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([10.0])) for _ in range(self.param.q)])
        self.qs = lambda x: self.qsi[x]
        
        # different as these polynomials end up as a ratio
        
        self.w = torch.nn.Parameter(torch.tensor([30.0]))
        
    def _fwd(self, u, x):
        # du = torch.gradient(u, x)#
        # du = torch.diff(u, dim=-1) / torch.diff(x, dim=-1) # forward finite difference
        # 2nd order central difference
        # du = (u[2:] - u[:-2]) / (x[2:] - x[:-2])
        # # pad with 1st order difference
        # du_left = (u[1] - u[0]) / (x[1] - x[0])
        # du_right = (u[-1] - u[-2]) / (x[-1] - x[-2])
        # # print(du.shape,du_left.shape,du_right.shape)
        # du = torch.cat([du_left[None,...], du, du_right[None,...]], dim=0)
        
        # faster version of the above without torch cat, using torch  diff
        du_f = torch.diff(u, dim=0) / torch.diff(x, dim=0) # forward finite difference
        du_b = torch.diff(u, dim=0) / torch.diff(x, dim=0) # backward finite difference
        du_c = (du_f[1:] + du_b[:-1]) / 2.0 # central difference
        du = torch.cat([du_f[0:1], du_c, du_b[-1:]], dim=0)
        
        # print(du)
        
        return self.fc2(du)
    
    def q_fwd(self, u, x, q):
        # p_ = 1.0
        # q_ = 10.0
        for i in range(q):
            k = self.ps(i)/self.qs(i)
            # u = u + self._fwd(self.ps(i) * u, self.qs(i) * x) # scale changing is important
            u = u + k*self._fwd(u, x) # scale changing is important
            u = u / torch.sqrt(1 + k**2)
        return u
    
    def enc(self, x):
        return torch.sin(self.fc1(self.w*x))
    
    def dec(self, x):
        return self.fc4(x)
    
    def forward(self, x, q):
        u = self.enc(x)
        # print(u.shape,x.shape)
        u = self.q_fwd(u, x, q)
        u = self.dec(u)
        return u

    def loss(self, x, y, criterion, q):
        # l = 1.0
        # for _ in range(self.q):
        #     x = self._fwd(x)
        #     l = l + criterion(x, y)
        # return l
        out = self(x, q)
        main = criterion(out, y)
        
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
        # noise = torch.mean(torch.nn.functional.conv1d(out[:,None,:], nkern[None, None, :].to(device=device), padding='same')**2)
        # delta = 0.1
        # k = 5
        # lcz = torch.mean(torch.relu((self(x + delta, q) - out)**2 - (delta*k)**2))
        
        return main# + 0.1*lcz #+ 0.1 * consistency
    #### Loss needs a way to do the following:
    # Penalize a noisy output, but keep fractal-like structure (which is not smooth)
    # So wildly oscillating outputs are good, but only as long as they are close to the ground truth
    # not sure how to do this, but maybe a consistency loss?
    # i.e. make sure each layer is close to the previous layer, but also close to the ground truth
    # and also always improving the prediction (monotonicity)?
