import torch
from utils.util import device
from einops import rearrange
nkern = torch.tensor([1.0, -2.0, 1.0]).to(device=device)

# Implicit function model
class Weierstrass(torch.nn.Module):
    def __init__(self, param):
        super(Weierstrass, self).__init__()
        
        self.path = 'sin_deriv_4.pth'
        self.param = param
        
        w = self.param.w
        self.w = w
        
        self.fc1 = torch.nn.Linear(1, w)
        self.fc2 = torch.nn.Linear(w, w, bias=False)
        self.fc3 = torch.nn.Linear(w, w, bias=False)
        self.fc4 = torch.nn.Linear(w, 1)

        # 1/sqrt(w) initialization
        torch.nn.init.normal_(self.fc1.weight, std=1.0 / w**0.5)
        torch.nn.init.eye_(self.fc2.weight)
        torch.nn.init.eye_(self.fc3.weight)
        torch.nn.init.normal_(self.fc4.weight, std=1.0 / w**0.5)
        [torch.nn.init.constant_(layer.bias, 0) for layer in [self.fc1, self.fc4]]
        

        self.dt_ = torch.nn.Parameter(torch.tensor([0.01]))

    def gradient(self, u, x):
        
        du_f = torch.diff(u, dim=0) / torch.diff(x, dim=0) # forward finite difference
        du_b = torch.diff(u, dim=0) / torch.diff(x, dim=0) # backward finite difference
        du_c = (du_f[1:] + du_b[:-1]) / 2.0 # central difference
        du = torch.cat([du_f[0:1], du_c, du_b[-1:]], dim=0)
        return self.fc2(du)
    
    def q_fwd(self, u, x, q):
        v = u
        for i in range(q):
            dt_ = self.dt_ / q
            adv = self.fc3(v) * self.gradient(u,  x)
            
            u = u + dt_ * adv # transport (forward)
            # u = u / (1 + dt_**2)
            
            v = v - dt_ * adv # transport (backward)
            # v = v / (1 + dt_**2)
            
        return u
    
    def enc(self, x):
        return torch.sin(self.fc1(self.param.receptive_field*x))
    
    def dec(self, x):
        return self.fc4(x)
    
    def forward(self, x, q):
        u = self.enc(x)
        # print(u.shape,x.shape)
        u = self.q_fwd(u, x, q)
        u = self.dec(u)
        return u

    def loss(self, x, y, criterion, q):
        out = self(x, q)
        main = criterion(out, y)
        
        return main 
