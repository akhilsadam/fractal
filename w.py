import torch
import matplotlib.pyplot as plt
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = 'w.pth'
train = False# True

# Training data
# Weierstrass function
def weierstrass(x, q=25):
    a = 0.5
    b = 3.0
    result = 0
    for k in range(q):
        result += a**k * torch.cos(b**k * x * torch.pi / 5.0) # 2.5 is a scale factor
    return result / 2.0 # bound to [-1, 1]

# Training samples:
# sparse x for training
w = 0.01 # rough sample widths
x_train = torch.arange(-1.0, 1.0, w).to(device)
x_test = torch.arange(-1.0, 1.0, 0.1*w).to(device)

y_train = weierstrass(x_train)
y_test = weierstrass(x_test)


scale_down = lambda x, a, b: (1/a) * torch.exp(-a * x) + b 
scale_up = lambda x, a, b: (1/a) * torch.exp(a * x) + b 
# Implicit function model
class Weierstrass(torch.nn.Module):
    def __init__(self, w=10):
        super(Weierstrass, self).__init__()
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
        y = self.fc2(torch.frac(x)-0.5)
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

# training if not already saved
model = Weierstrass().to(device)
criterion = torch.nn.MSELoss()
try:
    assert os.path.exists(path)
    model.load_state_dict(torch.load(path, weights_only=True))
except:
    print('Model not found or different, training...')
    train = True
    
if train:            
    q_train = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 5000
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(x_train.unsqueeze(1), y_train.unsqueeze(1), criterion, q_train)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch}, Train Loss {loss.item()}')
    torch.save(model.state_dict(), path)
        
        
# testing and plotting
model.eval()
y_pred = model(x_test.unsqueeze(1), 10)
# test error
print(f'Test error: {criterion(y_pred, y_test.unsqueeze(1)).item()}')

plt.plot(x_test.cpu().numpy(), y_test.cpu().numpy(), label='Ground truth')
plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), label='Training points')
plt.plot(x_test.cpu().numpy(), y_pred.cpu().detach().numpy(), label='Prediction')
plt.legend()
plt.savefig('weierstrass.png')
plt.close()
