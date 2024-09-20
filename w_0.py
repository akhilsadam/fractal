import torch
import matplotlib.pyplot as plt
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = 'w.pth'

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


scaling = lambda x, a, b: (1/a) * torch.exp(-a * x) + b 

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
        
        
        self.q = 30
                
        # 1/sqrt(w) initialization
        torch.nn.init.normal_(self.fc1.weight, std=1.0 / w**0.5)
        torch.nn.init.eye_(self.fc2.weight)
        torch.nn.init.eye_(self.fc3.weight)
        torch.nn.init.normal_(self.fc4.weight, std=1.0 / w**0.5)
        # torch.nn.init.normal_(self.f_out.weight, std=1.0 / w**0.5)
        # torch.nn.init.eye_(self.qc.weight)
        [torch.nn.init.constant_(layer.bias, 0) for layer in [self.fc1, self.fc2, self.fc3, self.fc4]]
        
        self.ps = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([1.0])) for _ in range(self.q)])
        # self.p_a = torch.nn.Parameter(torch.tensor([1.0]))
        # self.p_b = torch.nn.Parameter(torch.tensor([2.0]))
        # self.ps = lambda x: scaling(x, self.p_a, self.p_b)
        self.qs = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([10.0])) for _ in range(self.q)])
        # self.q_a = torch.nn.Parameter(torch.tensor([10.0]))
        # self.q_b = torch.nn.Parameter(torch.tensor([2.0]))
        # self.qs = lambda x: scaling(x, self.q_a, self.q_b)      
         
    def _fwd(self, u, x):
        y = self.fc2(torch.frac(x)-0.5)
        z = torch.sin(self.fc3(u))
        return (z * y)
    
    def forward(self, x):
        x = self.fc1(x)
        u = x
        for i in range(self.q):
            u = u + self._fwd(self.ps[i] *u, self.qs[i] * x) # scale changing is important
            # x = self.qc(x)
        # x = self._fwd(x)   
        u = self.fc4(u)
        return u

    def loss(self, x, y, criterion):
        # l = 1.0
        # for _ in range(self.q):
        #     x = self._fwd(x)
        #     l = l + criterion(x, y)
        # return l
        
        return criterion(self(x), y)

# training if not already saved
model = Weierstrass().to(device)
criterion = torch.nn.MSELoss()
try:
    assert os.path.exists(path)
    model.load_state_dict(torch.load(path, weights_only=True))
except:            
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    epochs = 40000
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(x_train.unsqueeze(1), y_train.unsqueeze(1), criterion)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch}, Train Loss {loss.item()}')
    torch.save(model.state_dict(), path)
        
        
# testing and plotting
model.eval()
y_pred = model(x_test.unsqueeze(1))
# test error
print(f'Test error: {criterion(y_pred, y_test.unsqueeze(1)).item()}')

plt.plot(x_test.cpu().numpy(), y_test.cpu().numpy(), label='Ground truth')
plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), label='Training points')
plt.plot(x_test.cpu().numpy(), y_pred.cpu().detach().numpy(), label='Prediction')
plt.legend()
plt.savefig('weierstrass.png')
plt.close()
