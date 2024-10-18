import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, yaml
from utils.util import device
from utils.mod_check import consistent, time_stat


# from models.base_sine import Weierstrass
# from models.sin_derivative_2 import Weierstrass
from models.sin_derivative_4 import Weierstrass
from parameters import Parameters

fit = 'fit'
os.makedirs(fit, exist_ok=True)


# Training data
# Weierstrass function
def weierstrass(x, q=35):
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

# training if not already saved
parameters = Parameters()
model = Weierstrass(parameters).to(device)

model_path = os.path.join(fit,model.path)
parameter_path = model_path.replace('.pth', '_parameter.yml')
parameters.z_tstat = time_stat(Weierstrass)

criterion = torch.nn.MSELoss()

if not parameters.train:
    print('Autoloading model...')
    try:
        assert os.path.exists(model_path), 'Model not found; retraining model from scratch'
        assert os.path.exists(parameter_path), 'Parameters not found; retraining model from scratch'
        
        # check if parameters match
        saved_p = yaml.load(open(parameter_path).read(), Loader=yaml.FullLoader)
        consistent(parameters, saved_p)
        
        # check if architecture matches
        saved = torch.load(model_path, weights_only=True)
        assert model.state_dict().keys() == saved.keys(), 'Architecture mismatch; retraining model from scratch'
        
        # load weights
        print('Loading model...')
        model.load_state_dict(saved)
        
    except Exception as e:
        print(e)
        parameters.train = True
    
if parameters.train:           
    model.train() 
    q_train = parameters.q
    optimizer = torch.optim.Adam(model.parameters(), lr=model.param.lr, fused=True)
    epochs = parameters.epochs
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        q_train_ = q_train #+ (epoch // 10) % 2
        loss = model.loss(x_train.unsqueeze(1), y_train.unsqueeze(1), criterion, q_train_)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch}, Train Loss {loss.item() / (y_train**2).mean().item()}')
            # print([getattr(model.ps,f'p_{i}').item() for i in range(3)])
            # print([getattr(model.qs,f'p_{i}').item() for i in range(3)])
    with open(parameter_path, 'w') as f:
        yaml.dump(parameters.__dict__, f)
    torch.save(model.state_dict(), model_path)
        
        
# testing and plotting
model.eval()
y_pred = model.forward(x_test.unsqueeze(1), parameters.test_q)
# test error
print(f'Test error: {criterion(y_pred, y_test.unsqueeze(1)).item() / (y_test**2).mean().item()}')

plt.plot(x_test.cpu().numpy(), y_test.cpu().numpy(), label='Ground truth')
plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), label='Training points')
plt.plot(x_test.cpu().numpy(), y_pred.cpu().detach().numpy(), label='Prediction')
plt.ylim(0.3,1.0)
plt.legend()
plt.savefig('weierstrass.png')
plt.close()
