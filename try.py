import paddle
from paddle import optimizer
from paddle.nn import functional as F
from model.mamba import Mamba, MambaConfig
# Model configuration
config = MambaConfig(d_model=16, n_layers=2)
model = Mamba(config)
# Random seed setting
paddle.seed(42)
# Random input and target
B, L, D = 2, 64, 16
x = paddle.randn([B, L, D], dtype=paddle.float32)
target = paddle.randn([B, L, D], dtype=paddle.float32)  
# Optimizer
learning_rate = 0.01
optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=learning_rate)
# Forward
y = model(x)
assert y.shape == x.shape, "Output shape is not equal to input shape!"
# Loss
loss = F.mse_loss(y, target)  
# Backward
loss.backward()
# Update
optimizer.step()
optimizer.clear_grad()
# Print loss
print("Loss:", loss.item())
