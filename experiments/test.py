from src.layers.conv2d import Conv2D

import time
import numpy as np

X = np.random.randn(
    32,
    3,
    224,
    224
).astype(np.float32)

conv = Conv2D(
    out_channels=32,
    kernel_size=3,
    padding=1
)

# build
Y = conv.forward(X)

dZ = np.random.randn(*Y.shape).astype(np.float32)



# =====================
# backward
# =====================

conv.forward(X)

t0 = time.perf_counter()

for _ in range(100):
    conv.backward(dZ)

backward_time = time.perf_counter() - t0

print(f"Backward : {backward_time:.3f}s")
