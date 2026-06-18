class Model:
    def __init__(self, layers, verbose = 0):
        self.verbose = verbose
        self.layers = layers
        self.losses = []

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                grad = layer.backward(grad)

    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, "step"):
                layer.step(lr)

    def fit(self, X, y, loss_fn, lr=0.01, epochs=100):
        for epoch in range(epochs):

            y_pred = self.forward(X)
            loss = loss_fn.forward(y, y_pred)
            self.losses.append(loss)

            grad = loss_fn.backward()
            self.backward(grad)

            self.step(lr)

            if (self.verbose !=0 and epoch % self.verbose == 0) or epoch == epochs - 1:
                print(f"[{epoch:03d}/{epochs}] loss={loss:.4f}")

    
    def predict(self, X):
        return self.forward(X)

