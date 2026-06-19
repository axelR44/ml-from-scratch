from src.losses.MSE import MSE
import numpy as np
import copy
from tqdm import tqdm
import pickle
from src.optim.sgd import SGD
from src.optim.adam import Adam
from src.optim.scheduler import StepLR, ExponentialLR, CosineAnnealingLR, WarmupCosineLR


def clip_gradients(model, max_norm):
    total_norm = 0

    # calcul norme totale
    for layer in model.layers:
        if hasattr(layer, "dW"):
            total_norm += np.sum(layer.dW ** 2)
            total_norm += np.sum(layer.db ** 2)

    total_norm = np.sqrt(total_norm)

    # scaling
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)

        for layer in model.layers:
            if hasattr(layer, "dW"):
                layer.dW *= scale
                layer.db *= scale

class Model:
    def __init__(self, layers):
        self.layers = layers
        best_weights = None
        self.train_losses = []
        self.val_losses = []

    def forward(self, X, training=True):
        out = X
        for layer in self.layers:
            if "training" in layer.forward.__code__.co_varnames:
                out = layer.forward(out, training=training)
            else:
                out = layer.forward(out)
        return out

    def backward(self, grad, lambda_l2):
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                if hasattr(layer, "W"):  # seulement Dense
                    grad = layer.backward(grad, lambda_l2=lambda_l2)
                else:
                    grad = layer.backward(grad)

    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, "step"):
                layer.step(lr)

        
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params
    

    def fit(self, X, y,  X_val=None, y_val=None, loss_fn =None, optimizer_name = None, scheduler_name = None,
            lr=0.01, epochs=100, batch_size = 32, patience = 100, lambda_l2=1e-4,
            use_best_model = True, save_path = None):
        if loss_fn is None:
            loss_fn = MSE()
            
        if optimizer_name is None or optimizer_name == "SGD":
            optimizer = SGD(self, lr=lr)
        elif optimizer_name == "Adam":
            optimizer = Adam(self, lr)
        else:
            raise ValueError("mauvais choix d'optimizer")
        
        if scheduler_name == "COS":
            print('cos scheduler')
            scheduler = CosineAnnealingLR(optimizer)
        elif scheduler_name == "STEP":
            scheduler = StepLR(optimizer)
        elif scheduler_name == "EXP":
            scheduler = ExponentialLR(optimizer)
        elif scheduler_name == "WCOS":
            scheduler = WarmupCosineLR(optimizer, epochs)
        else:
            scheduler = None
            
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        
        pbar = tqdm(range(epochs), desc="Training")
        step = max(1, epochs // 100)

        for epoch in pbar:
            epoch_loss = 0

            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                
                y_batch = y[i:i+batch_size]
                y_pred = self.forward(X_batch, training=True)

                train_loss = loss_fn.forward(y_batch, y_pred)
                epoch_loss += train_loss

                grad = loss_fn.backward()
                self.backward(grad, lambda_l2 = lambda_l2)
                optimizer.step(clip_norm=1)
            
            epoch_loss /= (len(X) // batch_size)            
            self.train_losses.append(epoch_loss)
                        
            if X_val is not None:
                y_val_pred = self.forward(X_val, training=False)
                val_loss = loss_fn.forward(y_val, y_val_pred)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.layers)
                    patience_counter = 0
                    
                    if save_path is not None:
                        self.save(save_path)

                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    if use_best_model:
                        self.layers = best_weights
                    break

            if scheduler is not None:
                scheduler.step()

            if epoch % step == 0:      
                pbar.set_postfix({
                        "train": f"{epoch_loss:.3e}",
                        "val": f"{val_loss:.3e}" if X_val is not None else "N/A",
                        "lr": f"{optimizer.lr:.2e}"
                    })

        if X_val is not None and best_weights is not None and use_best_model:
            self.layers = best_weights
        if hasattr(optimizer, "biggest_norm"):
            print("plus grande norme de gradient: ", optimizer.biggest_norm)
    
    def predict(self, X):
        return self.forward(X)
    
    
    def evaluate(self, X, y, loss_fn):
        y_pred = self.forward(X)
        return loss_fn.forward(y, y_pred)
    
    
    def save(self, path):
        with open(f"models_saved/{path}.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(f"models_saved/{path}.pkl", "rb") as f:
            return pickle.load(f)




