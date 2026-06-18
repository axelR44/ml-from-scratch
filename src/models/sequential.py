from src.losses.MSE import MSE
import numpy as np
import copy
from tqdm import tqdm

import pickle



class Model:
    def __init__(self, layers):
        self.layers = layers
        best_weights = None
        self.train_losses = []
        self.val_losses = []

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

    def fit(self, X, y,  X_val=None, y_val=None, loss_fn =None, 
            lr=0.01, epochs=100, batch_size = 32, patience = 100, 
            use_best_model = True, save_path = None):
        if loss_fn == None:
            loss_fn = MSE()
        
        best_val_loss = float("inf")
        epoch_loss = 0
        patience_counter = 0
        
        
        pbar = tqdm(range(epochs), desc="Training")
        step = max(1, epochs // 100)

        for epoch in pbar:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                
                y_batch = y[i:i+batch_size]
                y_pred = self.forward(X_batch)

                train_loss = loss_fn.forward(y_batch, y_pred)
                epoch_loss += train_loss

                grad = loss_fn.backward()
                self.backward(grad)
                self.step(lr)
            
            epoch_loss /= (len(X) // batch_size)            
            self.train_losses.append(epoch_loss)
                        
            if X_val is not None:
                y_val_pred = self.forward(X_val)
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

            if epoch % step == 0:      
                pbar.set_postfix({
                        "train": f"{epoch_loss:.3e}",
                        "val": f"{val_loss:.3e}" if X_val is not None else "N/A"
                    })

        if X_val is not None and best_weights is not None and use_best_model:
            self.layers = best_weights

            
    def save(self, path):
        with open(f"models_saved/{path}.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(f"models_saved/{path}.pkl", "rb") as f:
            return pickle.load(f)




    
    def predict(self, X):
        return self.forward(X)

