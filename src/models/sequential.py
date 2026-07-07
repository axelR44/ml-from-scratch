from src.losses.MSE import MSE
import numpy as np
import copy
from tqdm import tqdm
import pickle
from src.optim.sgd import SGD
from src.optim.adam import Adam
from src.optim.scheduler import StepLR, ExponentialLR, CosineAnnealingLR, WarmupCosineLR
import h5py


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
        self.loss_fn = None
        self.train_losses = []
        self.val_losses = []
        self.training = True
        self.history = None

        self.metrics = []

    def forward(self, X):
        out = X
        for layer in self.layers:
            if "training" in layer.forward.__code__.co_varnames:
                out = layer.forward(out, training=self.training)
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
    

    def fit(self, X, y,  X_val=None, y_val=None, optimizer_name = None, scheduler_name = None,
            lr=0.01, epochs=100, batch_size = 32, patience = 100, lambda_l2=1e-4,
            use_best_model = True, save_path = None, callbacks=None):
        self.stop_training = False
        self.save_path = save_path
        self.epochs = epochs
        self.use_best_model = use_best_model
        if self.loss_fn is None:
            self.loss_fn = MSE()
            
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

        callbacks = callbacks or []
                
        for cb in callbacks:
            cb.on_train_begin(self)


        self.train()
            
        
        best_val_loss = float("inf")
        patience_counter = 0
        

        for epoch in range(epochs):
            logs = {}
            epoch_loss = 0

            for cb in callbacks:
                cb.on_epoch_begin(self, epoch)

            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            num_batch = 0
            
            train_metrics_epoch = {}

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                
                y_batch = y[i:i+batch_size]
                y_pred = self.forward(X_batch)

                train_loss = self.loss_fn.forward(y_batch, y_pred)
                epoch_loss += train_loss

                grad = self.loss_fn.backward()
                self.backward(grad, lambda_l2 = lambda_l2)
                optimizer.step(clip_norm=1)
                num_batch+=1
                
                batch_metrics = self.compute_metrics(y_batch, y_pred)
                
                for name, value in batch_metrics.items():
                    if name not in train_metrics_epoch:
                        train_metrics_epoch[name] = []
                    train_metrics_epoch[name].append(value)

            for name, values in train_metrics_epoch.items():
                key = f"train_{name}"
                logs[key] = np.mean(values)
                        
            epoch_loss /= num_batch          
            self.train_losses.append(epoch_loss)
                        
            if X_val is not None:
                self.eval()
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_fn.forward(y_val, y_val_pred)
                self.val_losses.append(val_loss)

                metrics = self.compute_metrics(y_val, y_val_pred)
                
                for name, value in metrics.items():
                    logs[name] = value

                self.train()

            if scheduler is not None:
                scheduler.step()
            
            logs = logs | {
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "lr": optimizer.lr
            }

            for cb in callbacks:
                cb.on_epoch_end(self, epoch, logs)
            if self.stop_training:
                break


        for cb in callbacks:
            cb.on_train_end(self, epoch, logs)


    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    
    def predict(self, X):
        self.eval()
        return self.forward(X)
    
    
    def evaluate(self, X, y, loss_fn):
        self.eval()
        y_pred = self.forward(X)
        loss = loss_fn.forward(y, y_pred)
        self.train()
        return loss
        
    def compile(self, loss, optimizer_name="Adam", metrics=None, lr=0.001):
        self.loss_fn = loss
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.metrics = metrics if metrics is not None else []

    def compute_metrics(self, y_true, y_pred):
        results = {}        
        for metric in self.metrics:
            name = metric.__name__
            results[name] = metric(y_true, y_pred)


        return results
        
        

    def save(self, path):
        with h5py.File(f"models_saved/{path}.h5", "w") as f:
            for i, layer in enumerate(self.layers):
                if hasattr(layer, "W"):
                    grp = f.create_group(f"layer_{i}")
                    grp.create_dataset("W", data=layer.W)
                    grp.create_dataset("b", data=layer.b)

    def load(self, path):
        with h5py.File(f"models_saved/{path}.h5", "r") as f:
            for i, layer in enumerate(self.layers):
                if hasattr(layer, "W"):
                    layer.W = f[f"layer_{i}/W"][:]
                    layer.b = f[f"layer_{i}/b"][:]




