import numpy as np

class MaxPool2D:

    def __init__(self, pool_size=2):
        self.pool_size = pool_size


    def naive_forward(self, X):

        self.X = X
        batch_size, C, H, W = X.shape

        H_out = H // self.pool_size

        W_out = W // self.pool_size

        out = np.zeros((batch_size, C, H_out, W_out))

        # masque complet de la taille de l'entrée
        self.max_mask = np.zeros_like(X, dtype=bool)

        for i in range(H_out):
            for j in range(W_out):

                region = X[
                    :,
                    :,
                    i*self.pool_size:(i+1)*self.pool_size,
                    j*self.pool_size:(j+1)*self.pool_size
                ]

                # (batch,C,1,1)
                max_vals = np.max(region, axis=(2, 3), keepdims=True)

                # masque des maxima
                mask = (region == max_vals)

                # stockage du masque
                self.max_mask[
                    :,
                    :,
                    i*self.pool_size:(i+1)*self.pool_size,
                    j*self.pool_size:(j+1)*self.pool_size
                ] = mask

                # suppression des dimensions 1x1
                out[:, :, i, j] = max_vals[:, :, 0, 0]
        return out
    
    def forward(self, X):
        self.X = X
        batch_size, C, H, W = X.shape
        H_out = H // self.pool_size
        W_out = W // self.pool_size

        # (B,C,H_out,p,W_out,p)
        patches = X.reshape(batch_size, C, H_out, self.pool_size, W_out, self.pool_size)
        # (B,C,H_out,W_out,p,p)
        patches = patches.transpose(0, 1, 2, 4, 3, 5)
        # (B,C,H_out,W_out,p*p)
        patches = patches.reshape(batch_size, C, H_out,W_out,self.pool_size * self.pool_size)
        #position des maximums
        self.argmax = np.argmax(patches,axis=-1)

        #ne garde que les valeurs max
        out = np.max(patches,axis=-1)
        return out

    def naive_backward(self, dZ):
        dX = np.zeros_like(self.X)

        H_out = dZ.shape[2]
        W_out = dZ.shape[3]

        dX = np.zeros_like(self.X)

        for i in range(H_out):
            for j in range(W_out):

                mask = self.max_mask[
                    :,
                    :,
                    i*self.pool_size:(i+1)*self.pool_size,
                    j*self.pool_size:(j+1)*self.pool_size
                ]

                grad = dZ[:, :, i, j]
                grad = grad[:, :, None, None]

                dX[
                    :,
                    :,
                    i*self.pool_size:(i+1)*self.pool_size,
                    j*self.pool_size:(j+1)*self.pool_size
                ] += mask * grad
        return dX
        
    def backward(self, dZ):

        batch_size, C, H_out, W_out = dZ.shape

        dPatches = np.zeros((batch_size, C, H_out, W_out, self.pool_size*self.pool_size))

        np.put_along_axis(dPatches, self.argmax[..., None], dZ[..., None], axis=-1)

        dPatches = dPatches.reshape(batch_size, C, H_out, W_out, self.pool_size, self.pool_size)
        dPatches = dPatches.transpose(0, 1, 2, 4, 3, 5)

        dX = dPatches.reshape(batch_size, C, H_out * self.pool_size, W_out * self.pool_size)

        return dX