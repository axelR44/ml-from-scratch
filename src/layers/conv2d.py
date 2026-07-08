import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, stride = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # poids : (out_channels, in_channels, k, k)
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.b = np.zeros((out_channels, 1))

        self.padding = padding
        self.stride = stride

    def naive_forward(self, X):
        """
        Deprecated.
        use forward() instead.
        """
        self.X = X

        batch_size, _, H, W = X.shape
        k = self.kernel_size

        H_out = H - k + 1
        W_out = W - k + 1

        out = np.zeros((batch_size, self.out_channels, H_out, W_out))

        for n in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        #patch, k*k pixels
                        region = X[n, :, i:i+k, j:j+k]
                        out[n, oc, i, j] = np.sum(region * self.W[oc]) + self.b[oc]

        return out
        
    def forward(self, X):
        self.X = X
        if self.padding > 0:
            X = np.pad(X,(
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding)
                ),mode="constant")
        self.X_padded = X
        batch_size, _, H, W = X.shape
        k = self.kernel_size

        H_out = (H - k)//self.stride + 1
        W_out = (W - k)//self.stride + 1

        cols = self._im2col(X)

        self.cols = cols

        # (out_channels, C*k*k)
        W_col = self.W.reshape(self.out_channels, -1)

        # (batch, H_out*W_out, out_channels)
        out = cols @ W_col.T

        out += self.b.reshape(1, 1, self.out_channels)

        # (batch, out_channels, H_out, W_out)
        out = out.reshape(batch_size, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        return out
    
    def _im2col(self, X):
        batch_size, C, H, W = X.shape
        k = self.kernel_size
                
        H_out = (H - k)//self.stride + 1
        W_out = (W - k)//self.stride + 1

        cols = []

        for i in range(H_out):
            for j in range(W_out):
                
                row = i * self.stride
                col = j * self.stride

                patch = X[:, :, row:row+k, col:col+k]

                # (batch, C*k*k)
                patch = patch.reshape(batch_size, -1)

                cols.append(patch)

        # (batch, H_out*W_out, C*k*k)
        cols = np.stack(cols, axis=1)

        return cols
    
    def _col2im(self, cols, X_shape):

        batch_size, C, H, W = X_shape

        k = self.kernel_size

        
        H_out = (H - k)//self.stride + 1
        W_out = (W - k)//self.stride + 1


        dX = np.zeros(X_shape)

        patch_idx = 0

        for i in range(H_out):
            for j in range(W_out):
                
                row = i * self.stride
                col = j * self.stride

                patch = cols[:, patch_idx]
                patch = patch.reshape(batch_size, C,k, k)
                dX[:, :, row:row+k,col:col+k] += patch
                patch_idx += 1
        return dX
        
    def naive_backward(self, dZ, lambda_l2=0.0):
        """
        Deprecated.
        use backward() instead.
        """

        batch_size, _, H, W = self.X.shape
        k = self.kernel_size

        H_out = H - k + 1
        W_out = W - k + 1

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        dX = np.zeros_like(self.X)

        for n in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        region = self.X[n, :, i:i+k, j:j+k]
                        grad = dZ[n, oc, i, j]
                        self.dW[oc] += grad * region
                        self.db[oc] += grad
                        
                        dX[n, :, i:i+k, j:j+k] += grad * self.W[oc]

        if lambda_l2 > 0:
            self.dW += lambda_l2 * self.W

        return dX
    

    def backward(self, dZ, lambda_l2=0.0):
        #on reprend la shape du X qui avait été paddé
        batch_size, C, H, W = self.X_padded.shape
        k = self.kernel_size

                
        H_out = (H - k)//self.stride + 1
        W_out = (W - k)//self.stride + 1

        nb_patches = H_out * W_out

        self.db = np.sum(dZ, axis=(0, 2, 3), keepdims=False).reshape(self.out_channels, 1)

        dZ_col = dZ.transpose(0, 2, 3, 1)
        dZ_col = dZ_col.reshape(batch_size, H_out * W_out, self.out_channels)
        _, _, kernel_dim = self.cols.shape

        grad_output = dZ_col.reshape(batch_size * nb_patches, self.out_channels)

        patches = self.cols.reshape(batch_size * nb_patches, kernel_dim)

        dW_col = grad_output.T @ patches
        
        self.dW = dW_col.reshape(self.W.shape)
        W_col = self.W.reshape(self.out_channels,-1)

        dCols = np.matmul(dZ_col, W_col)
        dX = self._col2im(dCols, self.X_padded.shape)
        if self.padding > 0:
            dX = dX[
                :,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding
            ]

        if lambda_l2 > 0:
            self.dW += lambda_l2 * self.W

        return dX