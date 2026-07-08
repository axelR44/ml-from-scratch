import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Pool2DBase:
    def __init__(self, pool_size=2, stride=None, padding=0):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding

    def _pad_input(self, X, mode):
        if self.padding == 0:
            return X
        if mode == "max":
            pad_value = -np.inf
        else:
            pad_value = 0.0
        return np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant", constant_values=pad_value)

    def _compute_output_shape(self, H, W):
        H_out = (H - self.pool_size) // self.stride+ 1
        W_out = (W - self.pool_size) // self.stride + 1
        return H_out, W_out

    def _extract_patches(self, X):
        batch_size, C, H, W = X.shape
        H_out, W_out = self._compute_output_shape(H, W)

        patches = sliding_window_view(X, window_shape=(self.pool_size, self.pool_size), axis=(2, 3))
        patches = patches[:, :, ::self.stride, ::self.stride, :, :]
        patches = patches[:, :, :H_out, :W_out, :, :]
        return patches

    def _scatter_patches_to_input(self, dPatches):
        batch_size, C, H_pad, W_pad = self.X_padded.shape
        _, _, H_out, W_out, _, _ = dPatches.shape
        dX_padded = np.zeros_like(self.X_padded)

        batch_idx = np.arange(batch_size)[:, None, None, None, None, None]
        channel_idx = np.arange(C)[None, :, None, None, None, None]

        rows = (np.arange(H_out)[:, None]*self.stride+np.arange(self.pool_size)[None,:])
        cols = (np.arange(W_out)[:, None]*self.stride+np.arange(self.pool_size)[None,:])
        row_idx = rows[None, None,:, None,:, None]
        col_idx = cols[None, None, None,:, None, :]
        np.add.at(dX_padded, (batch_idx, channel_idx, row_idx, col_idx), dPatches)

        if self.padding > 0:
            return dX_padded[:,:, self.padding:-self.padding, self.padding:-self.padding]

        return dX_padded
    
class AvgPool2D(Pool2DBase):
    def forward(self, X):
        self.X_shape = X.shape
        self.X_padded = self._pad_input(X, mode="avg")
        self.patches = self._extract_patches(self.X_padded)
        out = np.mean(self.patches, axis=(4, 5))
        return out

    def backward(self, dZ):
        dPatches = dZ[:,:,:,:, None, None] / (self.pool_size * self.pool_size)
        dPatches = np.ones_like(self.patches) * dPatches
        dX = self._scatter_patches_to_input(dPatches)
        return dX
    
class MaxPool2D(Pool2DBase):
    def forward(self, X):
        self.X_shape = X.shape
        self.X_padded = self._pad_input(X, mode="max")

        self.patches = self._extract_patches(self.X_padded)
        batch_size, C, H_out, W_out, p, _ = self.patches.shape
        flat = self.patches.reshape(batch_size, C, H_out, W_out, p * p)

        self.argmax = np.argmax(flat, axis=-1)
        out = np.max(flat, axis=-1)
        return out

    def backward(self, dZ):
        batch_size, C, H_out, W_out = dZ.shape
        dPatches_flat = np.zeros((batch_size, C, H_out, W_out, self.pool_size*self.pool_size))

        np.put_along_axis(dPatches_flat, self.argmax[:, :, :, :, None], dZ[:, :, :, :, None], axis=-1)

        dPatches = dPatches_flat.reshape(batch_size, C, H_out, W_out, self.pool_size, self.pool_size)
        dX = self._scatter_patches_to_input(dPatches)
        return dX
    

class GlobalAveragePooling2D:
    def forward(self, X):
        self.input_shape = X.shape
        return np.mean(X, axis=(2, 3))

    def backward(self, dZ):
        batch_size, C, H, W = self.input_shape
        return np.ones(self.input_shape) * (dZ[:, :, None, None] / (H * W))