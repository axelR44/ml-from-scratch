import numpy as np

from src.data.dataloader import DataLoader


def test_dataloader_shapes():
    X = np.random.randn(100, 1, 28, 28)
    y = np.random.randint(0, 10, size=(100,))

    loader = DataLoader(X, y, batch_size=32, shuffle=False)
    batches = list(loader)
    assert len(batches) == 4

    X0, y0 = batches[0]

    assert X0.shape == (32, 1, 28, 28)
    assert y0.shape == (32,)

    X_last, y_last = batches[-1]

    assert X_last.shape == (4, 1, 28, 28)
    assert y_last.shape == (4,)

def test_dataloader_drop_last():
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    loader = DataLoader(X,y, batch_size=32,shuffle=False, drop_last=True)
    batches = list(loader)

    assert len(batches) == 3

    for X_batch, y_batch in batches:
        assert X_batch.shape[0] == 32
        assert y_batch.shape[0] == 32

def all_test_dataloader():
    test_dataloader_shapes()
    test_dataloader_drop_last()

    print("Tests DataLoader OK")
    
if __name__ == "__main__":
    all_test_dataloader