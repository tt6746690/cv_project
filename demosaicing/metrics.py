import numpy as np

# Compute mean squared error for images `A` and `B`
def mse(A, B):
    assert(A.shape == B.shape)
    mse = (A.astype(np.float32) - B.astype(np.float32)) ** 2
    mse = np.sum(mse) / (A.shape[0] * A.shape[1])
    return mse

# Compute Peak signal to noise ratio (PSNR) for images `A` and `B`
def psnr(A,B):
    assert(np.iinfo(A.dtype).bits == np.iinfo(B.dtype).bits)
    info = np.iinfo(A.dtype)
    return 10 * np.log10(float((info.max-info.min)^2) / np.sqrt([ mse(A,B) ]))[0]
    