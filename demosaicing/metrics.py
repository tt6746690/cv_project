import math
import numpy as np

# Compute mean squared error for images `A` and `B`
# 
def mse(A, B):
    assert(A.shape == B.shape), f"A ({A.shape}) B ({B.shape})"
    mse = (A.astype(np.float32) - B.astype(np.float32)) ** 2
    mse = np.mean(mse)
    return mse

# Compute Peak signal to noise ratio (PSNR) for images `A` and `B`
def psnr(A,B):
    assert(np.iinfo(A.dtype).bits == np.iinfo(B.dtype).bits)
    peak = 2**(np.iinfo(np.uint8).bits) - 1
    sqrtmse = math.sqrt(mse(A,B))
    if sqrtmse == 0:
        return 0
    else:
        return 10 * math.log10(peak**2/sqrtmse)
    