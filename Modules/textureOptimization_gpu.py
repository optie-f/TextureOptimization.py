import cupy as cp
from cupy.lib.stride_tricks import as_strided
import cupyx.scipy.sparse as cpsp
from sklearn.decomposition import PCA
import faiss
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class TextureOptimization:
    def __init__(self, tex):
        self.Z = [cp.array(tex)]
        self.history = []

    def synthesis(self, Z, X, W, init=True, dimMax=100):
        """
        Z: icput texture (r,c,ch)
        X: output (r,c,ch)
        W: width of a patch
        """
        X = cp.array(X)
        SynthInfo = {}
        step = W // 2

        Zr, Zc, ch = Z.shape
        Z_viewSize = (
            Zr - step * 2,
            Zc - step * 2,
            W,
            W,
            ch
        )
        Z_strides = Z.strides[:2] + Z.strides
        # Z から取りうるすべてのブロック (w*w*ch) を縦横に並べた五次元配列
        blocks = as_strided(Z, Z_viewSize, Z_strides)
        r, c = blocks.shape[:2]

        N = r * c
        p_dim = W * W * ch
        allBlockVecs = blocks.reshape(N, p_dim)

        self.pca = None
        if p_dim > dimMax:
            self.pca = PCA(n_components=dimMax)
            DB = self.pca.fit_transform(allBlockVecs)
        else:
            DB = allBlockVecs

        SynthInfo['N'] = N
        SynthInfo['D'] = min(p_dim, dimMax)
        print('Search Space: N={0}, D={1} ', SynthInfo['N'], SynthInfo['D'])

        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        index = faiss.GpuIndexFlatL2(res, min(p_dim, dimMax), flat_config)
        index.add(cp.asnumpy(allBlockVecs))

        Xr, Xc = X.shape[:2]
        p_rowRange = cp.arange(0, Xr, step)
        p_colRange = cp.arange(0, Xc, step)
        p_rowRange[-1] = min(p_rowRange[-1], Xr - step - 1)
        p_colRange[-1] = min(p_colRange[-1], Xc - step - 1)
        row_p_num = len(p_rowRange)
        col_p_num = len(p_colRange)
        Q = row_p_num * col_p_num

        CoefMat = cp.zeros((Q * p_dim, Xr * Xc * ch))

        for (i, pos_y) in enumerate(p_rowRange):
            for (j, pos_x) in enumerate(p_colRange):
                for k in range(W):
                    for l in range(W):
                        for m in range(ch):
                            row_ix = i * row_p_num + j * col_p_num + k * W + l * W + m
                            col_ix = (pos_y + k) * Xr + (pos_x + l) + m
                            CoefMat[row_ix, col_ix] = 1
        A = cpsp.csr_matrix(CoefMat)
        SynthInfo['Q'] = Q
        print('query size: Q=', Q)

        ix_mat = cp.zeros((Q, p_dim))
        ix = cp.arange(Xr * Xc * ch).reshape(Xr, Xc, ch)
        for (i, pos_y) in enumerate(p_rowRange):
            for (j, pos_x) in enumerate(p_colRange):
                ix_mat[i * col_p_num + j, :] = ix[(pos_y - W):(pos_y + W + 1),
                                                  (pos_x - W):(pos_x + W + 1)].flatten()

        Ix = cp.zeros(Q)
        itr = 0
        fig = plt.figure(figsize=(5, 5))
        ims = []
        SynthInfo['iteration'] = []
        while True:
            # Maximization: find nearest {z_p}
            Query = X[ix_mat]
            if p_dim > dimMax:
                Query = self.pca.fit_transform(Query)

            _D, Ix_next = index.search(Query, 1)

            if (Ix == Ix_next) | (itr > 100):
                break
            Ix = cp.copy(Ix_next)

            # Expectation: update x
            b = DB[Ix].flatten()
            sol = cpsp.linalg.lsqr(A, b)[0]
            X = sol.reshape(Xr, Xc, -1)
            itr += 1

            im = plt.imshow(X[:, :, [2, 1, 0]].astype(
                'int').clip(0, 255), animated=True)
            ims.append([im])

            itrInfo = {"itr": itr, "log energy": cp.log(_D.sum())}
            SynthInfo['iteration'].append(itrInfo)
            print(itrInfo)

        fps = 8
        self.animation = animation.ArtistAnimation(
            fig, ims, interval=1000 // fps, blit=True, repeat_delay=1000)
        print('- synthesis converged')
        self.history.append(SynthInfo)

        return cp.asnumpy(X)
