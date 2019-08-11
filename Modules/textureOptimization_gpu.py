import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse as spsp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import faiss


class TextureOptimization:
    def __init__(self, tex):
        self.Z = tex
        self.history = []

    def synthesis(self, Z, X, W, init=True, dimMax=200):
        """
        Z: input texture (r,c,ch)
        X: output (r,c,ch)
        W: width of a patch
        """
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
            self.pca = self.pca.fit(allBlockVecs)
            DB = self.pca.transform(allBlockVecs)
            print('dim. reduction: {0} -> {1}'.format(p_dim, dimMax))
            print('explained cov. :', self.pca.explained_variance_ratio_.sum())
        else:
            DB = allBlockVecs

        SynthInfo['N'] = N
        SynthInfo['D'] = min(p_dim, dimMax)
        print('Search Space: N={0}, D={1} '.format(
            SynthInfo['N'], SynthInfo['D']))

        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        index = faiss.GpuIndexFlatL2(res, min(p_dim, dimMax), flat_config)
        index.add(DB.astype('float32'))
        print('index added')

        Xr, Xc = X.shape[:2]
        p_rowRange = np.arange(0, Xr - W, step)
        p_colRange = np.arange(0, Xc - W, step)
        row_p_num = len(p_rowRange)
        col_p_num = len(p_colRange)
        Q = row_p_num * col_p_num

        CoefMat = spsp.lil_matrix((Q * p_dim, Xr * Xc * ch))
        print('coefMat size: ({0},{1})'.format(Q * p_dim, Xr * Xc * ch))
        ix_mat = np.zeros((Q, p_dim), dtype=int)
        ix = np.arange(Xr * Xc * ch, dtype=int).reshape(Xr, Xc, ch)

        Q_ix = 0
        for (i, pos_y) in enumerate(p_rowRange):
            for (j, pos_x) in enumerate(p_colRange):
                # 出力画像をクエリに変換したい
                coef_col_ix = ix[pos_y:(pos_y + W),
                                 pos_x:(pos_x + W)].flatten()
                ix_mat[i * col_p_num + j, :] = coef_col_ix

                coef_row_ix = np.arange(Q_ix, Q_ix + p_dim)
                for (row_ix, col_ix) in zip(coef_row_ix, coef_col_ix):
                    CoefMat[row_ix, col_ix] = 1
                Q_ix += p_dim

        A = spsp.csr_matrix(CoefMat)
        SynthInfo['Q'] = Q
        print('query size: Q =', Q)

        Ix = np.zeros(Q, dtype=int)
        itr = 0
        fig = plt.figure(figsize=(15, 15))
        ims = []
        SynthInfo['iteration'] = []

        if init:
            Ix = np.random.randint(Q, size=Q)
            b = allBlockVecs[Ix].flatten()
            sol, istop, itn, norm = spsp.linalg.lsmr(A, b)[:4]
            X = sol.reshape(Xr, Xc, -1)
            itr += 1
            itrInfo = {"itr": itr, "log energy": norm,
                       "lsmr istop": istop, "lsmr iter": itn}
            SynthInfo['iteration'].append(itrInfo)
            print(itrInfo)

        while True:
            im = plt.imshow(X[:, :, [2, 1, 0]].astype(
                'int').clip(0, 255), animated=True)
            ims.append([im])
            # Maximization: find nearest {z_p}
            Query = X.flatten()[ix_mat]
            if p_dim > dimMax:
                Query = self.pca.transform(Query)

            _D, Ix_next = index.search(Query.astype('float32'), 1)

            if np.all(Ix == Ix_next) | (itr > 100):
                break
            Ix = np.copy(Ix_next)

            # Expectation: update x
            b = allBlockVecs[Ix].flatten()
            sol, istop, itn, norm = spsp.linalg.lsmr(A, b)[:4]
            X = sol.reshape(Xr, Xc, -1)
            itr += 1
            itrInfo = {"itr": itr, "log energy": norm,
                       "lsmr istop": istop, "lsmr iter": itn}
            SynthInfo['iteration'].append(itrInfo)
            print(itrInfo)

        fps = 8
        self.animation = animation.ArtistAnimation(
            fig, ims, interval=1000 // fps, blit=True, repeat_delay=1000)
        print('- synthesis converged')
        self.history.append(SynthInfo)

        return X
