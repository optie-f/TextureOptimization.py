import numpy as np
import time
from sklearn.cluster import KMeans
from numpy.lib.stride_tricks import as_strided


class HierarchicalKMeansTree:
    def __init__(self, X, k=4, thr=0.001):
        """
        X: (N, d)
        """
        self.total_N = len(X)
        self.indices = np.arange(len(X))
        self.k = k
        self.thr = thr
        self.tree = self.__build__(X, self.indices)

    def __build__(self, X, indices):
        """
        X: (N, d)
        indices: (N)
        """
        if len(X) < self.thr * self.total_N:
            return {
                'isLeaf': True,
                'entities': X,
                'ptrs': indices
            }

        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(X)
        return {
            'isLeaf': False,
            'branches': [
                {
                    'center': kmeans.cluster_centers_[i],
                    'node': self.__build__(X[kmeans.labels_ == i], indices[kmeans.labels_ == i])
                }
                for i in range(self.k)
            ]
        }

    def search(self, x):
        node = self.tree

        while not node.isLeaf:
            argmin_ix = 0
            min_dist = 99999999999
            for i, branch in enumerate(node.branches):
                dist = np.sum((branch.center - x)**2)
                if dist < min_dist:
                    argmin_ix = i
                    min_dist = dist
            node = node.branches[argmin_ix].node

        dist = np.sum((node.entities - x)**2, axis=1)
        argmin_ix = np.argmin(dist)
        return node.ptrs[argmin_ix]


class TextureOptimization:
    def __init__(self, tex):
        self.Z = [tex]

    def synthesis(self, Z, X, W, init=True):
        startTime = time.time()
        """
        Z: input texture (r,c,ch)
        X: output (r,c,ch)
        W: width of a neighbourhood (from center to border)
        """
        Zr, Zc = Z.shape[:2]
        Z_viewSize = (
            Zr - W,
            Zc - W,
            W * 2 + 1,
            W * 2 + 1,
            Z.shape[2]
        )
        Z_strides = Z.strides[:2] + Z.strides
        # Z から取りうるすべてのブロック (w*w*ch) を縦横に並べた五次元配列
        blocks = as_strided(Z, Z_viewSize, Z_strides)
        r, c, w, w, ch = blocks.shape
        # ブロックをベクトル化したものを並べた二次元配列にする
        N = r * c
        p_dim = w * w * ch
        allBlockVecs = blocks.reshape(N, p_dim)
        currentTime = time.time()
        hierarchicalKMeansTree = HierarchicalKMeansTree(allBlockVecs)
        print('- built Hierarchical K-Means Tree')
        print('elapsed:', time.time() - startTime, 'delta:', time.time() - currentTime)

        Xr, Xc = X.shape[:2]
        p_rowRange = np.arange(W, Xr, W)
        p_colRange = np.arange(W, Xc, W)
        row_p_num = len(p_rowRange)
        col_p_num = len(p_colRange)

        # X_viewSize = (
        #     row_p_num,
        #     col_p_num,
        #     w,
        #     w,
        #     ch
        # )
        # X_strides = (X.strides[0]*W, X.strides[0]*W) + X.strides
        # ix = np.arange(Xr * Xc * ch).reshape(Xr, Xc, ch)
        # px_ix = as_strided(ix, X_viewSize, X_strides)

        currentTime = time.time()

        # Xの画素値に関する線形方程式の係数行列. Ax=b の A
        # とくに重みがなければ 0 か 1
        print('- will build CoefMat of', (row_p_num * col_p_num * p_dim, Xr * Xc * ch))
        CoefMat = np.zeros((row_p_num * col_p_num * p_dim, Xr * Xc * ch))

        for (i, pos_y) in enumerate(p_rowRange):
            for (j, pos_x) in enumerate(p_colRange):
                for k in range(w):
                    for l in range(w):
                        for m in range(Z.shape[2]):
                            row_ix = i * row_p_num + j * col_p_num + k * w + l * w + m
                            col_ix = (pos_y + k - W) * Xr + (pos_x + l - W) + m
                            CoefMat[row_ix, col_ix] = 1

        print('- built CoefMat')
        print('elapsed:', time.time() - startTime, 'delta:', time.time() - currentTime)
        itr = 0
        # M-step で選ばれた Zp をすべて繋げたベクトルを作る. 線形方程式 Ax=b の b. 最初はランダムパッチで初期化
        currentTime = time.time()
        blockPtrs = np.random.choice(N, row_p_num * col_p_num)
        if init:
            allZpVec = allBlockVecs[blockPtrs].flatten()
            sol, res, _r, _s = np.linalg.lstsq(CoefMat, allZpVec, rcond=None)
            X = sol.reshape(Xr, Xc, -1)
            print('itr:', itr, 'E:', res, 'elapsed:', time.time() - startTime, 'delta:', time.time() - currentTime)
            itr += 1

        while True:
            currentTime = time.time()
            # Maximization: find nearest {z_p}
            converge = True
            for (i, pos_y) in enumerate(p_rowRange):
                for (j, pos_x) in enumerate(p_colRange):
                    x_p = X[(pos_y - W):(pos_y + W + 1), (pos_x - W):(pos_x + W + 1)].flatten()
                    ptr = hierarchicalKMeansTree.search(x_p)
                    converge &= (blockPtrs[i * col_p_num + j] != ptr)
                    blockPtrs[i * col_p_num + j] = ptr
            if converge:
                break
            # Expectation: update x
            allZpVec = allBlockVecs[blockPtrs].flatten()
            sol, res, _r, _s = np.linalg.lstsq(CoefMat, allZpVec, rcond=None)
            X = sol.reshape(Xr, Xc, -1)
            print('itr:', itr, 'E:', res, 'elapsed:', time.time() - startTime, 'delta:', time.time() - currentTime)
            itr += 1

        print('- synthesis converged')
        print('elapsed:', time.time() - startTime, 'sec')

        return X
