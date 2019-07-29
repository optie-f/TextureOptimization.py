import numpy as np
import time
from sklearn.cluster import KMeans
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Timer:
    def __init__(self):
        self.startTime = time.time()

    def startLap(self):
        self.currentTime = time.time()

    def printLap(self):
        print('Lap:', self.sec2str(time.time() - self.currentTime))

    def printTotal(self):
        print('elapsed:', self.sec2str(time.time() - self.startTime))

    def sec2str(self, time):
        sec = int(time)
        return '{0:02d}:{1:02d}:{2:02d}:{3:03d}'.format(sec // (60 * 60), sec // 60, sec % 60, int((time - sec) * 1000))


class HierarchicalKMeansTree:
    def __init__(self, X, k=4, thr=0.01):
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

        while not node['isLeaf']:
            argmin_ix = 0
            min_dist = 99999999999
            for i, branch in enumerate(node['branches']):
                dist = np.sum((branch['center'] - x)**2)
                if dist < min_dist:
                    argmin_ix = i
                    min_dist = dist
            node = node['branches'][argmin_ix]['node']

        dist = np.sum((node['entities'] - x)**2, axis=1)
        argmin_ix = np.argmin(dist)
        return node['ptrs'][argmin_ix], node['entities'][argmin_ix]


class TextureOptimization:
    def __init__(self, tex):
        self.Z = [tex]

    def synthesis(self, Z, X, W, init=True):
        """
        Z: input texture (r,c,ch)
        X: output (r,c,ch)
        W: width of a neighbourhood (from center to border)
        """
        timer = Timer()

        Zr, Zc = Z.shape[:2]
        Z_viewSize = (
            Zr - W * 2,
            Zc - W * 2,
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
        print('total block num of input:', N)

        timer.startLap()
        hierarchicalKMeansTree = HierarchicalKMeansTree(allBlockVecs)
        print('- built Hierarchical K-Means Tree')
        timer.printLap()

        Xr, Xc = X.shape[:2]
        p_rowRange = np.arange(W, Xr, W + 1)
        p_colRange = np.arange(W, Xc, W + 1)
        p_rowRange[-1] = Xr - W - 1
        p_colRange[-1] = Xc - W - 1
        row_p_num = len(p_rowRange)
        col_p_num = len(p_colRange)

        itr = 0
        print('x_p num:', row_p_num * col_p_num)
        blockPtrs = np.zeros(row_p_num * col_p_num * ch, dtype='uint32')

        if init:
            # 出力の初期化 簡単のため, 近傍がはみ出す部分も生成しておく
            X = np.random.randint(0, 256, X.shape, dtype='uint8')

        # 座標とチャンネルを指定すると1d-arrayとしてのindexが帰ってくる関数としての配列
        X_ind1d = np.arange(Xr * Xc * ch).reshape(Xr, Xc, ch)

        fig = plt.figure(figsize=(5, 5))
        ims = []
        while True:
            timer.startLap()
            z_p_stacks = [[] for i in range(Xr * Xc * ch)]
            # Maximization: find nearest {z_p}
            diff = 0
            searchCnt = 0
            for pos_y in p_rowRange:
                for pos_x in p_colRange:
                    x_p = X[(pos_y - W):(pos_y + W + 1),
                            (pos_x - W):(pos_x + W + 1)].flatten()
                    ptr, z_p = hierarchicalKMeansTree.search(x_p)
                    z_p = z_p.reshape(w, w, ch)
                    for i, k in enumerate(range((pos_y - W), (pos_y + W + 1))):
                        for j, l in enumerate(range((pos_x - W), (pos_x + W + 1))):
                            for m in range(ch):
                                z_p_stacks[X_ind1d[k, l, m]].append(
                                    z_p[i, j, m])
                    diff += (blockPtrs[searchCnt] != ptr)
                    blockPtrs[searchCnt] = ptr
                    searchCnt += 1
            if (diff == 0) | (itr >= 50):
                break
            # Expectation: update x
            E = 0
            X = X.flatten()
            for i, stack in enumerate(z_p_stacks):
                arr = np.array(stack)
                mu = arr.mean()
                X[i] = mu
                E += 0 if len(arr) == 1 else ((arr - mu)**2).sum()
            X = X.reshape(Xr, Xc, ch)
            im = plt.imshow(X[:, :, [2, 1, 0]].astype('int').clip(0, 255), animated=True)
            ims.append([im])

            print('itr:', itr, 'diff:', diff, 'E:', E, end=' ')
            itr += 1
            timer.printLap()

        fps = 8
        self.animation = animation.ArtistAnimation(fig, ims, interval=1000 // fps, blit=True, repeat_delay=1000)
        print('- synthesis converged')
        timer.printTotal()

        return X
