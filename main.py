from Modules.textureOptimization import TextureOptimization
import os
import numpy as np
import cv2


def main():
    texdir = './tex/'
    texs = os.listdir(texdir)
    texs.sort()

    ws = [2, 4, 8, 16]
    Ow = 256
    Oh = 256
    for w in ws:
        for i, tex in enumerate(texs):
            img = cv2.imread(texdir + texs[i])

            textureOptimization = TextureOptimization(0)
            out = np.zeros((Ow, Oh, 3))
            result = textureOptimization.synthesis(img, out, w)

            name = texs[i].split('.')[0]
            prefix = '{0}_{1}x{2}_b{3}'.format(name, Ow, Oh, w * 2 + 1)

            textureOptimization.animation.save(prefix + '_anim.gif', writer="imagemagick")

            outname = prefix + '_result.jpg'
            cv2.imwrite('./result/' + outname, result)


if __name__ == '__main__':
    main()
