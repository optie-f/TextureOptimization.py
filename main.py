from Modules.textureOptimization import TextureOptimization
import os
import numpy as np
import cv2


def main():
    texdir = './tex/'
    texs = os.listdir(texdir)
    texs.sort()
    img = cv2.imread(texdir + texs[0])

    textureOptimization = TextureOptimization(0)
    out = np.zeros((512, 512, 3))
    result = textureOptimization.synthesis(img, out, 16)

    name = 'result.jpeg'
    cv2.imwrite('./result/' + name, result)


if __name__ == '__main__':
    main()
