from Modules.textureOptimization import TextureOptimization
import os
import numpy as np
import cv2
import datetime


def main():
    texdir = './tex/'
    texs = os.listdir(texdir)
    texs.sort()

    ws = [34, 65, 74]
    Ow = 1024
    Oh = 1024
    for w in ws:
        for i, tex in enumerate(texs):
            start = datetime.datetime.now()
            print(start.isoformat())
            img = cv2.imread(texdir + tex)

            textureOptimization = TextureOptimization(img)
            out = np.random.rand(Ow, Oh, 3) * 255
            result = textureOptimization.synthesis(img, out, w)

            name = texs[i].split('.')[0]
            prefix = '{0}_{1}x{2}_b{3}'.format(name, Ow, Oh, w)

            textureOptimization.animation.save(
                './process/' + prefix + '_anim.gif', writer="imagemagick")

            outname = prefix + '_result.jpg'
            cv2.imwrite('./result/' + outname, result)

            print(textureOptimization.history)
            end = datetime.datetime.now()
            print(end.isoformat())
            print('time: ', end - start)
            print("=======================")


if __name__ == '__main__':
    main()
