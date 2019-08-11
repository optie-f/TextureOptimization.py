import os
import textwrap


def exp190730():
    text = textwrap.dedent("""
    # 2019/07/30 Result

    前回同様, textures.com の素材を使うことにした.

    効率化のため実験記録も Python で生成することにした. 次回以降、実行時間も(printで書き出すだけではなくて)変数として記録するようにしたい.

    結果としては, いずれにしてもブロックサイズがおそらく小さすぎるように見える. 高速化とともに, もう少し大きいサイズで試してみる必要がありそう.

    ブロックサイズに応じて k-means tree の構築時間がそこそこボトルネックになっている.
    [近似最近傍探索の最前線](https://speakerdeck.com/matsui_528/jin-si-zui-jin-bang-tan-suo-falsezui-qian-xian) を参考に, 時空間計算量とアルゴリズムについていくらか考察したい.

    収束判定についてもやや面倒がある. Z_p の diff に関して振動することが多々あるが, 振動がいつどこで発生するかはかなり謎. 各ループを t < 1s  程度に高速化できたとして, itr=100 くらいまでで打ち切るほうが精神衛生上はよさそう……？

    | input | output | optimization |
    |:-----:|:------:|:------------:|
    """)
    ws = [2, 4, 8, 16]
    Ow = 256
    Oh = 256
    texdir = './tex/'
    resultdir = './result/'
    gifdir = './optimization_gif/'
    texs = os.listdir(texdir)
    texs.sort()
    for w in ws:
        for i, texname in enumerate(texs):
            inpath = texdir + texname

            name = texs[i].split('.')[0]
            prefix = '{0}_{1}x{2}_b{3}'.format(name, Ow, Oh, w * 2 + 1)

            gifpath = gifdir + prefix + '_anim.gif'
            outpath = resultdir + prefix + '_result.jpg'

            text += '| ![x]({0}) | ![x]({1})  | ![x]({2}) | \n'.format(inpath, outpath, gifpath)
            text += '| {0}  size:(256,256) | size:({1},{2})   block:({3},{3}) | | \n'.format(texname, Ow, Oh, w * 2 + 1)

    return text


def main():
    md = textwrap.dedent("""
    # Texture Optimization implementation & experiments
    """)

    expReports = [exp190730]

    for expReport in expReports:
        md += expReport()

    with open('./experiments.md', 'w') as f:
        f.write(md)


if __name__ == "__main__":
    main()
