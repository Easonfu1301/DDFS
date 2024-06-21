# import uproot3 as uproot
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('Tkagg')
import time


def _write_root(file, table, treename='kalman', compression=-1):
    # for key in table:
    #     table[key] = np.array(table[key])

    # if compression == -1:
    #     compression = uproot.write.compress.LZ4(4)

    with uproot.recreate(file) as fout:
        print("start storing data to root file")
        t = time.time()
        fout[treename] = table
        fout[treename].show()
        # print(fout[treename].keys())

    print(f"Write to {file} successfully!, time cost: {time.time()-t} s")




def root_plot_test(filename):
    treename = "kalman"

    with uproot.open(filename) as f:
        tree = f[treename]
        print(tree)
        # 假设要绘制名为"some_variable"的变量的直方图
        print(tree.keys())
        x = "p"
        y = "backward_dr"
        datax = np.array(tree[x].array())
        datay = np.array(tree[y].array())[:, 1]


    # 绘制直方图
    plt.plot(datax, datay, 'o', markersize=5)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("Histogram of some_variable")
    plt.show()






if __name__ == "__main__":
    # table = {}
    # table["a"] = np.random.random((200,20000))
    # table["b"] = np.random.random((200,20000))
    # table["c"] = np.random.random((200,20000))
    # # print(table)
    # t = time.time()
    # _write_root('test2.root', table)

    root_plot_test("test_new2/kalman_kalman_post.root")



    print(1)

    # # 读取ROOT文件
    # file = "formal.root"
    #
    # # 读取数据
    # root_plot_test(file)
