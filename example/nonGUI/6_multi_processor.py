import math
import numpy as np
import pandas as pd
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from multiprocessing import Pool


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import DDFS.common_test_parameter2 as ctp
import DDFS.analytic_method as am
import DDFS.kalman_method as km

from DDFS.element import *
from copy import deepcopy
from DDFS.design_dectetor import export_design

# plt.ion()
np.set_printoptions(precision=4, suppress=True)
para_p = {
        "type": "steps",
        "maxvalue": 100,
        "minvalue": 5,
        "steps": 5,
        "count": 0
    }

para_t = {
    "type": "fixed",
    "value": 0.3
}

para_phi = {
    "type": "even",
    "minvalue": 1,
    "maxvalue": 1,
}

emit_mode = {
    "p": para_p,
    "theta": para_t,
    "phi": para_phi
}




def worker(N):
    d = Detector()
    d.add_layer(SiLayer(0.0015, 10, 4000, 0, 9.9, 9.9))
    for rr in np.linspace(20, 2020, N):
        d.add_layer(SiLayer(0.002  , rr, 4000, 1, 0.02, 0.02))

    e = Environment()
    e.update_environment("B", 3)
    e.update_environment("position_resolution", True)

    m = Emitter()
    p = Particle()
    p.update_particle("Charge", -1)
    p.update_particle("Mass", 0.106)
    m.add_particle(p, 1, deepcopy(emit_mode))

    dec_info = d.get_param()
    envir_info = e.get_param()
    res_k = km.Resolution(dec_info, envir_info, m)



    testnum = 60000
    re_k = Result(testnum)
    re_k.set_emit_mode(deepcopy(emit_mode))

    for i in tqdm(range(testnum), desc="Processing" + str(N), position=0):
        res_k.generate_path()
        res_k.generate_ref_path()
        res_k.backward_kalman_estimate()
        ini, ret = res_k.result_analysis()
        re_k.append(ini, ret)
    re_k.kalman_post_process_all(re_k.emit_mode, len(d))
    # print(idx)
    ini = np.array([re_k.post_initial["p"]])
    rsl = np.array(re_k.post_result["backward_dp"])[:, 1]
    # print(np.array([ini[0], rsl]))
    return np.array([ini[0], rsl])


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    pool = Pool(20)
    result00 = []
    test_N = np.linspace(5, 200, 20, dtype=int)
    for N in test_N:
        result00.append(pool.apply_async(func=worker, args=(N,)))  # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去

    result1 = [res.get() for res in result00]  # get()方法返回的结果和传入的顺序一致

    pool.close()
    pool.join()  # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭。
    pool.terminate()
    # print(np.array(result1))

    re = {
        "re": np.array(result1)
    }


    plt.plot(test_N, np.array(result1)[:, 1], '.-')
    plt.xlabel("Number of layers", fontweight='bold', fontsize=10)
    # plt.ylabel(y_axis + " (" + Y_UNIT[y_axis] + ")")
    plt.ylabel("$\mathbf{dp_t / p_t}$", fontweight='bold', usetex=True, fontsize=10)
    plt.title("Resolution dp vs. Number of layers", fontweight='bold', fontsize=10)

    ax = plt.gca()
    plt.legend(np.linspace(5, 100, 5))

    ax.minorticks_on()
    # 设置主刻度线的样式
    ax.tick_params(which='major', length=5, width=1.5, direction='in', right=True,
                   top=True)
    # 设置小刻度线的样式
    ax.tick_params(which='minor', length=3, width=1, direction='in', right=True,
                   top=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # plt.scatter(x_data, y_data, marker='o', label='Result Plot', c=c_data, cmap='jet', s=20)
    plt.grid(True, linestyle='-.', linewidth=0.5, color='black',
             alpha=0.2)  # Add denser grid lines



    plt.pause(10000000)
