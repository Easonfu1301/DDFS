import time
# from pympler import asizeof
import numpy as np
import pandas as pd
import warnings
import uproot
import os
import scipy
import sys
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import copy
from itertools import product
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import norm
from tqdm import tqdm

# matplotlib.use('TkAgg')
import random
import json

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.weight'] = 'bold'

emit_para1 = {
    "type": "steps",
    "maxvalue": 120,
    "minvalue": 5,
    "steps": 10,
    "count": 0
}
#
# emit_para2 = {
#     "type": "even",
#     "maxvalue": 1e-3,
#     "minvalue": -1e-3
# }
#
# emit_para3 = {
#     "type": "even",
#     "maxvalue": 0.5 * 2 * math.pi,
#     "minvalue": -0.5 * 2 * math.pi
# }

# emit_para1 = {
#         "type": "steps",
#         "maxvalue": 10,
#         "minvalue": 1,
#         "steps": 10,
#         "count": 0
#     }

emit_para2 = {
    "type": "steps",
    "maxvalue": 0.7,
    "minvalue": 0.1,
    "steps": 4,
    "count": 0
}

emit_para3 = {
    "type": "fixed",
    "value": 1
}

default_mode = {
    "p": emit_para1,
    "theta": emit_para2,
    "phi": emit_para3
}

unit_dict = {
    "p": "GeV",
    "theta": "°",
    "phi": "°",
    "Mass": "GeV",
    "Charge": "e",
    "B": "T",
}

X_AXIS_TYPE_KALMAN = ["p", "theta", "phi"]
Y_AXIS_TYPE_KALMAN = ["dr", "dz", "dt", "df", "dp", "dp2"]
FILTER_TYPE_KALMAN = ["forward", "backward"]
C_AXIS_TYPE_KALMAN = ["dr", "dz", "dt", "df", "dp", "dp2", "p", "theta", "phi"]

X_UNIT_LABEL = {
    "p": r"$\mathbf{p_t}$  (GeV)",
    "theta": r"$\mathbf{\theta}$  (°)",
    "phi": r"$\mathbf{\phi}$  (rads)",
    "det": r"\textbf{det}"
}

Y_UNIT_LABEL = {
    "dr": r"\textbf{$\mathbf{dr}$  $(\mu m)$}",
    "dz": r"\textbf{$\mathbf{dz}$  $(\mu m)$}",
    "dt": r"\textbf{$\mathbf{d \theta}$  (mrads)}",
    "df": r"\textbf{$\mathbf{d \phi}$  (mrads)}",
    "dp": r"\textbf{$\mathbf{dp_t/p_t (\times 10^{-3})} $}",
    "dp2": r"\textbf{$\mathbf{dp_t/p_t^2} \ \ (\times 10^{-5}\ \mathrm{GeV^{-1}})$}",
    "p": r"$\mathbf{p_t}$  (GeV)",
    "theta": r"$\mathbf{\theta}$  (°)",
    "phi": r"$\mathbf{\phi}$  (rads)",
    "det": r"\textbf{det}"
}

INITIAL_TYPE = ["B", "p", "theta", "phi", "MS", "RE", "mass", "charge", "beam_spot"]
A_RESULT_TYPE = ["dr", "dz", "dt", "df", "dp", "dp2", "det"]
K_POST_RESULT_TYPE = [
    "forward_dr", "forward_dz", "forward_dt", "forward_df", "forward_dp", "forward_dp2",
    "backward_dr", "backward_dz", "backward_dt", "backward_df", "backward_dp", "backward_dp2",
]

K_RESULT_TYPE = [
    "forward_dr", "forward_dz", "forward_dt", "forward_df", "forward_dp", "forward_dp2",
    "backward_dr", "backward_dz", "backward_dt", "backward_df", "backward_dp", "backward_dp2",
    "f_dr_first", "f_dz_first", "f_dt_first", "f_df_first", "f_dp_first", "f_dp2_first",
    "b_dr_first", "b_dz_first", "b_dt_first", "b_df_first", "b_dp_first", "b_dp2_first",
    "measure_path", "res_ori", "res_forward", "res_backward", "chi2_forward", "chi2_backward",
]


class Detector:
    def __init__(self):
        """
            创建一个探测器，由layer构成
        """

        self.layer_sum = 0

        self.layers = []

        self.layer_index = []

    def __str__(self):

        return f"\nLayer Counting: {self.layer_sum}\n" \
               f"Detector INFO: \n" \
               f"--------------------------------------------------\n" \
               f"{self.get_info('text')}" \
               f"--------------------------------------------------\n"

    def __len__(self):
        return self.layer_sum

    def copy(self):
        copy_temp = Detector()
        copy_temp.layer_index = self.layer_index
        copy_temp.layers = [layer.copy() for layer in self.layers]
        copy_temp.layer_sum = self.layer_sum
        return copy_temp

    def load_designed(self, file_path):
        """
        从文件中读取设计好的探测器
        :param file_path: 文件名
        """

        self.layer_sum = 0

        self.layers = []

        self.layer_index = []

        df = pd.read_csv(file_path)
        print(df)

        radius_list = df['Radius'].tolist()
        budget_list = df['Budget'].tolist()
        half_z_list = df['Half_z'].tolist()
        loc0 = df['Location 0'].tolist()
        loc1 = df['Location 1'].tolist()
        effi = df['Efficiency'].tolist()

        for i in range(0, len(radius_list)):
            self.add_layer(
                SiLayer(material_budget=budget_list[i], radius=radius_list[i], efficiency=effi[i],
                        loc0=loc0[i], loc1=loc1[i], half_z=half_z_list[i]))

    def store_designed(self, file_path):
        """
        将设计好的探测器存储到文件中
        :param file_path: 文件名
        """
        radius_list, material_budget, efficiency_list, loc0_list, loc1_list, halfz_list = self.get_param()

        data = {
            'Radius': radius_list,
            'Budget': material_budget,
            'Half_z': halfz_list,
            'Location 0': loc0_list,
            'Location 1': loc1_list,
            'Efficiency': efficiency_list
        }

        df = pd.DataFrame(data)
        # Export the data to a CSV file
        # Export the data to a CSV file test

        df.to_csv(file_path, index=False)

    def get_info(self, info_type):
        """
        用于获取当前的参数和信息
        :param info_type: "dir", "text", "list"
        :return: "dir" 返回一个字典，"text" 返回一个字符串，"list" 返回一个列表
        """
        if info_type == "dir":
            return self.layers
        elif info_type == "text":
            layer_info_text = ""
            for idx, layer_index in enumerate(self.layer_index):
                layer_info_text += "Idx:" + str(idx) + "\t" + self.layers[layer_index].get_info("text") + "\n"
            return layer_info_text
        elif info_type == "list":
            # print(self)
            list_all_layer = [list(self.layers[layer_index].get_info("dir").values()) for layer_index in
                              self.layer_index]
            return list_all_layer
        else:
            pass

    def sort_layer(self):
        """
        将增加进的所有层按照粒子经过的顺序进行排序， 防止设计的时候层数出错
        """
        layer_info_temp = [layer.get_info("dir") for layer in self.layers]
        sorted_layer_info_temp = sorted(enumerate(layer_info_temp), key=lambda x: x[1]["Radius"])
        self.layer_index = [index for index, _ in sorted_layer_info_temp]
        self.layer_sum = len(self.layers)

        # print(self.layer_index)

    def check_consistency(self):
        """
        检查当前情况下是否符合现实情况，目前的判断是layer的半径不能冲突
        :return: True or False
        """
        rad_temp = [layer.get_info("dir")["Radius"] for layer in self.layers if layer.get_info("dir")["Radius"] != -1]
        return len(rad_temp) == len(set(rad_temp))

    def add_layer(self, layer):
        """
        :param layer: 在保证设计合理的情况下为 Detector 增加一个 layer 对象,并更新 layer_index
        """
        #
        self.layers.append(layer)

        if self.check_consistency():
            self.sort_layer()
        else:
            del self.layers[-1]

    def delete_layer(self, layer_idx):
        """
        :param layer_idx: 通过最新的 layer_index 来删除一个layer 并更新 layer_index
        :return:
        """
        # 为 Detector 删除一个layer对象
        del self.layers[self.layer_index[layer_idx]]
        self.sort_layer()

    def update_layer(self, layer_index, layer_param_type, value):
        """
        :param layer_index: 要更新参数 layer 的 layer index
        :param layer_param_type: 更新参数类型，只能是 "Material_budget", "Radius", "Half_z", "Efficiency", "Loc0", "Loc1" 中的一个
        :param value: 要更新的值
        :return: 1
        """
        # 按照 layer_index 用 new_layer 替换 old_layer

        idx = self.layer_index[layer_index]

        tmp = self.layers[idx].get_info("dir")[layer_param_type]

        self.layers[idx].update_layer(layer_param_type, value)

        if self.check_consistency():
            # print("layer" +str(layer_index) + "'s " +  layer_param_type +" is modify to " + str(value))
            self.sort_layer()
        else:
            print("cannot perform change, check layer")
            self.layers[idx].update_layer(layer_param_type, tmp)

    def visualize_detector(self):
        radius_list, material_budget, efficiency_list, loc0_list, loc1_list, halfz_list = self.get_param()
        num_layers = len(halfz_list)

        fig, ax = plt.subplots()
        # ax.set_aspect('equal')
        bugmax = max(material_budget)
        bugmin = min(material_budget)
        for i in range(num_layers):
            # Calculate the coordinates of the layer boundaries
            x = [-halfz_list[i], halfz_list[i]]
            y = [radius_list[i], radius_list[i]]
            y2 = [-radius_list[i], -radius_list[i]]

            # Plot the layer boundaries as lines, color-coded by material
            ax.plot(x, y, color=cm.jet((material_budget[i] - bugmin) / (bugmax - bugmin + 1e-5)), linewidth=2,
                    alpha=0.8)
            ax.plot(x, y2, color=cm.jet((material_budget[i] - bugmin) / (bugmax - bugmin + 1e-5)), linewidth=2,
                    alpha=0.8)

        ax.plot(0, 0, marker='o', markersize=5, color='red', label='Collision Center')

        ax.set_xlim(-max(halfz_list) * 1.2, max(halfz_list) * 1.2)
        ax.set_ylim(-max(radius_list) * 1.2, max(radius_list) * 1.2)

        plt.xlabel('Z (mm)')
        plt.ylabel('R (mm)', labelpad=-5)
        plt.title('Detector Geometry', fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.93)
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

        # plt.show()
        return fig, ax

    def get_param(self):
        """
        返回实验所需要的参数，前面的get_info主要是内部调用，这个是给外部的
        :return:
        """
        # 返回我们所需的eff等参数
        radius_list = [self.layers[layer_idx].get_info("dir")["Radius"] for layer_idx in self.layer_index]
        material_budget = [self.layers[layer_idx].get_info("dir")["Material_budget"] for layer_idx in self.layer_index]
        efficiency_list = [self.layers[layer_idx].get_info("dir")["Efficiency"] for layer_idx in self.layer_index]
        loc0_list = [self.layers[layer_idx].get_info("dir")["Loc0"] for layer_idx in self.layer_index]
        loc1_list = [self.layers[layer_idx].get_info("dir")["Loc1"] for layer_idx in self.layer_index]
        halfz_list = [self.layers[layer_idx].get_info("dir")["Half_z"] for layer_idx in self.layer_index]

        # print("radius: \t\t\t", radius_list, "\n",
        #       "material_budget: \t\t", material_budget, "\n",
        #       "efficiency_list: \t", efficiency_list, "\n",
        #       "loc0_list: \t\t", loc0_list, "\n",
        #       "loc1_list: \t\t", loc1_list, "\n",
        #       "halfz_list: \t\t", halfz_list)
        return radius_list, material_budget, efficiency_list, loc0_list, loc1_list, halfz_list

    # def add_testing_list(self, layer_index, info_type, start_value, end_value, distribution="Liner"):
    #     pass
    #
    # def del_testing_list(self, layer_index):
    #     pass
    #
    # def get_testing_list_all(self):
    #     pass


class SiLayer:
    def __init__(self, material_budget=-1, radius=-1, half_z=-1, efficiency=-1, loc0=-1,
                 loc1=-1):  # 这里最好还是把loc0，loc1改为全名
        """
        初始化一个layer的信息
        :param material_budget: 散射强度（基本正比于一个层的厚度）
        :param radius: Layer 所在半径位置
        :param half_z: Layer 的半长度
        :param efficiency: 探测效率
        :param loc0: r-phi方向分辨能力
        :param loc1: z 方向分辨能力
        """
        self.layer_param = {
            "Material_budget": material_budget,
            "Radius": radius,
            "Half_z": half_z,
            "Efficiency": efficiency,
            "Loc0": loc0,
            "Loc1": loc1
        }
        # self.Material_budget = material_budget
        # self.Radius = radius
        # self.Half_z = half_z
        # self.Efficiency = efficiency
        # self.Loc0 = loc0
        # self.Loc1 = loc1
        pass

    def __str__(self):
        return f"Layer INFO: \n" \
               f"--------------------------------------------------\n" \
               f"{self.get_info('text')}\n" \
               f"--------------------------------------------------\n"

    def copy(self):
        temp = SiLayer(material_budget=self.layer_param["Material_budget"], radius=self.layer_param["Radius"],
                       half_z=self.layer_param["Half_z"], efficiency=self.layer_param["Efficiency"],
                       loc0=self.layer_param["Loc0"], loc1=self.layer_param["Loc1"])
        # print(temp)
        return temp

    def get_info(self, info_type):
        if info_type == "dir":
            return self.layer_param
        elif info_type == "text":
            return "Material_budget:" + str(round(self.layer_param["Material_budget"], 2)) + \
                   "\t Radius:" + str(round(self.layer_param["Radius"], 2)) + \
                   "\t Half_z:" + str(round(self.layer_param["Half_z"], 2)) + \
                   "\t Efficiency:" + str(round(self.layer_param["Efficiency"], 2)) + \
                   "\t Loc0:" + str(round(self.layer_param["Loc0"], 2)) + "\t Loc1:" + str(
                round(self.layer_param["Loc1"], 2))
        else:
            pass

    def update_layer(self, info_type, value):
        """
        更新某一层的某一个参数
        :param info_type: 参数名称，可选 "Material_budget", "Radius", "Half_z", "Efficiency", "Loc0", "Loc1"
        :param value: 更新的参数值
        :return: 1
        """
        self.layer_param[info_type] = value
        return 1
    #
    # def add_testing_list(self, info_type, start_value, end_value, distribution="Liner"):
    #     pass
    #
    # def del_testing_list(self, info_type):
    #     pass


class XLayer:
    def __init__(self):
        pass


class Environment:
    def __init__(self, B=3, multiple_scattering=True, position_resolution=True):
        """
        :param B: 磁场大小
        :param multiple_scattering: 是否开启多重散射
        :param position_resolution: 位置精度
        """
        self.environment_param = {
            "B": B,
            "multiple_scattering": multiple_scattering,
            "position_resolution": position_resolution
        }
        pass

    def __str__(self):
        return f"Environment INFO: \n" \
               f"--------------------------------------------------\n" \
               f"{self.get_info('text')}\n" \
               f"--------------------------------------------------\n"

    def update_environment(self, type_para, value):
        # 通过访问特定元素来修改环境因素
        self.environment_param[type_para] = value
        pass

    def get_info(self, info_type):
        if info_type == "dir":
            return self.environment_param
        elif info_type == "text":
            return "B:\t" + str(self.environment_param["B"]) + "\nmultiple_scattering:\t" + str(
                self.environment_param["multiple_scattering"]) + "\nposition_resolution:\t" + str(
                self.environment_param["position_resolution"])

        else:
            pass

    def get_param(self):
        return self.environment_param["B"], self.environment_param["multiple_scattering"], self.environment_param[
            "position_resolution"]

    def export(self, path='my_dict.json'):
        with open(path, 'w') as f:
            json.dump(self.environment_param, f, indent=4)

    def load(self, path):
        with open(path) as f:
            self.environment_param = json.load(f)

    # def add_testing_list(self, info_type, start_value, end_value, distribution="Liner"):
    #     pass
    #
    # def del_testing_list(self, info_type):
    #     pass


class Particle:
    def __init__(self, charge=0., mass=-1., beam_spot=-1.):
        """
        :param p: 粒子动量的绝对值
        :param charge: 电荷
        :param mass: 质量（区分种类）
        :param beam_spot: 束焦位置
        """
        self.particle_param = {
            "Mass": mass,
            "Charge": charge,
            "Beam_spot": beam_spot
        }

    def __str__(self):
        return f"Particle INFO: \n" \
               f"--------------------------------------------------\n" \
               f"{self.get_info('text')}\n" \
               f"--------------------------------------------------\n"

    def get_info(self, info_type):
        if info_type == "dir":
            return self.particle_param
        elif info_type == "text":
            return "Mass: " + str(self.particle_param["Mass"]) + \
                   "\t Charge: " + str(self.particle_param["Charge"]) + \
                   "\t Beam_spot: " + str(self.particle_param["Beam_spot"])
        else:
            pass

    def get_param(self):
        return self.particle_param["Mass"], self.particle_param["Charge"], \
               self.particle_param["Beam_spot"]

    def update_particle(self, info_type, value):
        print('\n' + info_type + " has been changed to " + str(value))
        self.particle_param[info_type] = value
        return 1


class Emitter:  # ParticleGun
    def __init__(self):
        """
        :param weight: 每种粒子的发射权重
        :param emit_mode: 每种粒子的发射方式，可选single,自定义（待加）
        """
        self.emit_param = {
            "Weight": [],
            "Particle_classes": [],
            "Emit_mode": []
        }
        self.trial_step = None

    def __str__(self):
        return f"Emitter INFO: \n" \
               f"--------------------------------------------------\n" \
               f"{self.get_info('text')}\n" \
               f"--------------------------------------------------\n"

    def __len__(self):
        return len(self.emit_param["Weight"])

    def export(self, path='emit_dict.json'):

        print(self.emit_param)
        export_dic = {
            "Weight": self.emit_param["Weight"],
            "Particle_classes": {
                "Mass": [particle.get_info("dir")["Mass"] for particle in self.emit_param["Particle_classes"]],
                "Charge": [particle.get_info("dir")["Charge"] for particle in self.emit_param["Particle_classes"]],
                "Beam_spot": [particle.get_info("dir")["Beam_spot"] for particle in
                              self.emit_param["Particle_classes"]],
            },
            "Emit_mode": self.emit_param["Emit_mode"]
        }

        with open(path, 'w') as f:
            json.dump(export_dic, f, indent=4)

    def load(self, path):
        with open(path) as f:
            self.emit_param = json.load(f)
        self.emit_param["Particle_classes"] = [Particle(mass=mass, charge=charge, beam_spot=beam_spot) for
                                               mass, charge, beam_spot in zip(
                self.emit_param["Particle_classes"]["Mass"], self.emit_param["Particle_classes"]["Charge"],
                self.emit_param["Particle_classes"]["Beam_spot"])]

    def copy(self):
        return copy.deepcopy(self)

    def get_info(self, info_type):
        if info_type == "dir":
            return self.emit_param
        elif info_type == "text":
            emitter_info_text = ""

            particles = self.emit_param["Particle_classes"]
            weights = self.emit_param["Weight"]
            emit_mode = self.emit_param["Emit_mode"]

            for idx, particle in enumerate(particles):
                print(round(particle.get_info("dir")["Mass"], 2))
                emitter_info_text += "Idx:" + str(idx) + "\t Mass" + str(
                    particle.get_info("dir")["Mass"]) + "\t Weight: " + str(
                    weights[idx] * 100) + "%\t Emit_mode: " + str(emit_mode[idx]) + "\n"
            return emitter_info_text
        else:
            pass

    def get_param(self):
        particles = self.emit_param["Particle_classes"]
        weights = self.emit_param["Weight"]
        emit_mode = self.emit_param["Emit_mode"]

        selected_category_index = np.random.choice(len(particles), p=weights)
        particle_choosen = particles[selected_category_index]

        emit_mode = emit_mode[selected_category_index]

        param, emit_mode_refresh = self.get_emit_param(emit_mode)
        self.emit_param["Emit_mode"][selected_category_index] = emit_mode_refresh
        # print(phi, theta, beta)
        return particle_choosen, param, selected_category_index

    def get_emit_param(self, emit_mode):
        param = {}
        FLAG_list = ['p', 'theta', 'phi']
        FLAG = np.array([[-1, -1, -1], [-2, -2, -2]])
        for key, value in emit_mode.items():
            if value["type"] == "even":
                max_value = value["maxvalue"]
                min_value = value["minvalue"]
                if key == "theta":
                    # print(key, min_value, max_value)
                    param[key] = math.acos(random.uniform(min_value, max_value))
                    # print(param[key])
                else:
                    param[key] = random.uniform(min_value, max_value)
            elif value["type"] == "fixed":
                if key == "theta":
                    param[key] = math.acos(value["value"])
                else:
                    param[key] = value["value"]
            elif value["type"] == "steps":

                step_list = np.linspace(value["minvalue"], value["maxvalue"], value["steps"])
                count = value["count"] % value["steps"]

                if key == "theta":
                    # print(key, min_value, max_value)
                    param[key] = math.acos(step_list[count])
                    # print(param[key])
                else:
                    param[key] = step_list[count]

                FLAG[0][FLAG_list.index(key)] = count
                FLAG[1][FLAG_list.index(key)] = value["steps"] - 1


            else:
                raise ValueError("Invalid emit_mode. Use 'even' or 'fixed'.")

        F = FLAG[1, :] - FLAG[0, :]
        F_inv = F[::-1]
        idx = np.argmax(F_inv > 0) if np.any(F_inv > 0) else False
        idx = 2 - idx if idx is not False else 0
        for i in range(idx, 3):
            if F[i] >= 0:
                emit_mode[FLAG_list[i]]["count"] = (emit_mode[FLAG_list[i]]["count"] + 1) % emit_mode[FLAG_list[i]][
                    "steps"]

        # print(F)
        return param, emit_mode

    def add_particle(self, particle, weight=0., emit_mode=default_mode):
        self.emit_param["Particle_classes"].append(particle)
        self.emit_param["Weight"].append(weight)
        self.emit_param["Emit_mode"].append(emit_mode)
        pass

    def update_particle(self, particle_idx, para_name, value):
        if para_name == "Weight":
            self.emit_param[para_name][particle_idx] = value
            return 1

        particle = self.emit_param["Particle_classes"][particle_idx]
        particle.update_particle(para_name, value)
        return 1

    def update_emit_param(self, par_idx, emit_mode):
        self.emit_param["Emit_mode"][par_idx] = emit_mode

    def delete_particle(self, particle_index):
        pass


class Experiment:
    def __init__(self, detector=None, particle=None, environment=None):
        """
        由 Particle, Environment, Detector 组成
        """
        self.detector = detector
        self.particle = particle
        self.environment = environment
        pass

    def __str__(self):
        return f"Experiment INFO: \n" \
               f"BEGIN ----------------------------------------------------------------------------------------\n\n" \
               f"Detector------------------------------------------\n" \
               f"{self.detector.get_info('text')}\n" \
               f"Particle------------------------------------------\n" \
               f"{self.particle.get_info('text')}\n" \
               f"Environment---------------------------------------\n" \
               f"{self.environment.get_info('text')}\n\n" \
               f"END ------------------------------------------------------------------------------------------\n"

    def get_info(self, info_type):
        if info_type == "text":
            str1 = self.detector.get_info('text') + '\n'
            str2 = self.particle.get_info('text') + '\n\n'
            str3 = self.environment.get_info('text')

            str_all = str1 + str2 + str3
            return str_all
        else:
            return False

    def get_testing_list(self):
        pass

    def generate_testing_param(self):
        pass

    def kalman_test(self):
        pass

    def analysis_test(self):
        pass


class Result:
    def __init__(self, test_num=None):
        if test_num is None:
            print(
                "\033[1;33mPlease input the test number When create a Result object. or load the data from file.\033[0m")
            # warnings.warn("")
        self.result = {}
        self.initial = {}
        self.test_num = test_num
        self.count = 0

    def __len__(self):
        return len(self.result)

    def __str__(self):
        return str(self.initial) + "\n" + str(self.result)

    def set_emit_mode(self, emit_mode):
        self.emit_mode = emit_mode

    def append(self, ini_para, result_para):

        for key, value in ini_para.items():
            value = np.array(value)
            try:
                self.initial[key][self.count] = value
            except Exception as e:
                self.initial[key] = np.inf * np.ones((self.test_num, *value.shape))
                self.initial[key][0] = value

        for key, value in result_para.items():
            value = np.array(value)
            try:
                self.result[key][self.count] = value
            except Exception as e:
                # print(key)
                self.result[key] = np.inf * np.ones((self.test_num, *value.shape))
                self.result[key][0] = value

        self.count += 1
        pass

    def analytic_plot(self, x_axis=None, y_axis=None, z_axis=None, colormap=None, emit_mode=None):
        ax = plt.gca()
        x_data = self.initial[x_axis]

        keys = self.judge_mode(emit_mode) if emit_mode else []

        try:
            y_data = self.initial[y_axis]
        except Exception as e:
            y_data = self.result[y_axis]

        # fig = plt.figure()
        if x_axis in ["theta", "phi"]:
            x_data = np.degrees(x_data)

        if y_axis in ["theta", "phi"]:
            y_data = np.degrees(y_data)

        if colormap:
            try:
                c_data = self.initial[colormap]
            except Exception as e:
                c_data = self.result[colormap]
        else:
            c_data = None

        if len(keys) > 1:
            key = [x for x in keys if x != x_axis][0]
            key_step = np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"],
                                   emit_mode[key]["steps"])
            if key == "theta" or key == "phi":
                key_step = np.arccos(key_step)

            for step in key_step:
                index = np.array(
                    [index for index, value in enumerate(self.initial[key]) if
                     np.abs(value - step) < 1e-10])

                sorted = np.argsort(x_data[index])
                index = index[sorted]

                ax.plot(np.array(x_data)[index], np.array(y_data)[index], '.-')
            if key == "theta":
                plt.legend(
                    [X_UNIT_LABEL[key] + "$ = " + str(np.round(np.degrees(s0), 3)) + "$" for s0
                     in key_step], fontsize=8)
            else:
                plt.legend([X_UNIT_LABEL[key] + "$ = " + str(np.round(s0, 3)) + "$" for s0 in
                            key_step], fontsize=8)

        elif len(keys) == 1:
            if x_axis in keys:
                index = np.argsort(x_data)

                ax.plot(np.array(x_data)[index], np.array(y_data)[index], '.-')



        else:

            ax.grid(True, linestyle='-.', linewidth=0.5, color='black', alpha=0.2)  # Add denser grid lines
            # plt.plot(x_data, y_data, '-', c='black')
            ax.scatter(x_data, y_data, marker='o', label='Result Plot', c=c_data, cmap='jet', s=20)

        try:
            x_unit = unit_dict[x_axis]
        except Exception as e:
            x_unit = "mm"

        # try:
        #     y_unit = unit_dict[y_axis]
        # except Exception as e:
        #     y_unit = "mm"

        plt.xlabel(X_UNIT_LABEL[x_axis], fontweight='bold', usetex=True, fontsize=10)
        # plt.ylabel(y_axis + " (" + Y_UNIT[y_axis] + ")")
        plt.ylabel(Y_UNIT_LABEL[y_axis], fontweight='bold', usetex=True, fontsize=10)
        # ax.xticks(np.round(np.linspace(np.min(x_data) - 0.2 * (np.max(x_data) - np.min(x_data)),
        #                                 np.max(x_data) + 0.2 * (np.max(x_data) - np.min(x_data)), 6, endpoint=True), 2))
        # plt.yticks(np.round(np.linspace(np.min(y_data) - 1.2 * (np.max(y_data) - np.min(y_data)),
        #                                 np.max(y_data) + 1.2 * (np.max(y_data) - np.min(y_data)), 6, endpoint=True), 2))

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

        # if c_data is not None:
        #     plt.colorbar()

        # plt.legend()

        # ax.scatter(x_data, y_data, '-')
        # return fig

        pass

    def kalman_plot(self, x_axis=None, y_axis=None, z_axis=None, colormap=None, emit_mode=None, p_step=None,
                    theta_step=None, phi_step=None, layer_idx=None, filter=None):
        ax = plt.gca()
        dic = {
            "p": p_step,
            "theta": theta_step,
            "phi": phi_step
        }

        if x_axis == None or y_axis == None:
            print("none selected")
            return -1

        if colormap:
            try:
                c_data = self.initial[colormap]
            except Exception as e:
                c_data = self.result[colormap]
        else:
            c_data = None

        keys = self.judge_mode(emit_mode)
        selected_step = 3 - [p_step, theta_step, phi_step].count(None)
        if layer_idx and filter:
            if keys:
                if len(keys) == selected_step:

                    key_value = []
                    for key in keys:
                        if key == "theta":
                            key_value.append(math.acos(
                                np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"],
                                            emit_mode[key]["steps"])[dic[key] - 1]))
                        else:
                            key_value.append(np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"],
                                                         emit_mode[key]["steps"])[dic[key] - 1])

                    # key_value = [np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"], emit_mode[key]["steps"])[dic[key]-1] for key in keys]

                    # arrays = [self.initial[key] for key in keys]

                    for i in range(len(self.initial["theta"])):
                        if [self.initial[key][i] for key in keys] == key_value:
                            jump = i
                            break

                    key_value = []
                    for key in keys:
                        if key == "theta":
                            key_value.append(np.arccos(
                                np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"],
                                            emit_mode[key]["steps"])).tolist())
                        else:
                            key_value.append(np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"],
                                                         emit_mode[key]["steps"]).tolist())
                    combinations = [list(comb) for comb in product(*[key_value[idx] for idx, key in enumerate(keys)])]

                    result = self.result[filter + "_" + y_axis][:, layer_idx - 1][jump::len(combinations)]
                    plt.hist(result, bins=50, histtype='step', edgecolor='blue')

                    title_0 = ""
                    for key in keys:
                        if dic[key] != None:
                            if key == "theta":
                                title_0 = title_0 + X_UNIT_LABEL[key] + r"$\mathbf{ = " + str(
                                    np.round(np.degrees(key_value[keys.index(key)][dic[key] - 1]), 3)) + "}$\ \ "
                            else:
                                title_0 = title_0 + X_UNIT_LABEL[key] + r"$\mathbf{ = " + str(
                                    np.round(key_value[keys.index(key)][dic[key] - 1], 3)) + "}$\ \ "
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
                    plt.title(title_0, usetex=True, fontsize=15)
                    plt.xlabel(Y_UNIT_LABEL[y_axis], fontweight='bold', usetex=True, fontsize=15)
                    plt.ylabel("Events Number", fontweight='bold', usetex=False, fontsize=15)



                else:
                    if self.post_initial:

                        if len(keys) == 1:
                            try:
                                x_data = self.post_initial[x_axis]
                                if x_axis == "theta":
                                    x_data = np.degrees(x_data)

                                plt.xlabel(X_UNIT_LABEL[x_axis], fontweight='bold', usetex=True, fontsize=15)
                                # plt.ylabel(y_axis + " (" + Y_UNIT[y_axis] + ")")
                                plt.ylabel(Y_UNIT_LABEL[y_axis], fontweight='bold', usetex=True, fontsize=15)

                                y_data = np.array(self.post_result[filter + "_" + y_axis])[:, layer_idx - 1]
                                ax.plot(x_data, y_data, '.-')
                                ax.minorticks_on()
                                # 设置主刻度线的样式
                                ax.tick_params(which='major', length=5, width=1.5, direction='in', right=True, top=True)
                                # 设置小刻度线的样式
                                ax.tick_params(which='minor', length=3, width=1, direction='in', right=True, top=True)
                                for spine in ax.spines.values():
                                    spine.set_linewidth(1.5)
                                # plt.scatter(x_data, y_data, marker='o', label='Result Plot', c=c_data, cmap='jet', s=20)
                                plt.grid(True, linestyle='-.', linewidth=0.5, color='black',
                                         alpha=0.2)  # Add denser grid lines

                            except Exception as e:
                                pass

                        if len(keys) == 2:
                            key = [x for x in keys if x != x_axis][0]
                            key_step = np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"],
                                                   emit_mode[key]["steps"])
                            if key == "theta":
                                key_step = np.arccos(key_step)
                            if selected_step == 0:
                                try:
                                    for key_value in key_step:
                                        filtered_indices = np.array(
                                            [index for index, value in enumerate(self.post_initial[key]) if
                                             np.abs(value - key_value) < 1e-10])
                                        # print(filtered_indices)

                                        x_data = np.array(self.post_initial[x_axis])
                                        if x_axis == "theta":
                                            x_data = np.degrees(x_data)

                                        plt.xlabel(X_UNIT_LABEL[x_axis], fontweight='bold', usetex=True, fontsize=15)
                                        # plt.ylabel(y_axis + " (" + Y_UNIT[y_axis] + ")")
                                        plt.ylabel(Y_UNIT_LABEL[y_axis], fontweight='bold', usetex=True, fontsize=15)

                                        y_data = np.array(self.post_result[filter + "_" + y_axis])[:, layer_idx - 1]
                                        ax.plot(x_data[filtered_indices], y_data[filtered_indices], '.-')
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

                                    if key == "theta":
                                        plt.legend(
                                            [X_UNIT_LABEL[key] + "$ = " + str(np.round(np.degrees(s0), 3)) + "$" for s0
                                             in key_step], fontsize=8)
                                    else:
                                        plt.legend([X_UNIT_LABEL[key] + "$ = " + str(np.round(s0, 3)) + "$" for s0 in
                                                    key_step], fontsize=8)

                                except Exception as e:
                                    print(e)
                                    pass

                            elif selected_step == 1:
                                # print(dic[key])
                                if dic[key] != None:
                                    filtered_indices = np.array(
                                        [index for index, value in enumerate(self.post_initial[key]) if
                                         np.abs(value - key_step[dic[key] - 1]) < 1e-10])
                                    # print(filtered_indices)

                                    x_data = np.array(self.post_initial[x_axis])
                                    if x_axis == "theta":
                                        x_data = np.degrees(x_data)

                                    plt.xlabel(X_UNIT_LABEL[x_axis], fontweight='bold', usetex=True, fontsize=15)
                                    # plt.ylabel(y_axis + " (" + Y_UNIT[y_axis] + ")")
                                    plt.ylabel(Y_UNIT_LABEL[y_axis], fontweight='bold', usetex=True, fontsize=15)

                                    y_data = np.array(self.post_result[filter + "_" + y_axis])[:, layer_idx - 1]
                                    ax.plot(x_data[filtered_indices], y_data[filtered_indices], '.-')
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

                                    if key == "theta":
                                        plt.legend(
                                            [X_UNIT_LABEL[key] + "$ = " + str(
                                                np.round(np.degrees(key_step[dic[key] - 1]), 3)) + "$"],
                                            fontsize=8)
                                    else:
                                        plt.legend(
                                            [X_UNIT_LABEL[key] + "$ = " + str(
                                                np.round(key_step[dic[key] - 1], 3)) + "$"],
                                            fontsize=8)

                    pass

            else:
                x_data = np.array(self.initial[x_axis])
                if x_axis == "theta":
                    x_data = np.degrees(x_data)
                y_data = np.array(self.result[filter + "_" + y_axis])[:, layer_idx - 1]
                plt.scatter(x_data, y_data, marker='o', label='Result Plot', c=c_data, cmap='jet', s=20)
                plt.grid(True, linestyle='-.', linewidth=0.5, color='black', alpha=0.2)  # Add denser grid lines

    def judge_mode(self, emit_mode):
        keys = []
        for key, value in emit_mode.items():
            if emit_mode[key]["type"] == "steps":
                keys.append(key)
        if keys:
            return keys
        else:
            return False

    def kalman_post_process(self, emit_mode, layer_num):
        self.delete_inf()
        keys = self.judge_mode(emit_mode)
        if keys and self.initial and self.result:
            self.post_initial = {}
            self.post_result = {}
            key_value = []
            for key in keys:
                if key == "theta":
                    key_value.append(np.arccos(
                        np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"],
                                    emit_mode[key]["steps"])).tolist())
                else:
                    key_value.append(np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"],
                                                 emit_mode[key]["steps"]).tolist())

            # key_value = [np.linspace(emit_mode[key]["minvalue"], emit_mode[key]["maxvalue"], emit_mode[key]["steps"])[dic[key]-1] for key in keys]

            combinations = [list(comb) for comb in product(*[key_value[idx] for idx, key in enumerate(keys)])]
            keys = [key for idx, key in enumerate(keys)]

            for idx, comb in enumerate(combinations):
                # print((idx+1) / len(combinations))
                yield (idx + 1) / len(combinations) * 100
                for idx, key in enumerate(keys):
                    try:
                        self.post_initial[key].append(comb[idx])
                    except Exception as e:
                        self.post_initial[key] = [comb[idx]]

                for i in range(len(self.initial["theta"])):
                    if [self.initial[key][i] for key in keys] == comb:
                        jump = i
                        break
                # t = time.time()
                try:
                    for filt in FILTER_TYPE_KALMAN:
                        for y_axis in Y_AXIS_TYPE_KALMAN:
                            temp = []
                            # t = time.time()
                            for layer in range(layer_num):
                                # t = time.time()
                                result = self.result[filt + "_" + y_axis][:, layer][jump::len(combinations)]
                                # print(self.initial["p"][jump::len(combinations)])
                                # print(time.time() - t)
                                params3 = list(stats.norm.fit(result))
                                temp.append(params3[1])
                                # rms = np.sqrt(np.mean(result ** 2))
                                # temp.append(rms)

                                # sigma = np.std(result)
                                # temp.append(result)
                            try:
                                self.post_result[filt + "_" + y_axis].append(temp)
                            except Exception as e:
                                self.post_result[filt + "_" + y_axis] = [temp]
                except Exception as e:
                    print(e)
                    pass
                # print(time.time() - t)
            # scipy.io.savemat(r"cov_dic\py\result.mat", self.post_result)
            # scipy.io.savemat(r"cov_dic\py\initial.mat", self.post_initial)

    def kalman_post_process_all(self, emit_mode, layer_num):
        for i in self.kalman_post_process(emit_mode, layer_num):
            pass

    def get(self, key):
        try:
            try:
                return self.result[key]
            except Exception as e:
                return self.initial[key]

        except Exception as e:
            print(e)
            return []

    def delete_inf(self):
        for key, value in self.result.items():
            self.result[key] = value[0:self.count]
        for key, value in self.initial.items():
            self.initial[key] = value[0:self.count]
        pass

    def load_root(self, file_path):
        if not os.path.exists(file_path):
            return False
        with uproot.open(file_path) as f:

            tree = f.get("analytic") or f.get("kalman") or f.get("kalman_post")
            if tree is None:
                return False
            tree_type = "analytic" if tree == f.get("analytic") else "kalman" if tree == f.get(
                "kalman") else "kalman_post"

            print(tree_type)

            if tree_type == "analytic":
                self.read_analytic(tree)
            elif tree_type == "kalman" or tree_type == "kalman_post":
                if tree_type == "kalman":
                    file_path2 = file_path.replace(".", "_kalman_post.")
                    if not os.path.exists(file_path2):
                        return False
                    with uproot.open(file_path2) as f2:
                        tree2 = f2.get("kalman_post")
                        if tree2 is None:
                            return False
                        self.read_kalman(tree, tree2)
                else:
                    file_path2 = file_path.replace("_kalman_post", "")
                    if not os.path.exists(file_path2):
                        return False
                    with uproot.open(file_path2) as f2:
                        tree2 = f2.get("kalman")
                        if tree2 is None:
                            return False
                        self.read_kalman(tree2, tree)

        return tree_type

    def read_analytic(self, tree):
        self.count = tree[tree.keys()[0]].array(library="np").shape[0]
        self.test_num = tree[tree.keys()[0]].array(library="np").shape[0]
        for key in tree.keys():
            if key in INITIAL_TYPE:
                self.initial[key] = tree[key].array(library="np")
            elif key in A_RESULT_TYPE:
                self.result[key] = tree[key].array(library="np")

    def read_kalman(self, tree1, tree2=None):
        self.count = tree1[tree1.keys()[0]].array(library="np").shape[0]
        self.test_num = tree1[tree1.keys()[0]].array(library="np").shape[0]


        for key in tree1.keys():
            if key in INITIAL_TYPE:
                self.initial[key] = tree1[key].array(library="np")
            elif key in K_RESULT_TYPE:
                self.result[key] = tree1[key].array(library="np")
        if tree2 is not None:
            self.post_initial = {}
            self.post_result = {}
            for key in tree2.keys():
                if key in INITIAL_TYPE:
                    self.post_initial[key] = tree2[key].array(library="np")
                elif key in K_POST_RESULT_TYPE:
                    self.post_result[key] = tree2[key].array(library="np")

        pass
if __name__ == "__main__":
    # t = time.time()
    # dec = Detector()
    # # print(dec)
    #
    # for i in range(10, 0, -1):
    #     dec.add_layer(SiLayer(radius=i ** 2))
    #
    # dec.get_param()

    # envir = Environment()
    # # print(envir)
    # # envir.update_environment("B", 1)
    # # envir.export()
    # envir.load(r"D:\files\pyproj\decPip\DDFS\my_dict.json")
    # print(envir)

    # # print(envir)
    #
    # pa = Particle()
    # print(pa)

    emit = Emitter()
    print(emit)
    # emit.add_particle(pa, 1 / 4, default_mode)
    # emit.add_particle(pa, 1 / 4, default_mode)
    # emit.add_particle(pa, 1 / 4, default_mode)
    # emit.add_particle(pa, 1 / 4, default_mode)
    # print(emit)

    emit.load(r"D:\files\pyproj\decPip\DDFS\emit_dict.json")
    print(emit)

    #
    # exp = Experiment(dec, pa, envir)

    # print(exp)
    # print(asizeof.asizeof(exp))
