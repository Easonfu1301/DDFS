import sys
import time

import numpy as np
# import cupy as np
import math
import DDFS.common_test_parameter2 as ctp
import matplotlib.pyplot as plt
import matplotlib
from filterpy.kalman import KalmanFilter
from DDFS.cal_dermat import calculate_dermat
from scipy.optimize import curve_fit
import scipy
from scipy import stats
from scipy.stats import norm
from tqdm import tqdm
from scipy.stats import chi2
from DDFS.element import Result

# matplotlib.use('TkAgg')
np.set_printoptions(precision=4)
np.set_printoptions(suppress=False)

c = 0.299792458e-3


def fy(y):
    if y < 1e-6:
        return 0.0
    else:
        return 0.0136 * y ** 0.5 * (1.0 + 0.038 * np.log(y))


def get_circle_para(p1, p2, p3):
    '''
        三点求圆，返回圆心和半径
    '''
    x, y, z = p1[0] + p1[1] * 1j, p2[0] + p2[1] * 1j, p3[0] + p3[1] * 1j
    w = z - x
    w /= y - x
    c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
    return (-c.real, -c.imag), abs(c + x)


def spiral_func(r_phi, x0, y0, z0, r0, tan_theta0):
    phi = r_phi[:, 0]
    radius = r_phi[:, 1]
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    Phi = np.arctan2(y - y0, x - x0)

    z = z0 + r0 * Phi / tan_theta0
    x = x0 + r0 * np.cos(Phi)
    y = y0 + r0 * np.sin(Phi)

    return np.array([x, y, z]).T.flatten()


class Resolution:
    def __init__(self, dec_info_list, environment_info_list, emitter):
        self.emitter = emitter
        self.radius_list, self.budget_list, self.efficiency_list, self.loc0_list, self.loc1_list, _ = dec_info_list
        self.B, mutiple_scattering_FLAG, position_resolution_FLAG = environment_info_list
        self.MS = mutiple_scattering_FLAG
        self.RE = position_resolution_FLAG
        # self.radius_list = [i/1000 for i in self.radius_list]
        self.get_param()
        self.first_layer_idx = np.argmax(np.array(self.efficiency_list) > 0)

        # self.alpha = np.float64(1000000. / c / self.B)  # magnetic-field-constant 1/(c*B)

    def get_param(self):
        self.par, ini_params, self.index_par = self.emitter.get_param()

        # print(ini_params)
        self.p = ini_params["p"]
        self.theta = ini_params["theta"]
        self.phi = ini_params["phi"]

        # print(self.p)

        self.particle_info_list = self.par.get_param()

        self.mass, self.charge, self.beam_spot = self.particle_info_list

    def generate_observe_value(self, param_helix, layer_now_idx):
        # layer_now_idx += 1
        phi_helix, z_helix, theta_helix, beta_helix, kappa_helix = param_helix

        observed_param = math.inf
        observed_XYZ = math.inf

        if z_helix > 3000:
            # raise a warning with yellow words
            print("\033[1;33mWarning: z_helix is too large!\033[0m")

        if layer_now_idx >= 0:
            if self.efficiency_list[layer_now_idx] > 0:
                sigma_loc0 = self.loc0_list[layer_now_idx]
                sigma_loc1 = self.loc1_list[layer_now_idx]

                ran_loc0 = np.random.normal(0, sigma_loc0)
                ran_loc1 = np.random.normal(0, sigma_loc1)

                phi_helix = phi_helix + ran_loc0 / self.radius_list[layer_now_idx]
                z_helix = z_helix + ran_loc1

                observed_param = [phi_helix, z_helix]
                observed_X = self.radius_list[layer_now_idx] * math.cos(phi_helix)
                observed_Y = self.radius_list[layer_now_idx] * math.sin(phi_helix)
                observed_Z = z_helix

                observed_XYZ = [observed_X, observed_Y, observed_Z]

        return observed_param, observed_XYZ

    def para_perturbation(self, param, layer_next_idx):
        x, y, z, theta, phi, kappa = param
        sigma = np.nan
        material_budget = self.budget_list[layer_next_idx]

        if material_budget < 9:
            pT = c * self.B / kappa
            # print(math.sin(bparami[2]))
            p = pT / math.sin(theta)

            ag = math.atan2(y, x) - phi

            X = material_budget / math.fabs(math.sin(theta) * math.cos(ag))

            # if X<0:
            #     sigma = 1e-10
            #     print("wrong")
            # else:

            sigma = 0.0136 * \
                    math.sqrt((self.mass ** 2 + p ** 2) / p ** 4) * \
                    math.sqrt(X) * (1 + 0.038 * math.log(X))

            ran = np.random.normal(0, sigma, 2) * [1, 1 / math.sin(theta)]

            theta_new = theta + ran[0]
            phi_new = phi + ran[1]

            # 保证微扰后phi永远在[-pi, pi]之间
            if phi_new > np.pi:
                phi_new -= 2 * np.pi
            elif phi_new < -np.pi:
                phi_new += 2 * np.pi

            kappa_new = kappa * math.sin(theta) / math.sin(theta_new)

            param = [x, y, z, theta_new, phi_new, kappa_new]


            # print(kappa, kappa_new)

        return param, sigma ** 2

    def update_circle_eqn(self, param):
        x, y, z, theta, phi, kappa = param
        R_track = 1 / kappa

        if self.charge > 0:

            phi_circle = phi + np.pi / 2

            x_h = x + R_track * math.cos(phi_circle)
            y_h = y + R_track * math.sin(phi_circle)
        else:

            phi_circle = phi - np.pi / 2

            x_h = x + R_track * math.cos(phi_circle)
            y_h = y + R_track * math.sin(phi_circle)

        return R_track, x_h, y_h

    def cal_next_layer_position(self, param, R_trk, x_h, y_h, radius_next):
        x, y, z, theta, phi, kappa = param

        tmp1 = -radius_next ** 2 + 2 * radius_next * R_trk - R_trk ** 2 + x_h ** 2 + y_h ** 2  # 见推导
        tmp2 = radius_next ** 2 + 2 * radius_next * R_trk + R_trk ** 2 - x_h ** 2 - y_h ** 2

        if tmp1 * tmp2 < 0:
            # plt.pause(100)
            # time.sleep(5)
            raise ValueError("the particle cannot reach next layer")

        tmp3 = radius_next ** 2 * y_h - R_trk ** 2 * y_h + x_h ** 2 * y_h + y_h ** 3

        tmp4 = x_h * math.sqrt(tmp1 * tmp2)

        y_1 = (tmp3 + tmp4) / (2 * (x_h ** 2 + y_h ** 2))
        y_2 = (tmp3 - tmp4) / (2 * (x_h ** 2 + y_h ** 2))

        y_solved = [y_1, y_2]

        x_1 = (radius_next ** 2 - R_trk ** 2 + x_h ** 2 + y_h ** 2 - 2 * y_1 * y_h) / (2 * x_h)
        x_2 = (radius_next ** 2 - R_trk ** 2 + x_h ** 2 + y_h ** 2 - 2 * y_2 * y_h) / (2 * x_h)

        x_solved = [x_1, x_2]

        phi_1 = math.atan2(y_h - y_1, x_h - x_1)
        phi_2 = math.atan2(y_h - y_2, x_h - x_2)

        phi_solved = [phi_1, phi_2]

        if self.charge > 0:
            x_next = x_solved[0]
            y_next = y_solved[0]
            phi_next = phi_solved[0] - np.pi / 2

            z_next = z - R_trk * np.fmod(phi_next - phi - 2 * np.pi, 2 * np.pi) / math.tan(theta)
            theta_next = theta
            kappa_next = kappa
        else:
            x_next = x_solved[1]
            y_next = y_solved[1]
            phi_next = phi_solved[1] + np.pi / 2  # ?我不理解，是否出现在phi的atan上？

            z_next = z + R_trk * np.fmod(phi_next - phi + 2 * np.pi, 2 * np.pi) / math.tan(theta)
            theta_next = theta
            kappa_next = kappa

        param = [x_next, y_next, z_next, theta_next, phi_next, kappa_next]

        return param

    def xyz_tpk2helix(self, param):
        x, y, z, theta, phi, kappa = param

        phi_helix = math.atan2(y, x)

        # if layer_now_idx >= 0:
        #     if self.charge < 0 and phi_helix < self.param_helix_store[layer_now_idx - 1, 0]:
        #         phi_helix += 2 * np.pi
        #
        #     if self.charge > 0 and phi_helix > self.param_helix_store[layer_now_idx - 1, 0]:
        #         phi_helix -= 2 * np.pi

        z_helix = z
        theta_helix = theta
        if self.charge > 0:
            beta_helix = math.fmod(phi - phi_helix - 1 * math.pi, 2 * math.pi)
        else:
            beta_helix = math.fmod(phi - phi_helix + 1 * math.pi, 2 * math.pi)

        kappa_helix = kappa

        return [phi_helix, z_helix, theta_helix, beta_helix, kappa_helix]

    def helix2xyz_tpk(self, param, radius_now):
        phi_helix, z_helix, theta_helix, beta_helix, kappa_helix = param

        x = radius_now * math.cos(phi_helix)
        y = radius_now * math.sin(phi_helix)

        z = z_helix
        theta = theta_helix
        phi = beta_helix - phi_helix
        kappa = kappa_helix

        return [x, y, z, theta, phi, kappa]

    def generate_path(self):
        self.get_param()
        # np.random.seed(0)

        # t = time.time()

        self.var_store = np.zeros((len(self.radius_list), 1))

        self.param_store = np.zeros((len(self.radius_list), 6))
        self.observe_store = np.zeros((len(self.radius_list), 2))
        self.param_helix_store = np.zeros((len(self.radius_list), 5))
        self.observe_store_XYZ = np.zeros((len(self.radius_list), 3))

        self.cov_list = []
        self.A = []

        x = 0
        y = 0
        z = 0

        # kappa = c * self.B / (self.p * math.sin(self.theta))
        kappa = c * self.B / self.p  # 就是要设定pT

        self.param_start = [x, y, z, self.theta, self.phi, kappa]  # z = 0

        param = self.param_start

        for layer_now_idx in range(0, len(self.radius_list)):

            radius_now = self.radius_list[layer_now_idx - 1] if layer_now_idx > 0 else 0
            radius_next = self.radius_list[layer_now_idx]

            R_trk, x_h, y_h = self.update_circle_eqn(param)

            param = self.cal_next_layer_position(param, R_trk, x_h, y_h, radius_next)

            if self.MS:
                param, var = self.para_perturbation(param, layer_now_idx)
                self.var_store[layer_now_idx] = var


            if layer_now_idx == 1:
                self.MS_start = param

            data_helix = self.xyz_tpk2helix(param)
            data_observed, observed_XYZ = self.generate_observe_value(data_helix, layer_now_idx)


            self.param_store[layer_now_idx, :] = param
            self.param_helix_store[layer_now_idx, :] = data_helix
            self.observe_store[layer_now_idx, :] = data_observed
            self.observe_store_XYZ[layer_now_idx, :] = observed_XYZ

        # self.PLOT()

        return self.param_store, self.observe_store_XYZ
        # plt.pause(100)

    def generate_ref_path(self):
        self.var_store_ref = np.zeros((len(self.radius_list), 1))

        # self.param_store_ref = np.zeros((len(self.radius_list), 6))
        # self.observe_store_ref = np.zeros((len(self.radius_list), 2))
        self.param_helix_store_ref = np.zeros((len(self.radius_list), 5))
        self.demat_store_ref = [None for i in range(len(self.radius_list))]
        # self.observe_store_XYZ_ref = np.zeros((len(self.radius_list), 3))

        self.cov_list_ref = []
        self.A_ref = []

        x = 0
        y = 0
        z = 0

        # kappa = c * self.B / (self.p * math.sin(self.theta))
        kappa = c * self.B / self.p  # 就是要设定pT

        self.param_start_ref = [x, y, z, self.theta, self.phi, kappa]  # z = 0

        param_ref = self.param_start_ref
        # param_ref = self.MS_start

        for layer_now_idx in range(0, len(self.radius_list)):

            radius_now = self.radius_list[layer_now_idx - 1] if layer_now_idx > 0 else 0
            radius_next = self.radius_list[layer_now_idx]


            R_trk, x_h, y_h = self.update_circle_eqn(param_ref)




            param_ref = self.cal_next_layer_position(param_ref, R_trk, x_h, y_h, radius_next)

            if self.MS:
                _, var_ref = self.para_perturbation(param_ref, layer_now_idx)# ref 不更新param

                self.var_store_ref[layer_now_idx] = var_ref


            if layer_now_idx == 1:
                # print(1)
                param_ref = self.MS_start


            data_helix_ref = self.xyz_tpk2helix(param_ref)
            # data_observed_ref, observed_XYZ_ref = self.generate_observe_value(data_helix_ref, layer_now_idx)

            # self.param_store_ref[layer_now_idx, :] = param_ref
            self.param_helix_store_ref[layer_now_idx, :] = data_helix_ref
            # self.observe_store_ref[layer_now_idx, :] = data_observed_ref
            # self.observe_store_XYZ_ref[layer_now_idx, :] = observed_XYZ_ref

        # self.PLOT()

        return self.param_store, self.observe_store_XYZ



    def direct_fit(self):
        phi_array = self.observe_store[:, 0]
        for i in range(1, len(phi_array)):
            if phi_array[i] - phi_array[i - 1] < -np.pi * 1.5:
                phi_array[i:] += 2 * np.pi
        x_arr = self.radius_list * np.cos(phi_array)
        y_arr = self.radius_list * np.sin(phi_array)
        z_arr = self.observe_store[:, 1]
        # fig = plt.figure()
        # ax00 = fig.add_subplot(111, projection="3d")
        # plt.plot(x_arr, y_arr, z_arr, 'o')
        # # plt.show()

        R_trk, x_h, y_h = self.update_circle_eqn(self.param_start)

        popt1, pcov1 = curve_fit(spiral_func, np.array([phi_array[1:], self.radius_list[1:]]).T,
                                 np.array([x_arr[1:], y_arr[1:], z_arr[1:]]).T.flatten(),
                                 p0=[x_h, y_h, 0, R_trk, math.tan(self.param_start[3])])
        # print(popt1)

        re = spiral_func(np.array([phi_array[:], self.radius_list[:]]).T, *popt1).reshape(-1, 3)

        # ax00.plot(re[:, 0], re[:, 1], re[:, 2], 'o')
        re = np.append(re, math.atan(popt1[4]) * np.ones((re.shape[0], 1)), axis=1)

        phi = phi_array
        radius = self.radius_list
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        Phi = np.arctan2(y - popt1[1], x - popt1[0])
        Phi = Phi + np.pi / 2 * self.charge

        re = np.append(re, np.array([Phi]).T, axis=1)

        re = np.append(re, 1 / popt1[3] * np.ones((re.shape[0], 1)), axis=1)

        self.direct_fit_store_helix = np.full((re.shape[0], 5), np.nan)

        for i in range(1, re.shape[0]):
            self.direct_fit_store_helix[i, :] = self.xyz_tpk2helix(re[i, :])

        # plt.show()

        pass

    def ini_fig(self):
        self.fig1 = plt.figure(1)
        self.fig2 = plt.figure(2)
        self.f1ax1 = self.fig1.add_subplot(111, projection='3d')

        # generate 2x3 subfigure for fig2 named f2ax n

        self.f2ax1 = self.fig2.add_subplot(231)
        self.f2ax2 = self.fig2.add_subplot(232)
        self.f2ax3 = self.fig2.add_subplot(233)
        self.f2ax4 = self.fig2.add_subplot(234)
        self.f2ax5 = self.fig2.add_subplot(235)

    def PLOT(self):
        self.f1ax1.plot(self.param_store[:, 0], self.param_store[:, 1], self.param_store[:, 2], '-',
                        color='b')

        self.f1ax1.plot(self.observe_store_XYZ[:, 0], self.observe_store_XYZ[:, 1], self.observe_store_XYZ[:, 2], '-',
                        color='r')
        # set f1x1 axis equal and from -1000 to 1000
        self.f1ax1.set_xlim(-2000, 2000)
        self.f1ax1.set_ylim(-2000, 2000)
        # self.f1ax1.set_zlim(-100, 100)
        # self.f1ax1.set_aspect('equal')

        self.f2ax1.plot(self.radius_list, self.param_helix_store[:, 0] + np.pi, '-', color='b')
        self.f2ax2.plot(self.radius_list, self.param_helix_store[:, 1], '-', color='b')
        self.f2ax3.plot(self.radius_list, self.param_helix_store[:, 2], '-', color='b')
        self.f2ax4.plot(self.radius_list, self.param_helix_store[:, 3], '-', color='b')
        self.f2ax5.plot(self.radius_list, c * self.B / self.param_helix_store[:, 4], '-', color='b')

        self.f1ax1.plot(self.radius_list * np.cos(self.kalman_fit_store_forward[:, 0]),
                        self.radius_list * np.sin(self.kalman_fit_store_forward[:, 0]),
                        self.kalman_fit_store_forward[:, 1], '-',
                        color='g')

        self.f1ax1.plot(self.radius_list * np.cos(self.kalman_fit_store_backward[:, 0]),
                        self.radius_list * np.sin(self.kalman_fit_store_backward[:, 0]),
                        self.kalman_fit_store_backward[:, 1], '-',
                        color='y')

    def get_transform_matrix(self, param_now, layer_index=None):
        r_now = self.radius_list[layer_index]
        r_next = self.radius_list[layer_index + 1]

        transform_matrix = calculate_dermat(param_now, r_now, r_next)
        return transform_matrix

    def initialize_filter(self, Forward=True):

        if Forward:
            self.kalman_cov_store_forward = [None for i in range(len(self.radius_list))]
            self.ob_cov_store_forward = [None for i in range(len(self.radius_list))]
            self.kalman_fit_store_forward = np.full((len(self.radius_list), 5), np.nan)
        else:
            self.kalman_cov_store_backward = [None for i in range(len(self.radius_list))]
            self.ob_cov_store_backward = [None for i in range(len(self.radius_list))]
            self.kalman_fit_store_backward = np.full((len(self.radius_list), 5), np.nan)
            self.kalman_bud_store_backward = [None for i in range(len(self.radius_list))]
            self.kalman_F_store_backward = [None for i in range(len(self.radius_list))]

        param_error = np.array([0., 0., 0., 0., 0.])
        # print(self.var_store * 1e10)

        if Forward:
            self.kft = KalmanFilter(dim_x=5, dim_z=2)
            # ini_param = self.xyz_tpk2helix(self.param_start)
            # ini_param[3] = np.fmod(ini_param[3] - 2 * np.pi, 2 * np.pi)
            # print(ini_param)
            # ini_param += np.random.normal(0, 1e-2, 5)
            # print(ini_param)
            ini_param = self.param_helix_store_ref[1, :]

            F_mat = calculate_dermat(ini_param, self.radius_list[1], self.radius_list[2])
            # print(F_mat)

            self.kft.x = param_error  # 初始状态估计：五个参数的误差 = 0

            P0 = np.array([
                [(20 / self.radius_list[1]) ** 2, 0, 0, 0, 0],
                [0, 20 ** 2, 0, 0, 0],
                [0, 0, 0.02 ** 2, 0, 0],
                [0, 0, 0, 0.2 ** 2 / 4, 0],
                [0, 0, 0, 0, (c / 2) ** 2]
            ])
            P0 *= 100000
            #
            #
            # self.kft.P = P0  # 初始状态协方差矩阵
            self.kft.P = P0
            self.kft.Q *= 0
            ob_cov_0 = 1e0 * np.array(
                [
                    [self.loc0_list[1] ** 2 / self.radius_list[1] ** 2, 0],
                    [0, self.loc1_list[1] ** 2]
                ]
            )
        else:

            try:
                self.kft.x = param_error  # 初始状态估计：五个参数的误差 = 0
                # self.kft.P *= 1

                P0 = np.array([
                    [(20 / self.radius_list[-2]) ** 2, 0, 0, 0, 0],
                    [0, 20 ** 2, 0, 0, 0],
                    [0, 0, 0.02 ** 2, 0, 0],
                    [0, 0, 0, 0.2 ** 2 / 4, 0],
                    [0, 0, 0, 0, (c / 2) ** 2]
                ])
                P0 *= 100000

                self.kft.P = P0


            except Exception as e:
                self.kalman_cov_store_forward = [None for i in range(len(self.radius_list))]
                self.ob_cov_store_forward = [None for i in range(len(self.radius_list))]
                self.kalman_fit_store_forward = np.full((len(self.radius_list), 5), np.nan)
                # print(e)
                self.kft = KalmanFilter(dim_x=5, dim_z=2)
                self.kft.x = param_error
                P0 = np.array([
                    [(20 / self.radius_list[-2]) ** 2, 0, 0, 0, 0],
                    [0, 20 ** 2, 0, 0, 0],
                    [0, 0, 0.02 ** 2, 0, 0],
                    [0, 0, 0, 0.2 ** 2 / 4, 0],
                    [0, 0, 0, 0, (c / 2) ** 2]
                ])
                P0 *= 100000

                self.kft.P = P0
                self.kft.Q *= 0

            if self.MS:
                var_ref = self.var_store_ref[-1, 0]
                pr = c * self.B / float(self.param_helix_store_ref[0, 4]) / np.sin(self.param_helix_store_ref[0, 2])

                fac = c * self.B / pr * np.cos(self.param_helix_store_ref[-1, 2]) / np.sin(
                    self.param_helix_store_ref[-1, 2]) ** 2

                bud = np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, var_ref, 0, fac * var_ref],
                        [0, 0, 0, var_ref / np.sin(self.param_helix_store_ref[-1, 2]) ** 2, 0],
                        [0, 0, fac * var_ref, 0, fac * fac * var_ref]
                    ]
                )
                self.kalman_bud_store_backward[-1] = self.kft.P
                # print('\n\n', self.kft.P, '\n\n')
                self.kft.P += bud

                # print('\n\n', self.kft.P, '\n\n')

            F_mat = calculate_dermat(self.param_helix_store_ref[-2, :], self.radius_list[-2], self.radius_list[-1])
            F_mat = np.linalg.inv(F_mat)

            ob_cov_0 = 1e0 * np.array(
                [
                    [self.loc0_list[-1] ** 2 / self.radius_list[-1] ** 2, 0],
                    [0, self.loc1_list[-1] ** 2]
                ]
            )
        self.kft.F = F_mat

        self.kft.R = ob_cov_0  # 测量噪声协方差
        self.kft.H = np.array([[1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0]])  # 测量矩阵


        return param_error

    def kalman_one_step_forward(self, layer_idx):
        if np.isinf(self.observe_store[layer_idx, 0]):
            pass
        else:

            ob = self.observe_store[layer_idx, :] - self.kft.H @ self.param_helix_store_ref[layer_idx, :]

            self.kft.update(ob)

            self.ob_cov_store_forward[layer_idx] = self.kft.R
            self.kalman_cov_store_forward[layer_idx] = self.kft.P
            self.kalman_fit_store_forward[layer_idx, :] = self.kft.x + self.param_helix_store_ref[layer_idx, :]

        r_now = self.radius_list[layer_idx]
        r_next = self.radius_list[layer_idx + 1]

        trans_mat = calculate_dermat(self.param_helix_store_ref[layer_idx, :], r_now, r_next)
        self.kft.F = trans_mat

        if self.MS and not np.isinf(self.observe_store[layer_idx, 0]):
            var = self.var_store[layer_idx, 0]
            pr = c * self.B / float(self.param_helix_store_ref[1, 4]) / np.sin(self.param_helix_store_ref[1, 2])

            fac =  c * self.B / pr * np.cos(self.param_helix_store_ref[layer_idx, 2]) / np.sin(
                self.param_helix_store_ref[layer_idx, 2]) ** 2

            bud = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, var, 0, fac * var],
                    [0, 0, 0, var / np.sin(self.param_helix_store_ref[layer_idx, 2]) ** 2, 0],
                    [0, 0, fac * var, 0, fac * fac * var]
                ]
            )
            self.kft.P += bud

        self.kft.predict()

        ob_cov = 1e0 * np.array(
            [
                [self.loc0_list[layer_idx + 1] ** 2 / self.radius_list[layer_idx + 1] ** 2, 0],
                [0, self.loc1_list[layer_idx + 1] ** 2]
            ]
        )

        self.kft.R = ob_cov

    def kalman_one_step_backward(self, layer_idx):

        if np.isnan(self.kalman_fit_store_forward).all():
            if np.isinf(self.observe_store[layer_idx, 0]):
                pass
            else:
                # print(1)
                ob = self.observe_store[layer_idx, :] - self.kft.H @ self.param_helix_store_ref[layer_idx, :]
                # print('only back', ob)
        else:
            # if np.isinf(self.observe_store[layer_idx, 0]):
            #     pass
            # else:
            #     print(1)
            #     ob = self.observe_store[layer_idx, :] - self.kft.H @ self.param_helix_store_ref[layer_idx, :]
            #     print('only back', ob)
            if np.isnan(self.kalman_fit_store_forward[layer_idx, 0]):
                # ob = self.observe_store[layer_idx, :] - self.kft.H @ self.param_helix_store_ref[layer_idx, :]
                # print('only back', ob)
                pass
            else:
                # print(2)
                ob = self.kft.H @ self.kalman_fit_store_forward[layer_idx, :] - self.kft.H @ self.param_helix_store_ref[layer_idx, :]
                # print('both', ob)

        try:
            self.kft.update(ob)
            # print(ob)
            # print('\n\n', self.kft.P, '\n\n')

            self.kalman_fit_store_backward[layer_idx, :] = self.kft.x + self.param_helix_store_ref[layer_idx, :]
            self.ob_cov_store_backward[layer_idx] = self.kft.R
            self.kalman_cov_store_backward[layer_idx] = self.kft.P
            self.kalman_F_store_backward[layer_idx] = self.kft.F
        except Exception as e:
            # print(e)
            pass

        r_now = self.radius_list[layer_idx - 1] if layer_idx > 0 else 0
        r_next = self.radius_list[layer_idx]

        temp_param = self.param_helix_store_ref[layer_idx-1, :]
        trans_mat = calculate_dermat(temp_param, r_now, r_next)
        trans_mat = np.linalg.inv(trans_mat)
        # print("\n\n",trans_mat,"\n\n")
        self.kft.F = trans_mat
        self.kft.predict()

        if self.MS:
            var = self.var_store_ref[layer_idx - 1, 0]
            pr = c * self.B / float(self.param_helix_store_ref[1, 4]) / np.sin(self.param_helix_store_ref[1, 2])

            fac = c * self.B / pr * \
                  np.cos(self.param_helix_store_ref[layer_idx - 1, 2]) / np.sin(
                self.param_helix_store_ref[layer_idx - 1, 2]) ** 2

            bud = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, var, 0, fac * var],
                    [0, 0, 0, var / np.sin(self.param_helix_store_ref[layer_idx - 1, 2]) ** 2, 0],
                    [0, 0, fac * var, 0, fac * fac * var]
                ]
            )
            self.kalman_bud_store_backward[layer_idx] = bud


            # np.set_printoptions(10)
            # print(bud)
            # print('\n\n', self.kft.P, '\n\n')
            self.kft.P += bud
            # print('\n\n', self.kft.P, '\n\n')

        ob_cov = 1e0 * np.array(
            [
                [self.loc0_list[layer_idx - 1] ** 2 / self.radius_list[layer_idx - 1] ** 2, 0],
                [0, self.loc1_list[layer_idx - 1] ** 2]
            ]
        )

        self.kft.R = ob_cov


    def forward_kalman_estimate(self):
        self.initialize_filter(Forward=True)
        self.kft.predict()
        for t in range(0, len(self.radius_list) - 1):  # ?t是否对
            self.kalman_one_step_forward(t)

        ob = self.observe_store[len(self.radius_list) - 1, :] - self.kft.H @ self.param_helix_store_ref[
                                                                             len(self.radius_list) - 1, :]
        # print('ob:', ob)
        # print(self.kft)
        if np.isinf(self.observe_store[len(self.radius_list) - 1, 0]):
            pass
        else:
            self.kft.update(ob)
            # print(self.kft.x)
            self.kalman_fit_store_forward[len(self.radius_list) - 1, :] = self.kft.x + self.param_helix_store_ref[
                                                                                       len(self.radius_list) - 1, :]
            self.ob_cov_store_forward[len(self.radius_list) - 1] = self.kft.R
            self.kalman_cov_store_forward[len(self.radius_list) - 1] = self.kft.P

        return self.kalman_fit_store_forward


    def backward_kalman_estimate(self):
        self.initialize_filter(Forward=False)
        for t in range(len(self.radius_list) - 1, 0, -1):  # ?t是否对

            self.kalman_one_step_backward(t)
        ob = self.observe_store[0, :] - self.kft.H @ self.param_helix_store_ref[0, :]

        if np.isinf(self.observe_store[0, 0]):
            # print("pass")
            pass
        else:
            self.kft.update(ob)

            self.kalman_fit_store_backward[0, :] = self.kft.x + self.param_helix_store_ref[0, :]

            self.ob_cov_store_backward[0] = self.kft.R
            self.kalman_cov_store_backward[0] = self.kft.P
        self.kalman_F_store_backward[0] = self.kft.F


        return self.kalman_fit_store_backward

    def result_analysis(self):
        observation = self.observe_store
        ori_path = self.param_helix_store
        # ori_path = self.param_helix_store_ref
        # direct_path = self.direct_fit_store_helix
        forward_path = self.kalman_fit_store_forward
        backward_path = self.kalman_fit_store_backward

        res_ori = observation - ori_path @ self.kft.H.T
        chi2_ori = 0
        for i in range(len(self.radius_list)):
            if self.kalman_cov_store_forward[i] is not None:
                P = self.kalman_cov_store_forward[i]
                R = self.ob_cov_store_forward[i]
                E_k = self.kft.H @ P @ self.kft.H.T
                R_k = R - E_k
                chi2plus = res_ori[i] @ np.linalg.inv(R_k) @ res_ori[i].T
                chi2_ori += chi2plus

        res_f = observation - forward_path @ self.kft.H.T
        chi2_f = 0
        for i in range(len(self.radius_list)):
            if self.kalman_cov_store_forward[i] is not None:
                P = self.kalman_cov_store_forward[i]
                R = self.ob_cov_store_forward[i]
                E_k = self.kft.H @ P @ self.kft.H.T
                R_k = R - E_k
                chi2plus = res_f[i] @ np.linalg.inv(R_k) @ res_f[i].T
                chi2_f += chi2plus

        res_b = observation - backward_path @ self.kft.H.T
        chi2_b = 0
        for i in range(len(self.radius_list)):
            if self.kalman_cov_store_backward[i] is not None:
                P = self.kalman_cov_store_backward[i]
                R = self.ob_cov_store_backward[i]
                E_k = self.kft.H @ P @ self.kft.H.T
                R_k = R - E_k
                chi2plus = res_b[i] @ np.linalg.inv(R_k) @ res_b[i].T
                chi2_b += chi2plus

        forward_path = np.append(forward_path, 1 / (c * self.B) * forward_path[:, 4][:, np.newaxis], axis=1)
        backward_path = np.append(backward_path, 1 / (c * self.B) * backward_path[:, 4][:, np.newaxis], axis=1)
        ori_path = np.append(ori_path, 1 / (c * self.B) * ori_path[:, 4][:, np.newaxis], axis=1)
        # direct_path = np.append(direct_path, 1 / (c * self.B) * direct_path[:, 4][:, np.newaxis], axis=1)

        forward_path[:, 4] = c * self.B / forward_path[:, 4]
        backward_path[:, 4] = c * self.B / backward_path[:, 4]
        ori_path[:, 4] = c * self.B / ori_path[:, 4]
        # direct_path[:, 4] = c * self.B / direct_path[:, 4]

        forward_error = forward_path - ori_path
        backward_error = backward_path - ori_path
        # direct_error = direct_path - ori_path

        # res_f = forward_path @ self.kft.H.T - ori_path @ self.kft.H.T
        # res_b = backward_path @ self.kft.H.T - ori_path @ self.kft.H.T

        forward_error = np.where(np.isinf(forward_error), 1, forward_error)
        forward_error = np.where(np.isnan(forward_error), 1, forward_error)

        backward_error = np.where(np.isinf(backward_error), 1, backward_error)
        backward_error = np.where(np.isnan(backward_error), 1, backward_error)

        # direct_error = np.where(np.isinf(direct_error), 1, direct_error)
        # direct_error = np.where(np.isnan(direct_error), 1, direct_error)

        res_ori = np.where(np.isinf(res_ori), 1, res_ori)
        res_ori = np.where(np.isnan(res_ori), 1, res_ori)

        res_f = np.where(np.isinf(res_f), 1, res_f)
        res_f = np.where(np.isnan(res_f), 1, res_f)

        res_b = np.where(np.isinf(res_b), 1, res_b)
        res_b = np.where(np.isnan(res_b), 1, res_b)

        # print("chi2_f: ", chi2_f, "chi2_b: ", chi2_b)
        first_idx = self.first_layer_idx
        result_table = {
            "chi2_forward": chi2_f,
            "chi2_backward": chi2_b,




            "forward_dr": self.radius_list * forward_error[:, 0],
            "forward_dz": forward_error[:, 1],
            "forward_dt": forward_error[:, 2],
            "forward_df": forward_error[:, 3],
            "forward_dp": forward_error[:, 4] * 1e3 / self.p,
            "forward_dp2": forward_error[:, 5] * 1e5,

            "backward_dr": self.radius_list * backward_error[:, 0],
            "backward_dz": backward_error[:, 1],
            "backward_dt": backward_error[:, 2],
            "backward_df": backward_error[:, 3] + backward_error[:, 0],
            "backward_dp": backward_error[:, 4] * 1e3 / self.p,
            "backward_dp2": backward_error[:, 5] * 1e5,

            # "direct_dr": direct_error[:, 0],
            # "direct_dz": direct_error[:, 1],
            # "direct_dt": direct_error[:, 2],
            # "direct_df": direct_error[:, 3],
            # "direct_dp": direct_error[:, 4] * 1e3 / self.p,
            # "direct_dp2": direct_error[:, 5] * 1e5,

            "ori_path": self.param_helix_store,
            # "direct_path": self.direct_fit_store_helix,
            "measure_path": self.observe_store,
            "forward_path": self.kalman_fit_store_forward,
            "backward_path": self.kalman_fit_store_backward,

            "res_ori": res_ori,
            "res_forward": res_f,
            "res_backward": res_b,

            "f_dr_first": self.radius_list[0] * forward_error[first_idx, 0],
            "f_dz_first": forward_error[first_idx, 1],
            "f_dt_first": forward_error[first_idx, 2],
            "f_df_first": forward_error[first_idx, 3],
            "f_dp_first": forward_error[first_idx, 4] * self.p,
            "f_dp2_first": c * self.B / forward_error[first_idx, 4] / self.p ** 2,

            "b_dr_first": self.radius_list[0] * backward_error[first_idx, 0],
            "b_dz_first": backward_error[first_idx, 1],
            "b_dt_first": backward_error[first_idx, 2],
            "b_df_first": backward_error[first_idx, 3],
            "b_dp_first": c * self.B / backward_error[first_idx, 4] / self.p,
            "b_dp2_first": c * self.B / backward_error[first_idx, 4] / self.p ** 2,



            # "d_dr_first": direct_error[first_idx, 0],
            # "d_dz_first": direct_error[first_idx, 1],
            # "d_dt_first": direct_error[first_idx, 2],
            # "d_df_first": direct_error[first_idx, 3],
            # "d_dp_first": c * self.B / direct_error[first_idx, 4] / self.p,
            # "d_dp2_first": c * self.B / direct_error[first_idx, 4] / self.p ** 2,

        }


        # cov = {
        #     "backward_dr": backward_error[:, 0],
        #     "backward_dz": backward_error[:, 1],
        #     "backward_dt": backward_error[:, 2],
        #     "backward_df": backward_error[:, 3] + backward_error[:, 0],
        #     "backward_dp": backward_error[:, 4] * 1e3 / self.p,
        #     "backward_dp2": backward_error[:, 5] * 1e5,
        #     "bud_py": self.kalman_bud_store_backward[1:] if None not in self.kalman_bud_store_backward  else 0,
        #     "backward_ob_cov": self.ob_cov_store_backward[1:],
        #     "backward_filter_cov": self.kalman_cov_store_backward[1:],
        #     "backward_F": self.kalman_F_store_backward[:-1],
        #     "ref_errorpy": self.param_helix_store_ref - self.param_helix_store,
        # }
        # scipy.io.savemat(r"D:\files\pyproj\detector\cov_dic\py\result" + str(np.random.randint(1000)) + ".mat", cov)
        ini_params = {
            "B": self.B,
            "p": self.p,
            "theta": self.theta,
            "phi": self.phi,
            "MS": self.MS,
            "RE": self.RE,
            "mass": self.mass,
            "charge": self.charge,
            "beam_spot": self.beam_spot
        }

        # return forward_error, backward_error, res_ori, res_f, res_b, chi2_ori, chi2_f, chi2_b
        del self.kft, self.kalman_fit_store_forward
        return ini_params, result_table


def get_one_track():
    dec, emit, envir = ctp.get_normal_dec_par_envir_set()
    dec_info_list = dec.get_param()
    envir_info_list = envir.get_param()

    res = Resolution(dec_info_list, envir_info_list, emit)
    return res.generate_path()


def vertconv(parami, Cp, Bz, R, iopt=0, unit=1.0, convf=0.299792458e-3):
    """
    Converts coordinates and momenta from DELPHI input parameters to cylindrical coordinates.
    :param parami: a list of five DELPHI input parameters (Phi, z, theta, beta, kappa)
    :param Bz: the z-component of the magnetic field
    :return: a tuple of the converted parameters (q1, q2, q3, px, py, pz) and the converted covariance matrix (Cq)
    """
    Phi, z, theta, beta, kappa = parami
    phi = Phi + beta
    convf = convf * unit

    # momentum and coordinates
    # print(kappa)
    fac = convf * math.fabs(Bz) / math.fabs(kappa)
    pz = 1. / math.tan(theta) * fac
    py = math.sin(phi) * fac
    px = math.cos(phi) * fac
    q1 = R * math.cos(Phi) / unit
    q2 = R * math.sin(Phi) / unit
    q3 = float(z) / unit

    # derivative matrix elements
    dpzkappa = -pz / kappa
    dpykappa = -py / kappa
    dpxkappa = -px / kappa
    dpybeta = px
    dpxbeta = -py
    dpztheta = -1 / (np.sin(theta)) ** 2
    dpyPhi = px
    dpxPhi = -py
    dzz = unit / 10
    dyPhi = q1
    dxPhi = -q2

    # derivative matrix
    D = np.zeros((6, 5))
    Cq = np.zeros((6, 6))
    if iopt > 0:
        D[0, 0] = dxPhi
        D[1, 0] = dyPhi
        D[3, 0] = dpxPhi
        D[4, 0] = dpyPhi
        D[2, 1] = dzz
        D[5, 2] = dpztheta
        D[3, 3] = dpxbeta
        D[3, 4] = dpxkappa
        D[4, 3] = dpybeta
        D[4, 4] = dpykappa
        D[5, 4] = dpzkappa

        # converted covariance matrix
        # Cq = D * Cp * D.T

    # converted parameters
    qaramf = np.array([[q1, q2, q3, px, py, pz], ])

    return qaramf


def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def chi2_pdf(x, df):
    return chi2.pdf(x, df)


if __name__ == "__main__":

    dec, emit, envir = ctp.get_normal_dec_par_envir_set()
    # envir.update_environment("multiple_scattering", False)
    print(envir)
    dec_info_list = dec.get_param()
    envir_info_list = envir.get_param()

    res = Resolution(dec_info_list, envir_info_list, emit)

    # plt.ion()
    forward_error_list = []
    backward_error_list = []
    chi2_f_list = []
    chi2_b_list = []
    chi2_ori_list = []
    res_ori_list = []
    res_f_list = []
    res_b_list = []

    t = time.time()

    test_num = 1000
    # res.ini_fig()
    re = Result(test_num)
    for i in tqdm(range(test_num)):
        ori_path, _ = res.generate_path()
        forward_path = res.forward_kalman_estimate()
        backward_path = res.backward_kalman_estimate()
        res.direct_fit()
        # res.PLOT()
        ini_param, result_table = res.result_analysis()
        re.append(ini_param, result_table)

    # print(emit.emit_param["Emit_mode"][0])
    for item in tqdm(re.kalman_post_process(emit.emit_param["Emit_mode"][0], len(dec))):
        print(item)
        pass

    # fig = plt.figure()
    # re.kalman_plot(x_axis="p", y_axis="dp2", emit_mode=emit.emit_param["Emit_mode"][0], layer_idx=2,
    #                filter="forward")
    fig = plt.figure()
    re.kalman_plot(x_axis="p", y_axis="dp2", emit_mode=emit.emit_param["Emit_mode"][0], layer_idx=2,
                   filter="backward")
    fig = plt.figure()
    re.kalman_plot(x_axis="p", y_axis="dp2", emit_mode=emit.emit_param["Emit_mode"][0], layer_idx=2,
                   filter="direct")

    # print(time.time() - t)
    #
    plt.show()
    sys.exit(0)
    #
    # res_ori_list = np.array(res_ori_list)
    # forward_error_list = np.array(forward_error_list)
    # backward_error_list = np.array(backward_error_list)
    # res_f_list = np.array(res_f_list)
    # res_b_list = np.array(res_b_list)
    # # plt.pause(100)
    # for j in range(0, 2):
    #     fig1 = plt.figure(figsize=(40, 30))
    #     fig2 = plt.figure(figsize=(40, 30))
    #     fig3 = plt.figure(figsize=(40, 30))
    #
    #     fig1.suptitle("ori error" + str(j), fontsize=30)
    #     fig2.suptitle("forward error" + str(j), fontsize=30)
    #     fig3.suptitle("backward error" + str(j), fontsize=30)
    #
    #     # 打印拟合后的参数
    #     for i in tqdm(range(0, len(dec))):
    #
    #         ax1 = fig1.add_subplot(3, 7, i + 1)
    #         ax2 = fig2.add_subplot(3, 7, i + 1)
    #         ax3 = fig3.add_subplot(3, 7, i + 1)
    #         if j == 0:
    #             ax1.hist(res_ori_list[:, i, j] * res.radius_list[i], bins=100, density=True, facecolor="blue",
    #                      edgecolor="black", alpha=0.7)
    #             ax2.hist(res_f_list[:, i, j] * res.radius_list[i], bins=100, density=True, facecolor="blue",
    #                      edgecolor="black", alpha=0.7)
    #             ax3.hist(res_b_list[:, i, j] * res.radius_list[i], bins=100, density=True, facecolor="blue",
    #                      edgecolor="black", alpha=0.7)
    #
    #             # 拟合高斯分布
    #             params1 = stats.norm.fit(res_ori_list[:, i, j] * res.radius_list[i])
    #             params2 = stats.norm.fit(res_f_list[:, i, j] * res.radius_list[i])
    #             params3 = stats.norm.fit(res_b_list[:, i, j] * res.radius_list[i])
    #
    #             # 画出拟合的高斯分布
    #             x1 = np.linspace(-max(abs(res_ori_list[:, i, j] * res.radius_list[i])),
    #                              max(abs(res_ori_list[:, i, j] * res.radius_list[i])), 100)
    #             y1 = stats.norm.pdf(x1, *params1)
    #
    #             x2 = np.linspace(-max(abs(res_f_list[:, i, j] * res.radius_list[i])),
    #                              max(abs(res_f_list[:, i, j] * res.radius_list[i])), 100)
    #             y2 = stats.norm.pdf(x2, *params2)
    #
    #             x3 = np.linspace(-max(abs(res_b_list[:, i, j] * res.radius_list[i])),
    #                              max(abs(res_b_list[:, i, j] * res.radius_list[i])), 100)
    #             y3 = stats.norm.pdf(x3, *params3)
    #
    #         else:
    #
    #             ax1.hist(res_ori_list[:, i, j], bins=100, density=True, facecolor="blue", edgecolor="black",
    #                      alpha=0.7)
    #             ax2.hist(res_f_list[:, i, j], bins=100, density=True, facecolor="blue", edgecolor="black",
    #                      alpha=0.7)
    #             ax3.hist(res_b_list[:, i, j], bins=100, density=True, facecolor="blue", edgecolor="black",
    #                      alpha=0.7)
    #
    #             # 拟合高斯分布
    #             params1 = stats.norm.fit(res_ori_list[:, i, j])
    #             params2 = stats.norm.fit(res_f_list[:, i, j])
    #             params3 = stats.norm.fit(res_b_list[:, i, j])
    #
    #             # 画出拟合的高斯分布
    #             x1 = np.linspace(-max(abs(res_ori_list[:, i, j])), max(abs(res_ori_list[:, i, j])), 100)
    #             y1 = stats.norm.pdf(x1, *params1)
    #
    #             x2 = np.linspace(-max(abs(res_f_list[:, i, j])), max(abs(res_f_list[:, i, j])), 100)
    #             y2 = stats.norm.pdf(x2, *params2)
    #
    #             x3 = np.linspace(-max(abs(res_b_list[:, i, j])), max(abs(res_b_list[:, i, j])), 100)
    #             y3 = stats.norm.pdf(x3, *params3)
    #
    #         ax1.plot(x1, y1, color='red', label='ori')
    #         ax2.plot(x2, y2, color='red', label='forward')
    #         ax3.plot(x3, y3, color='red', label='backward')
    #
    #         # title with sigma and mean
    #         ax1.set_title("s: %.1e, m: %.1e" % (params1[1], params1[0]), fontsize=20)
    #         ax2.set_title("s: %.1e, m: %.1e" % (params2[1], params2[0]), fontsize=20)
    #         ax3.set_title("s: %.1e, m: %.1e" % (params3[1], params3[0]), fontsize=20)
    #         # plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
    #
    #     # save and close
    #     fig1.savefig("res_ori" + str(j) + ".png", dpi=100)
    #     fig2.savefig("res_forward" + str(j) + ".png", dpi=100)
    #     fig3.savefig("res_backward" + str(j) + ".png", dpi=100)
    #     plt.close(fig1)
    #     plt.close(fig2)
    #     plt.close(fig3)
    #
    # # draw a histogram
    # for j in range(0, 5):
    #     fig2 = plt.figure(figsize=(40, 30))
    #     fig3 = plt.figure(figsize=(40, 30))
    #     fig2.suptitle("forward error" + str(j), fontsize=30)
    #     fig3.suptitle("backward error" + str(j), fontsize=30)
    #
    #     # 打印拟合后的参数
    #     for i in tqdm(range(0, len(dec))):
    #         ax2 = fig2.add_subplot(3, 7, i + 1)
    #         ax2.hist(forward_error_list[:, i, j], bins=100, density=True, facecolor="blue", edgecolor="black",
    #                  alpha=0.7)
    #
    #         ax3 = fig3.add_subplot(3, 7, i + 1)
    #         ax3.hist(backward_error_list[:, i, j], bins=100, density=True, facecolor="blue", edgecolor="black",
    #                  alpha=0.7)
    #
    #         # 拟合高斯分布
    #         params2 = stats.norm.fit(forward_error_list[:, i, j])
    #         params3 = stats.norm.fit(backward_error_list[:, i, j])
    #
    #         # 画出拟合的高斯分布
    #         x2 = np.linspace(-max(abs(forward_error_list[:, i, j])), max(abs(forward_error_list[:, i, j])), 100)
    #         x3 = np.linspace(-max(abs(backward_error_list[:, i, j])), max(abs(backward_error_list[:, i, j])), 100)
    #         y2 = stats.norm.pdf(x2, *params2)
    #         y3 = stats.norm.pdf(x3, *params3)
    #
    #         ax2.plot(x2, y2, color='blue', label='backward')
    #         ax3.plot(x3, y3, color='cyan', label='backward')
    #
    #         # title with sigma and mean
    #         ax2.set_title("s: %.1e, m: %.1e" % (params2[1], params2[0]), fontsize=20)
    #         ax3.set_title("s: %.1e, m: %.1e" % (params3[1], params3[0]), fontsize=20)
    #         # plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
    #
    #     # save and close
    #     fig2.savefig("forward_error" + str(j) + ".png", dpi=100)
    #     fig3.savefig("backward_error" + str(j) + ".png", dpi=100)
    #     plt.close(fig2)
    #     plt.close(fig3)
    #
    # # draw a histogram of chi2_f and chi2_b in the same figure
    # fig2 = plt.figure(figsize=(30, 10))
    # fig2.suptitle("chi2_f and chi2_b", fontsize=10)
    # ax1 = fig2.add_subplot(1, 3, 1)
    # ax1.hist(chi2_ori_list, bins=100, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    #
    # ax2 = fig2.add_subplot(1, 3, 2)
    # ax2.hist(chi2_f_list, bins=100, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    #
    # params = stats.chi2.fit(chi2_f_list)
    # x = np.linspace(0, max(chi2_f_list), 100)
    # y = stats.chi2.pdf(x, *params)
    # ax2.plot(x, y, color='red', label='chi2')
    #
    # max_index = np.argmax(y)
    # max_x = x[max_index]
    # max_y = y[max_index]
    # plt.scatter(max_x, max_y, color='blue', label='Max point')
    # plt.text(max_x, max_y, f'({max_x:.2f}, {max_y:.2f})', fontsize=10, verticalalignment='bottom', color='red')
    #
    # ax3 = fig2.add_subplot(1, 3, 3)
    # ax3.hist(chi2_b_list, bins=100, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    #
    # params = stats.chi2.fit(chi2_b_list)
    # x = np.linspace(0, max(chi2_b_list), 100)
    # y = stats.chi2.pdf(x, *params)
    # ax3.plot(x, y, color='red', label='chi2')
    #
    # max_index = np.argmax(y)
    # max_x = x[max_index]
    # max_y = y[max_index]
    # plt.scatter(max_x, max_y, color='blue', label='Max point')
    # plt.text(max_x, max_y, f'({max_x:.2f}, {max_y:.2f})', fontsize=10, verticalalignment='bottom', color='red')
    #
    # fig2.savefig("chi2_f and chi2_b.png", dpi=100)
    # plt.close(fig2)
    # # 调整布局
    # # plt.tight_layout()
    # #
    # # plt.pause(1000000)
