import time
from DDFS.element import Environment, Particle, Emitter, Detector, SiLayer, Result
import numpy as np
import math
import DDFS.common_test_parameter as ctp
import copy
from tqdm import tqdm

from pylab import *
from datetime import datetime

c = np.float64(0.299792458)


def fy(y):
    if y < 1e-6:
        return 0.0
    else:
        return 0.0136 * y ** 0.5 * (1.0 + 0.038 * math.log(y))


def Beta(p, m):
    return p / (p * p + m * m) ** 0.5


class Resolution:
    def __init__(self, dec_info_list, environment_info_list, emitter):
        self.emitter = emitter
        self.radius_list, self.budget_list, self.efficiency_list, self.loc0_list, self.loc1_list, self.half_z_list = dec_info_list
        self.radius_list = [i * 1000. for i in self.radius_list]  # convert to um
        self.loc0_list = [i * 1000 for i in self.loc0_list]
        self.loc1_list = [i * 1000 for i in self.loc1_list]
        self.zero_efficient_num = self.efficiency_list.count(0.0)
        # print(self.zero_efficient_num)
        self.B, mutiple_scattering_FLAG, position_resolution_FLAG = environment_info_list
        self.MS = mutiple_scattering_FLAG
        self.RE = position_resolution_FLAG

        # self.par, self.phi, self.theta, _, self.index_par = emitter.get_param()
        self.get_param()

        self.alpha = np.float64(1000000. / c / self.B)  # magnetic-field-constant 1/(c*B)

    def __str__(self):
        return "analytic method"

    def copy(self):
        radius_list = [i / 1000. for i in self.radius_list]  # convert to um
        loc0_list = [i / 1000 for i in self.loc0_list]
        loc1_list = [i / 1000 for i in self.loc1_list]
        dec_info_list = radius_list, self.budget_list, self.efficiency_list, loc0_list, loc1_list, self.half_z_list
        environment_info_list = self.B, self.MS, self.RE
        emitter = self.emitter.copy()
        # print(emitter)
        return Resolution(dec_info_list, environment_info_list, emitter)

    def get_scat_list(self):
        dphi = [self.getdPhiRL(R / 1000.) for R in self.radius_list]
        scat_list = [fy(x / math.sin(self.theta) / math.cos(y)) for x, y in zip(self.budget_list, dphi)]
        return scat_list

    def ms(self, m, n):
        ret = 1.0e-10
        pbeta = self.p / sin(self.theta) * Beta(self.p / sin(self.theta), self.mass)
        if m > 0 and n > 0:
            N = min(m, n)
            for j in range(0, N):  # m < n
                x = self.scat_list[j] / pbeta
                x2 = x * x
                ym = (self.radius_list[m] - self.radius_list[j])
                yn = (self.radius_list[n] - self.radius_list[j])
                x3 = x2 * ym * yn
                ret += x3
        return ret

    def getdPhiRL(self, rLayer):

        B = self.B
        trkpar = np.array([0, 0, self.theta, self.phi, self.p])
        HFPI = math.pi / 2.0
        PI2 = 2.0 * math.pi
        dr = trkpar[0]
        dz = trkpar[1]
        fi0 = trkpar[3]
        tanl = math.tan(HFPI - trkpar[3])
        kap = 1 / trkpar[4]

        rw = rLayer
        c = 0.299792458
        alp = 1000. / c / B

        phi0 = fi0 - HFPI
        if phi0 > math.pi:
            phi0 -= PI2
        elif phi0 < -math.pi:
            phi0 += PI2

        R = alp / kap
        aa = R * R + (dr + R) * (dr + R) - rw * rw
        bb = 2 * R * (R + dr)
        cc = aa / bb
        dd = math.acos(cc)

        if kap > 0.:
            phi = phi0 + dd
        else:
            phi = phi0 - dd

        if phi > math.pi:
            phi -= PI2
        if phi < -math.pi:
            phi += PI2

        x0 = dr * math.cos(phi0)
        y0 = dr * math.sin(phi0)
        z0 = dz

        pos = np.zeros(3)
        pos[0] = x0 + R * (math.cos(phi0) - math.cos(phi))
        pos[1] = y0 + R * (math.sin(phi0) - math.sin(phi))

        if kap > 0.:
            pos[2] = z0 - R * tanl * dd
        else:
            pos[2] = z0 + R * tanl * dd

        phiR = math.atan2(pos[1], pos[0]) + HFPI
        phi = phi - phiR
        if phi > math.pi:
            phi = phi - PI2
        elif phi < -math.pi:
            phi = phi + PI2
        # print("R, p, phi0, phi = %7.1f %5.1f, %6.1f %6.1f"%(rw, trkpar[4], phiR/math.pi*180,phi/math.pi*180))

        return phi  # , pos

    def AnaGrad(self, radius):
        Theta = self.theta
        Phi = self.phi
        p = self.p
        B = self.B
        alpha = self.alpha

        D0 = 0
        Z0 = 0

        dr = np.float64(D0)
        dz = np.float64(Z0)
        lambda_0 = np.float64(math.pi / 2. - Theta)
        tan_lambda = math.tan(lambda_0)
        phi_0 = np.float64(Phi)

        kappa = 1. / p

        R = alpha / kappa
        sign_R = np.sign(R)  # ???

        aa = (dr + R) * (dr + R) + R * R - radius * radius
        bb = 2 * R * (dr + R)
        cos_phi = aa / bb
        sinphi = np.power(1 - cos_phi * cos_phi, 0.5)
        phi = math.acos(aa / bb)
        # print("phi= %10.4f" % phi)
        if sign_R < 0.0:
            phi = - phi

        x = dr * math.cos(phi_0) + R * (math.cos(phi_0) - math.cos(phi_0 + phi))
        y = dr * math.sin(phi_0) + R * (math.sin(phi_0) - math.sin(phi_0 + phi))
        z = dz - (R * tan_lambda * phi) * sign_R  # why change sign?

        dloc_dxyz = np.mat(np.zeros((2, 3), dtype=np.float64))
        dxyz_dtrk = np.mat(np.zeros((3, 5), dtype=np.float64))

        # loc0    = r * np.atan2(y,x)
        # loc1    = z
        # 雅各比行列式 \partial m / \partial x
        dloc_dxyz[0, 0] = - radius * y / (x * x + y * y)
        dloc_dxyz[0, 1] = + radius * x / (x * x + y * y)
        dloc_dxyz[0, 2] = 0.
        # ?这里为什么是0，0，1是因为theta很接近pi/2？
        dloc_dxyz[1, 0] = - radius * y / (x * x + y * y) * math.cos(Theta)
        dloc_dxyz[1, 1] = + radius * x / (x * x + y * y) * math.cos(Theta)
        dloc_dxyz[1, 2] = math.sin(Theta)

        # dloc_dxyz[1, 0] = 0.
        # dloc_dxyz[1, 1] = 0.
        # dloc_dxyz[1, 2] = 1.

        dphi_ddr = sign_R * (1 / R - cos_phi / (dr + R)) / sinphi
        dphi_dkap = sign_R * kappa * (dr * dr - radius * radius) * (2. * alpha + dr * kappa) / (
                2. * alpha * (alpha + dr * kappa) * (alpha + dr * kappa) * sinphi)

        dx_ddr = math.cos(phi_0) + alpha / kappa * math.sin(phi_0 + phi) * dphi_ddr
        dy_ddr = math.sin(phi_0) - alpha / kappa * math.cos(phi_0 + phi) * dphi_ddr
        dz_ddr = alpha / kappa * tan_lambda * dphi_ddr

        dx_dphi0 = - dr * math.sin(phi_0) - alpha / kappa * (math.sin(phi_0) - math.sin(phi_0 + phi))
        dy_dphi0 = dr * math.cos(phi_0) + alpha / kappa * (math.cos(phi_0) - math.cos(phi_0 + phi))
        dz_dphi0 = 0.0

        dx_dkap = - alpha * (math.cos(phi_0) - math.cos(phi_0 + phi)) / kappa ** 2 + alpha / kappa * math.sin(
            phi_0 + phi) * dphi_dkap
        dy_dkap = - alpha * (math.sin(phi_0) - math.sin(phi_0 + phi)) / kappa ** 2 - alpha / kappa * math.cos(
            phi_0 + phi) * dphi_dkap
        dz_dkap = - alpha / kappa ** 2 * tan_lambda * (kappa * dphi_dkap - phi)  # ?缺个负号？

        dx_ddz = 0.0
        dy_ddz = 0.0
        dz_ddz = 1.0

        dx_dtanl = 0.0
        dy_dtanl = 0.0
        dz_dtanl = - alpha / kappa * phi
        dz_dtheta = - alpha / kappa * phi / math.cos(lambda_0) / math.cos(lambda_0)  # ？这是干什么

        dx_dpt = dx_dkap * sign_R * (+kappa * kappa)
        dy_dpt = dy_dkap * sign_R * (+kappa * kappa)
        dz_dpt = dz_dkap * sign_R * (-kappa * kappa)

        dxyz_dtrk[0, 0] = dx_ddr
        dxyz_dtrk[0, 1] = dx_ddz
        dxyz_dtrk[0, 2] = dx_dtanl
        dxyz_dtrk[0, 3] = dx_dphi0
        dxyz_dtrk[0, 4] = dx_dpt

        dxyz_dtrk[1, 0] = dy_ddr
        dxyz_dtrk[1, 1] = dy_ddz
        dxyz_dtrk[1, 2] = dy_dtanl
        dxyz_dtrk[1, 3] = dy_dphi0
        dxyz_dtrk[1, 4] = dy_dpt

        dxyz_dtrk[2, 0] = dz_ddr
        dxyz_dtrk[2, 1] = dz_ddz
        dxyz_dtrk[2, 2] = dz_dtanl
        dxyz_dtrk[2, 2] = dz_dtheta  # ？这又是啥，有个重复
        dxyz_dtrk[2, 3] = dz_dphi0
        dxyz_dtrk[2, 4] = dz_dpt  # * sz

        Grad = dloc_dxyz * dxyz_dtrk

        return Grad

    def get_Grad(self):
        num_of_layer = len(self.radius_list)
        G = np.mat(np.zeros((5, (num_of_layer - self.zero_efficient_num) * 2), dtype=np.float64))  # ? 为什么要-4

        m = 0
        for layer_idx in range(num_of_layer):
            loc0 = self.loc0_list[layer_idx]
            if loc0 < 8000.:  # 区分支撑材料， 误差大于8mm 应该用效率
                radius = self.radius_list[layer_idx]

                Grad = self.AnaGrad(radius)
                for row in range(5):
                    for col in range(2):
                        G[row, m + col] = Grad[col, row]
                m += 2
        return G

    def get_Cy(self):
        """
        :return: 通过计算C_yR和C_yM两个协方差矩阵来计算所有的C_y
        """
        num_of_layer = len(self.radius_list)  # ?这个SP又是啥

        CyRmn = np.mat(
            np.zeros(((num_of_layer - self.zero_efficient_num) * 2, (num_of_layer - self.zero_efficient_num) * 2),
                     dtype=np.float64))
        CyMmn = np.mat(
            np.zeros(((num_of_layer - self.zero_efficient_num) * 2, (num_of_layer - self.zero_efficient_num) * 2),
                     dtype=np.float64))

        if num_of_layer < 0:
            print("wrong")
            raise ValueError

        if self.RE:
            m = 0
            for layer_idx in range(0, num_of_layer):
                if self.loc0_list[layer_idx] < 8000.:  # 8000是什么？
                    CyRmn[m + 0, m + 0] = self.loc0_list[layer_idx] ** 2
                    CyRmn[m + 1, m + 1] = self.loc1_list[layer_idx] ** 2
                    m += 2

        # if self.RE:
        #     mask = np.array( self.loc0_list) < 8000.
        #     selected_layers = np.where(mask)[0]
        #
        #     CyRmn = np.zeros((2 * len(selected_layers), 2 * len(selected_layers)))
        #
        #     loc0_squared = np.array( self.loc0_list)[selected_layers] ** 2
        #     loc1_squared = np.array( self.loc1_list)[selected_layers] ** 2
        #
        #     indices = np.arange(0, 2 * len(selected_layers), 2)
        #     CyRmn[indices, indices] = loc0_squared
        #     CyRmn[indices + 1, indices + 1] = loc1_squared

        # if self.MS:
        #
        #     m = 0
        #     for layer_idx_0 in range(0, num_of_layer):
        #         if self.loc0_list[layer_idx_0] < 8000.:
        #             n = 0
        #             for layer_idx_temp in range(0, num_of_layer):
        #                 if self.loc0_list[layer_idx_temp] < 8000.:  # ? 判断了两次loc0_list[layer_idx_temp] < 8000
        #                     CyMmn[m + 0, n + 0] = self.ms(layer_idx_0, layer_idx_temp)
        #                     CyMmn[m + 1, n + 1] = self.ms(layer_idx_0, layer_idx_temp)
        #
        #                     n += 2
        #             m += 2
        # xxx = np.array(CyMmn)

        if self.MS:
            mask = np.array(self.loc0_list) < 8000.
            num_selected_layers = np.sum(mask)

            selected_layers = np.where(mask)[0]
            CyMmn = np.zeros((2 * num_selected_layers, 2 * num_selected_layers))

            for i, layer_idx_0 in enumerate(selected_layers):
                for j, layer_idx_temp in enumerate(selected_layers):
                    CyMmn[2 * i:2 * i + 2, 2 * j:2 * j + 2] = self.ms(layer_idx_0, layer_idx_temp) * np.eye(2)

        # xxx = np.array(CyMmn)

        Cymn = CyRmn + CyMmn

        return Cymn, CyRmn, CyMmn

    def get_param(self):
        self.par, ini_params, self.index_par = self.emitter.get_param()

        # print(ini_params)
        self.p = ini_params["p"]
        self.theta = ini_params["theta"]
        self.phi = ini_params["phi"]
        # print(self.p)

        self.particle_info_list = self.par.get_param()

        self.mass, self.charge, self.beam_spot = self.particle_info_list

        self.scat_list = self.get_scat_list()

    def analytic_estimate(self):
        self.get_param()
        Gmn = self.get_Grad()
        Cy, Cy_R, Cy_M = self.get_Cy()

        Camn = np.linalg.inv(Gmn @ np.linalg.inv(Cy) @ Gmn.transpose())

        # ?det 但是求迹
        det = np.trace(Camn)

        dr = (Camn[0, 0] ** 0.5)
        dz = (Camn[1, 1] ** 0.5)
        dt = (Camn[2, 2] ** 0.5) * 1e3
        df = (Camn[3, 3] ** 0.5) * 1e3
        # 不是已经除了一个p了，外面还要除？
        dp = (Camn[4, 4] ** 0.5) * 1e0 / self.p
        # print(dr, dz, dt, df, dp, det)

        result = {
            "dr": dr,
            "dz": dz,
            "dt": dt,
            "df": df,
            "dp": dp * 1e3,
            "det": det,
            "dp2" : dp / self.p * 1e5
        }
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

        return ini_params, result


if __name__ == "__main__":
    print(1)
    dec, emit, envir = ctp.get_normal_dec_par_envir_set()

    dec_info_list = dec.get_param()
    envir_info_list = envir.get_param()
    t = time.time()
    re = Result()
    resolution = Resolution(dec_info_list, envir_info_list, emit)
    resolution = resolution.copy()

    for i in tqdm(range(200)):
        # print(i)
        ini, ret = resolution.analytic_estimate()
        re.append(ini, ret)

    # print(re.initial)
    # print(re.result)

    re.plot("theta", "det")

    # print(ret, time.time() - t)

    # analytic_estimate(dec_info_list, envir_info_list, emit)
