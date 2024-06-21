import numpy as np
import math

np.set_printoptions(precision=4, suppress=False)

def calculate_dermat(param, r_now, r_next, reverse=False):
    try:
        phi, z, theta, beta, kappa = param
    except Exception as e:
        # print(e)
        try:
            phi, z, theta, beta, kappa = param[0]
            # print(param[0])
        except Exception as e2:
            print(param)
            raise TypeError("cannot unzip")

    dermat = np.eye(5)
    sinb = np.sin(beta)
    cosb = np.cos(beta)
    cotth = 1 / np.tan(theta)
    r_trk = 1 / kappa

    xc = (r_now - r_trk * sinb)
    yc = r_trk * cosb
    rc = np.sqrt(xc ** 2 + yc ** 2)

    rrr = (r_next ** 2 - r_trk ** 2 - rc ** 2) / (2 * r_trk)
    delt = rc ** 2 - rrr ** 2
    delt = np.sqrt(delt)

    sinf = (xc * rrr + yc * delt) / rc ** 2
    cosf = (xc * delt - yc * rrr) / rc ** 2

    xf = xc + r_trk * sinf
    yf = yc - r_trk * cosf

    alff = np.arctan2(sinf, cosf)
    dphi = alff - beta
    alrphi = r_trk * dphi

    sinbf = (sinf * xf - cosf * yf) / r_next
    cosbf = np.sqrt(1 - sinbf ** 2)

    ccpsi = r_now - r_trk * sinb
    scpsi = r_trk * cosb

    ccpsf = r_next - r_trk * sinbf
    scpsf = r_trk * cosbf

    cpsii = r_trk - r_now * sinb
    spsii = -r_now * cosb

    cpsif = r_trk - r_next * sinbf
    spsif = -r_next * cosbf

    sdphi = np.sin(dphi)
    cdphi = np.cos(dphi)

    fact = -r_trk / spsif

    dermat[0, 3] = sdphi * fact
    dermat[0, 4] = fact * r_trk * (1.0 - cdphi)
    dermat[1, 2] = -r_trk * dphi * ((1.0 + cotth ** 2) - 0)#np.pi/2 / theta)  # 似乎把-号去掉结果才能对上
    dermat[1, 3] = -r_trk * cotth * (r_next * ccpsf * spsii / spsif - r_now * ccpsi) / rc ** 2
    dermat[1, 4] = -r_trk ** 2 * cotth * (
                -dphi + sinbf / cosbf - (r_now * scpsi + r_next * ccpsf * cpsii / spsif) / rc ** 2)
    dermat[3, 3] = spsii / spsif
    dermat[3, 4] = r_trk * (cpsif - cpsii) / spsif
    if reverse:
        dermat = np.linalg.inv(dermat)

    return dermat



def predict_param_nonliner(param, dt, r_now, r_next, charge):
    try:
        phi, z, theta, beta, kappa = param
    except Exception as e:
        # print(e)
        try:
            phi, z, theta, beta, kappa = param[0]
            # print(param[0])
        except Exception as e2:
            print(param)
            raise TypeError("cannot unzip")
    param = [phi, z, theta, beta, kappa]

    sinb = math.sin(beta)
    cosb = math.cos(beta)

    if charge > 0:
        r_trk = -1 / param[4]
    else:
        r_trk = 1 / param[4]




    xc = r_now - r_trk * sinb
    yc = r_trk * cosb

    rc = math.sqrt(xc ** 2 + yc ** 2)


    rrr = (r_next ** 2 - r_trk ** 2 - rc ** 2) / (2 * r_trk)
    delt = rc ** 2 - rrr ** 2

    delt = math.sqrt(delt)

    sinf = (xc * rrr + yc * delt) / rc ** 2
    cosf = (xc * delt - yc * rrr) / rc ** 2

    xf = xc + r_trk * sinf
    yf = yc - r_trk * cosf

    sinbf = (sinf * xf - cosf * yf) / r_next

    alff = math.atan2(sinf, cosf)
    dphi = alff - beta
    alrphi = r_trk * dphi
    # zf = param[1] + 1 / tan(param[2]) * r_trk * dphi
    zf2 = param[1] + 1 / math.tan(param[2]) * r_trk * dphi

    phif = math.atan2(yf, xf)

    param_next = np.inf * np.ones((5, 1))

    # param_next[0] = np.fmod(param[0] + phif - np.pi, np.pi)

    param_next[0] = param[0] + phif


    param_next[1] = zf2
    param_next[2] = param[2]
    param_next[3] = alff - phif
    param_next[4] = param[4]
    return param_next.T




if __name__ == "__main__":
    # 例子
    # phi = 0
    # z = 0
    # theta = 1e-3
    # beta = 0.2
    # kappa = 0.00002

    phi = 0.1
    z = -0.2
    theta = math.acos(1e-3)
    beta = 0.0465
    kappa = 0.06

    # param = [phi, z, theta, beta, kappa]
    phi = np.fmod(-4.39822971502571 + 2*np.pi, 2*np.pi)
    param = [0.0, 0, 1.5607961601207294, phi, 0.0011400558261971832]
    r_now = 0  # 设置实际值
    r_next = 5  # 设置实际值

    dermat_result = calculate_dermat(param, r_now, r_next)
    print(dermat_result)
