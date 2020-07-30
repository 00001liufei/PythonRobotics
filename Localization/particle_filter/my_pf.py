"""
粒子滤波

练习：
    1、权重计算
    2、粒子采样

"""

import math
import matplotlib.pyplot as plt
import numpy as np

# 样本采样时的方差
Q = np.diag([0.2]) ** 2  # range error
R = np.diag([2.0, np.deg2rad(40.0)]) ** 2  # input error

# 预估的噪声方差
Q_sim = np.diag([0.2]) ** 2
R_sim = np.diag([1.0, np.deg2rad(30.0)]) ** 2

# 时间步长
DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range

# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # 有效粒子数，低于阈值时重采样

show_animation = True
# show_animation = False # DEBUG

def calc_input():
    v = 1.0
    yaw_rate = 0.1
    u = np.array([[v], [yaw_rate]])
    return u


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u
    return x


def observation(x_true, xd, u, rf_id):
    x_true = motion_model(x_true, u)  # 真实轨迹

    # 带噪声的控制量下状态估计
    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5
    ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5

    ud = np.array([[ud1], [ud2]])

    xd = motion_model(xd, ud)  # dead reckoning

    # 观测模型，产生权重
    z = np.zeros((0, 3))  # 全局变量，由GPS测量的数据

    # 由四个点的距离误差，表达的定位信息
    for i in range(len(rf_id)):
        dx = x_true[0, 0] - rf_id[i, 0]
        dy = x_true[1, 0] - rf_id[i, 1]
        d = math.hypot(dx, dy)
        if d < MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # 带噪声的距离误差
            zi = np.array([[dn, rf_id[i, 0], rf_id[i, 1]]])  # 距离误差，对应的固定点坐标
            z = np.vstack((z, zi))

    return x_true, z, xd, ud

def gauss_likelihood(x, sigma):
    p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))
    return p

def calc_covariance(x_est, px, pw):
    """
    calculate covariance matrix
    see ipynb doc
    """
    cov = np.zeros((3, 3))
    n_particle = px.shape[1]
    for i in range(n_particle):
        dx = (px[:, i:i + 1] - x_est)[0:3]
        cov += pw[0, i] * dx @ dx.T
    cov *= 1.0 / (1.0 - pw @ pw.T)

    return cov

def re_sampling(px, pw):
    """
    low variance re-sampling
    """

    w_cum = np.cumsum(pw)
    base = np.arange(0.0, 1.0, 1 / NP)
    re_sample_id = base + np.random.uniform(0, 1 / NP)
    indexes = []
    ind = 0
    for ip in range(NP):
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)

    px = px[:, indexes]
    pw = np.zeros((1, NP)) + 1.0 / NP  # init weight

    return px, pw



def pf_localization(px, pw, z, u):
    # 计算下一时刻的采样点
    for i in range(NP):
        # 上一时刻的采样点，及对应权重
        x = np.array([px[:,i]])
        w = pw[0, i]

        # 随机采样，下一时刻的控制量
        ud1 = u[0,0] + np.random.randn() * R[0, 0] ** 0.5
        ud2 = u[1,0] + np.random.randn() * R[1,1] ** 0.5
        ud = np.array([[ud1], [ud2]])
        x = motion_model(x, ud) # 当前时刻，建议分布下采样点

        # 计算权重
        for j in range(len(z[:, 0])):
            dx = x[0,0] - z[j, 1]
            dy = x[1,0] - z[j, 2]
            prez = math.hypot(dx,dy)
            dz = prez - z[j,0]
            w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0])) 

        px[:, i] = x[:, 0]
        pw[0, i] = w

    # 归一化
    pw = pw / pw.sum()  # normalize

    # 后验分布
    x_est = px @ pw
    p_est = calc_covariance(x_est, px, pw)

    # 是否需要重新采样
    N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # 权重大的粒子数量
    if N_eff < NTh:
        # 重新采样
        px, pw = re_sampling(px, pw)

    return x_est, p_est, px, pw



def main():
    print(__file__ + " start!!!")

    # RF_ID positions [x, y]
    # 4个固定点，GPS的定位模型
    rf_id = np.array([[10.0, 0.0],
                      [10.0, 10.0],
                      [0.0, 15.0],
                      [-5.0, 20.0]])

    time = 0.0

    x_est = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))

    x_dr = np.zeros((4, 1))  # Dead reckoning
    px = np.zeros((4, NP)) # 样本点
    pw = np.zeros((1, NP)) + 1.0 / NP # 样本点权重

    # 历史数据存储
    h_x_est = x_est
    hxdr = x_dr
    hxTrue = xTrue

    while time < SIM_TIME:
        time += DT

        u = calc_input()

        xTrue, z, x_dr, ud = observation(xTrue, x_dr, u, rf_id)

        # 粒子滤波
        x_est, PEst, px, pw = pf_localization(px, pw, z, ud)

        h_x_est = np.hstack((h_x_est, x_est))
        hxdr = np.hstack((hxdr, x_dr))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            for i in range(len(z[:, 0])):
                plt.plot([xTrue[0, 0], z[i, 1]], [xTrue[1, 0], z[i, 2]], "-k")
            plt.plot(rf_id[:, 0], rf_id[:, 1], "*k")
            plt.plot(px[0, :], px[1, :], ".r")
            plt.plot(np.array(hxTrue[0, :]).flatten(),
                     np.array(hxTrue[1, :]).flatten(), "-b")
            plt.plot(np.array(hxdr[0, :]), np.array(hxdr[1, :]), "-k")
            plt.plot(np.array(h_x_est[0, :]).flatten(),
                     np.array(h_x_est[1, :]).flatten(), "-r")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

if __name__ == '__main__':
    main()
