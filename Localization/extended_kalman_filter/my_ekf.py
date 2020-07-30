"""
自己实现EKF

本模型是离散时间的非线性系统，《机器人的状态估计》91-92页

练习过程：
    1.实现真实数据点的绘制
    2.实现EKF滤波

"""
import math
import numpy as np
import matplotlib.pyplot as plt


# 过程噪声和观测噪声的方差，Q和R
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

# 控制输入的噪声和GPS噪声
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

# 离散时间步长
DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]


def calc_input():
    # 移动速度、角速度
    v = 1.0  # 移动速度 m/s
    rate = 0.1  # 旋转角速度(绕z轴旋转) rad/s
    u = np.array([[v], [rate]])
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


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x
    return z

def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF

def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH

def ekf_estimation(xEst, PEst, z, u):
    # 预测阶段
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    # 更新
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R 
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst



def main():
    print(__file__ + " start!!")

    time = 0.0

    # 状态估计
    xEst = np.zeros((4,1)) # 状态向量估计
    PEst = np.eye(4) # 状态方差估计

    # 上一时刻的真实状态
    Xtrue = np.zeros((4, 1))  # 输出列数组

    # 带噪声的控制量和上一时刻状态，产生的预测状态，有叫Dead reckoning
    xDR = np.zeros((4,1))

    # 存储历史数据
    hxTrue = Xtrue
    hz = np.zeros((2, 1))
    hxDR = Xtrue # 等于初始值
    hxEst = xEst

    while SIM_TIME >= time:
        time += DT

        # 时刻控制量
        u = calc_input()
        ud = u + INPUT_NOISE @ np.random.randn(2, 1) # 高斯噪声下的控制量

        # 下一时刻的真实状态，由上一时刻的状态和控制量决定，无噪声
        Xtrue = motion_model(Xtrue, u)
        xDR = motion_model(xDR, ud) # 噪声下的状态计算

        # 观测数据，GPS对位置的测量，带有高斯噪声
        z = observation_model(Xtrue) + GPS_NOISE @ np.random.randn(2, 1)

        # EKF估计
        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)


        # 按列存储数据
        hxTrue = np.hstack((hxTrue, Xtrue))
        hz = np.hstack((hz, z))
        hxDR = np.hstack((hxDR, xDR))
        hxEst = np.hstack((hxEst, xEst))

    plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), '-b')
    plt.plot(hz[0, :].flatten(), hz[1, :].flatten(), '.g')
    plt.plot(hxDR[0,:].flatten(), hxDR[1,:].flatten(), '-k')
    plt.plot(hxEst[0,:].flatten(), hxEst[1,:].flatten(), '-r')

    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
