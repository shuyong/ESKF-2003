# ESKF-2003
An implementation of the Error State Kalman Filter (ESKF)
* * *

# Description
The ambition of this repository is to learn how to make an estimator that can integrate 9-DOF IMU (accelerometer/gyro/magnetometer) into a quaternion attitude.

The theory of the Error-State Kalman Filter described in:
* “Attitude Error Representations for Kalman Filtering - 2003” - F. Landis Markley

# References
01. [Circumventing Dynamic Modeling: Evaluation of the Error-State Kalman Filter applied to Mobile Robot Localization - 1999](https://www.academia.edu/13385785/Circumventing_dynamic_modeling_Evaluation_of_the_error-state_kalman_filter_applied_to_mobile_robot_localization)
    + 规避动态建模：应用于移动机器人定位的误差状态卡尔曼滤波器的评价

02. [Attitude Error Representations for Kalman Filtering - 2002](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20020060647.pdf)
    + [Attitude Error Representations for Kalman Filtering - 2003](https://www.researchgate.net/publication/245432681_Attitude_Error_Representations_for_Kalman_Filtering)
    + 卡尔曼滤波的姿态误差表示

03. [Attitude estimation or quaternion estimation? - 2003](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20030093641.pdf)
    + 姿态估计或四元数估计

04. [Multiplicative vs. Additive Filtering for Spacecraft Attitude Determination - 2004](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20040037784.pdf)
    + [Multiplicative vs. Additive Filtering for Spacecraft Attitude Determination](https://www.researchgate.net/publication/260347976_Multiplicative_vs_Additive_Filtering_for_Spacecraft_Attitude_Determination)
    + 航天器姿态确定的乘法与加法滤波器的对比

05. [Indirect Kalman filter for 3D attitude estimation - 2007](http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf)
    + 三维姿态估计的间接卡尔曼滤波
