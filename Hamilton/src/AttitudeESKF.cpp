/*
 * AttitudeESKF.cpp
 *
 *  Copyright (c) 2019 Shuyong Chen. Apache 2 License.
 *
 *  This file is part of attitude_eskf.
 *
 *	Created on: 12/14/2019
 *		  Author: shuyong
 */

#ifndef NDEBUG
#define NDEBUG
#endif

#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include "AttitudeESKF.hpp"

/* Cited sources
 * [1]. "Attitude Error Representations for Kalman Filtering", F. Landis Markley, JPL
 * [2]. "Indirect Kalman Filter for 3D Attitude Estimation", N. Trawny & S. Roumeliotis
 * [3]. "Quaternion kinematics for the error-state Kalman filter", J. Sola
 * [4]. "Fast Quaternion Attitude Estimation from two vector measurements", F. Landis Markley
 */

using namespace Eigen;

namespace AttitudeEKF {

// skew symmetric matrix
template <typename Scalar, typename Derived>
Eigen::Matrix<Scalar, 3, 3> toCrossMatrix(const Eigen::DenseBase<Derived> &v)
{
    Eigen::Matrix<Scalar, 3, 3> crossMatrix;
    // eq. 5
    crossMatrix << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
    return crossMatrix;
}

template <typename Scalar, typename Derived>
Eigen::Matrix<Scalar, 3, 3> toRotationMatrix(const Eigen::MatrixBase<Derived> &v)
{
    Eigen::Matrix<Scalar, 3, 3> rotationMatrix;
    // eq. 20
    Scalar norm_sqr = v(0) * v(0) + v(1) * v(1) + v(2) * v(2);
    Eigen::Matrix<Scalar, 3, 3> vvT = v * v.transpose();
    rotationMatrix = Eigen::Matrix<Scalar, 3, 3>::Identity() + toCrossMatrix<Scalar>(v) - 0.5 * (norm_sqr * Eigen::Matrix<Scalar, 3, 3>::Identity() - vvT);
    return rotationMatrix;
}

// hardcoded 3x3 invert (unchecked)
template <typename T>
static inline Matrix<T, 3, 3> invert(const Matrix<T, 3, 3> &A, T det) {
  Matrix<T, 3, 3> C;
  det = 1 / det;

  C(0, 0) = (-A(2, 1) * A(1, 2) + A(1, 1) * A(2, 2)) * det;
  C(0, 1) = (-A(0, 1) * A(2, 2) + A(0, 2) * A(2, 1)) * det;
  C(0, 2) = ( A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1)) * det;

  C(1, 0) = ( A(2, 0) * A(1, 2) - A(1, 0) * A(2, 2)) * det;
  C(1, 1) = (-A(2, 0) * A(0, 2) + A(0, 0) * A(2, 2)) * det;
  C(1, 2) = ( A(1, 0) * A(0, 2) - A(0, 0) * A(1, 2)) * det;

  C(2, 0) = (-A(2, 0) * A(1, 1) + A(1, 0) * A(2, 1)) * det;
  C(2, 1) = ( A(2, 0) * A(0, 1) - A(0, 0) * A(2, 1)) * det;
  C(2, 2) = (-A(1, 0) * A(0, 1) + A(0, 0) * A(1, 1)) * det;

  return C;
}

// Eigen does not define these operators, which we use for integration
template <typename Scalar>
static inline Eigen::Quaternion<Scalar> operator + (const Eigen::Quaternion<Scalar>& a,
                                      const Eigen::Quaternion<Scalar>& b) {
  return Eigen::Quaternion<Scalar>(a.w() + b.w(),
                                   a.x() + b.x(),
                                   a.y() + b.y(),
                                   a.z() + b.z());
}

template <typename Scalar>
static inline Eigen::Quaternion<Scalar> operator * (const Eigen::Quaternion<Scalar>& q,
                                      Scalar s) {
  return Eigen::Quaternion<Scalar>(q.w() * s,
                                   q.x() * s,
                                   q.y() * s,
                                   q.z() * s);
}

template <typename Scalar>
static inline Eigen::Quaternion<Scalar> operator * (Scalar s, const Eigen::Quaternion<Scalar>& q) {
  return Eigen::Quaternion<Scalar>(q.w() * s,
                                   q.x() * s,
                                   q.y() * s,
                                   q.z() * s);
}

/**
 *  @brief Integrate a rotation quaterion using Euler integration
 *  @param q Quaternion to integrate
 *  @param w Angular velocity (body frame), stored in 3 complex terms
 *  @param dt Time interval in seconds
 */
template <typename Scalar>
static inline void integrateEuler(Eigen::Quaternion<Scalar>& q, const Eigen::Quaternion<Scalar>& w, Scalar dt) {
  // eq.(23)
  q = q + static_cast<Scalar>(0.5) * (q * w * dt);

  q.normalize();
}

/**
 *  @brief Integrate a rotation quaternion using 4th order Runge Kutta
 *  @param q Quaternion to integrate
 *  @param w Angular velocity (body frame), stored in 3 complex terms
 *  @param dt Time interval in seconds
 */
template <typename Scalar>
static inline void integrateRungeKutta4(Eigen::Quaternion<Scalar>& q, const Eigen::Quaternion<Scalar>& w, Scalar dt) {
  const static Scalar half = static_cast<Scalar>(0.5);
  const static Scalar two  = static_cast<Scalar>(2.0);
  const static Scalar six  = static_cast<Scalar>(6.0);

  Eigen::Quaternion<Scalar> qw = half * w *  q;
  Eigen::Quaternion<Scalar> k2 = half * w * (q + qw * dt * half);
  Eigen::Quaternion<Scalar> k3 = half * w * (q + k2 * dt * half);
  Eigen::Quaternion<Scalar> k4 = half * w * (q + k3 * dt);

  q = q + (qw + k2 * two + k3 * two + k4) * (dt / six);

  q.normalize();
}

/**
 *  @brief Rodrigues Parameters
 *  @param w Angular velocity (body frame), Angle-Axis
 *  @return Rotation Matrix
 */
template <typename Scalar>
static inline Eigen::Matrix<Scalar,3,3> 
rodrigues(const Eigen::Matrix<Scalar,3,1>& w) {
  const auto norm = w.norm();	// Angle
  if (norm < std::numeric_limits<Scalar>::epsilon()*10) {
    return Eigen::Matrix<Scalar,3,3>::Identity() + crossSkew(w);
  }
  return Eigen::AngleAxis<Scalar>(norm, w / norm).matrix();
}

///////////////////////////////////////////////////////////////////////

#ifndef SGN
#define SGN(X) ((X > 0) - (X < 0))
#endif

const Eigen::Matrix<AttitudeESKF::scalar_t, 3, 3> AttitudeESKF::I3 = Eigen::Matrix<AttitudeESKF::scalar_t, 3, 3>::Identity();

AttitudeESKF::AttitudeESKF()
    : q_ref_(1,0,0,0), isStable_(true) {
  initialize();
}

void AttitudeESKF::predict(AttitudeESKF::scalar_t dt) {

  if (dt == 0) return;

  // integrate state and covariance
  // [ 0, x, y, z ]
  quat wQuat(0, w_ref_[0], w_ref_[1], w_ref_[2]);
#if 1
  integrateEuler(q_ref_, wQuat, dt);
#else
  integrateRungeKutta4(q_ref_, wQuat, dt);
#endif

  // transition matrix
  F_.setZero();
  //     [ Θ      Ψ     ]
  // F = [              ]
  //     [ 0_3×3  I_3×3 ]
  //

  // rotation matrix
  mat3 w_cross = toCrossMatrix<scalar_t>(w_ref_ * dt);
  // error-state Jacobian
  F_00_ =  I3 - w_cross + 0.5 * w_cross * w_cross;
  F_01_ = -I3 * dt + 0.5 * dt * w_cross + 1.0/6.0 * dt  * w_cross * w_cross;
  //F_10_.setZero();
  // bias Jacobian
  F_11_ = I3;

  // P = F P F'
  P_ = F_ * P_ * F_.transpose();

  // noise jacobian

  Q_11_ = sigma_gyro_ * sigma_gyro_ * dt * I3 + sigma_gyro_drift_ * sigma_gyro_drift_ * dt * (I3 * 1.0/3.0 * dt * dt + 2.0/120.0  * dt * dt * w_cross * w_cross);
  Q_12_ = -sigma_gyro_drift_ * sigma_gyro_drift_ * (I3 * 0.5 * dt * dt - 1.0/6.0 * dt * dt * w_cross + 1.0/24.0 * dt * dt * w_cross * w_cross);
  Q_21_	= Q_12_.transpose();
  Q_22_ = sigma_gyro_drift_ * sigma_gyro_drift_ * dt * I3;

  // P = F P F' + Q
  P_ = P_ + Q_;

  // second-order term correct angular velocity
  w_c_[0] = 0.5 * (P_c_(2, 1) - P_c_(1, 2));
  w_c_[1] = 0.5 * (P_c_(0, 2) - P_c_(2, 0));
  w_c_[2] = 0.5 * (P_c_(1, 0) - P_c_(0, 1));

}

// Time Update (“Predict”)
void AttitudeESKF::timePropagation(const AttitudeESKF::vec3& wb,
                           AttitudeESKF::scalar_t dt) {

  // The angular rate vector of gyro output.
  // Uncorrected gyroscope readings in body frame.
  w_out_ = wb;
  // The estimated angular velocity ω
  // true gyro reading
  // eq.(27), eq.(30) & eq.(58)
  w_ref_ = w_out_ - b_ - w_c_;

  predict(dt);
  // P = F P F' + Q

}

// Measurement Update (“Correct”)
void AttitudeESKF::measurementUpdateWithVector(const AttitudeESKF::mat3& R) {
  // predicted reference vector
  // eq.(20)
  // FIXME: maybe A_a == I, because after reset, a_ == 0;
  mat3 A_a = toRotationMatrix<scalar_t>(a_);
  // eq.(42)
  mat3 A_q = A_a * q_ref_.toRotationMatrix();
  // rotation from Inertial frame to Body frame
  v_B_ = A_q.transpose() * v_I_;

  // calculate jacobian
  // eq.(45)
  // H = [ H_a  0_3×3 ]
  H_.setZero();
  // eq.(44)
  H_a_ = toCrossMatrix<scalar_t>(v_B_);

  // ???eq.(70)
  //scalar_t p_a = 0.001;
  //scalar_t p_a = 0.12;

  // predicted gravity vector
  // eq.(43)
  // FIXME: maybe H_a_ * a_ == 0, because after reset, a_ == 0;
  // see section "3.5 reset".
  h_pred(v_B_) = h_pred(v_B_) + H_a_ * a_;
  //h_pred(v_B_) = h_pred(v_B_) * (1.0 - p_a) + H_a_ * a_;

  // The residual is the difference between a measurement and
  // the value predicted by the filter. Compute the residual 
  // between the predicted and actual measurement to assess 
  // how well the filter is performing and converging. 
  //
  // The formula for the updated (a posteriori) estimate covariance
  // is valid for the optimal K gain that minimizes the residual error,
  // in which form it is most widely used in applications.
  //
  // Innovation (or pre-fit residual) covariance.
  // The covariance of the residual.
  mat3 S = H_a_ * P_a_ * H_a_.transpose() + R;
  // Innovation or measurement pre-fit residual.
  vec3 z_res = h_obs_ - h_pred(v_B_);

  Eigen::FullPivLU<mat3> LU(S);
  isStable_ = LU.isInvertible();

  if (!isStable_) {
    return;
  }

  // solve for the kalman gain
  // The Kalman gain matrix.
  // eq.(46)
  K_ = P_col_ * H_a_.transpose() * S.inverse();

  // correct state vector
  // eq.(47)
  x_ = x_ + K_ * z_res;
  // x_ = [ a_  b_ ]
  //
  // a_ = K * z_residual
  // because after reset : a_ == 0;
  // see section "3.5 reset".

  // eq.(48)
  P_ -= K_ * H_a_ * P_row_;

  // reset
  reset();
}

void AttitudeESKF::reset() {
  // see section "3.5 reset".
  // eq.(18b), eq.(19)
  auto norm_sqr = a_[0] * a_[0] + a_[1] * a_[1] + a_[2] * a_[2];
  auto w = 1.0 - norm_sqr / 8.0;
  dq_ = quat(w, a_[0] / 2.0, a_[1] / 2.0, a_[2] / 2.0);
  // eq.(13) q = δq(a) ⊗ q_ref
  q_ref_ = q_ref_ * dq_;
  // eq.(22)
  q_ref_.normalize();
  a_.setZero();
}

// Measurement Update (“Correct”)
void AttitudeESKF::measurementUpdateAcc(const AttitudeESKF::vec3& ab,
                           AttitudeESKF::scalar_t dt) {
  predict(dt);
  // P = F P F' + Q

  h_obs_ = ab;

  vec3 gravity;
  gravity[0] = 0.0;
  gravity[1] = 0.0;
  gravity[2] = kOneG;

  v_I_ = gravity;

  // predicted gravity vector
  measurementUpdateWithVector(R_acc_);
}

// Measurement Update (“Correct”)
void AttitudeESKF::measurementUpdateMag(const AttitudeESKF::vec3& mb,
                           AttitudeESKF::scalar_t dt) {

  predict(dt);
  // P = F P F' + Q

  h_obs_ = mb;

  /* 
   * 北京地区地磁数据:
   * N: 27913.8 nT
   * E: -3312.1 nT
   * D: 46771.9 nT
   *
   * 地磁数据很容易受外界影响。只选择水平面指向北方的数据即可。填写'1'也可。
   * 要想精确估计 yaw，还需要 GPS 等其它数据校正。
   */
  vec3 m_field;
  m_field[0] = 0.0;
  m_field[1] = 0.279138;
  m_field[2] = 0.0;

  v_I_ = m_field;

  // predicted m-field vector
  measurementUpdateWithVector(R_mag_);
}

//void AttitudeESKF::measurementUpdatePos(const vec3& mp, scalar_t dt){}

void AttitudeESKF::measurementUpdateQuat(const vec3& ma, scalar_t dt) {
  
  predict(dt);
  // P = F P F' + Q

  // see section "3.6 Quaternion Measurements"
  a_obs_ = ma;
  // eq.(18b), eq.(19)
  auto norm_sqr = a_obs_[0] * a_obs_[0] + a_obs_[1] * a_obs_[1] + a_obs_[2] * a_obs_[2];
  auto w = 1.0 - norm_sqr / 8.0;
  dq_ = quat(w, a_obs_[0] / 2.0, a_obs_[1] / 2.0, a_obs_[2] / 2.0);

  // eq.(49)
  // q_obs = δq(a_obs) ⊗ q_ref
  q_obs_ = q_ref_ * dq_;
  q_obs_.normalize();

  // The measurement model is simply
  // eq.(50) h(a) = a
  // so that H_a is the 3x3 identity matrix
  // H_a_ = I3;
  
  mat3 S = P_a_ + R_rot_;
  vec3 z_res = a_obs_ - a_;

  Eigen::FullPivLU<mat3> LU(S);
  isStable_ = LU.isInvertible();

  if (!isStable_) {
    return;
  }

  // eq.(51)
  K_ = P_col_ * S.inverse();
  x_ = x_ + K_ * z_res;

  // eq.(48)
  P_ -= K_ * P_row_; 

  // reset
  reset();
}

void AttitudeESKF::externalYawUpdate(scalar_t yaw, scalar_t alpha) {
  // check if we are near the hover state
  const Matrix<scalar_t,3,3> wRb = q_ref_.matrix();
  Matrix<scalar_t,3,1> g;
  g[0] = 0;
  g[1] = 0;
  g[2] = 1;
  
  g = wRb.transpose() * g;
  if (g[2] > 0.85) {
    // break into roll pitch yaw
    Matrix<scalar_t,3,1> rpy = getRPY();
    // interpolate between prediction and estimate
    rpy[2] = rpy[2]*(1-alpha) + yaw*alpha;
    q_ref_ = Eigen::AngleAxis<scalar_t>(rpy[2],vec3(0,0,1)) *
             Eigen::AngleAxis<scalar_t>(rpy[1],vec3(0,1,0)) *
             Eigen::AngleAxis<scalar_t>(rpy[0],vec3(1,0,0));
  }
}

void AttitudeESKF::initialize()
{
  // q_ref(0) = [ 1, 0, 0, 0 ]
  q_ref_ = quat(1, 0, 0, 0);

  // x_0 = 0;
  x_.setZero();
#if 0
  a_.setZero();
  b_.setZero();
#endif

  // P0 = I
  P_.setIdentity();
#if 0
  P_c_.setZero();
#endif
  // start w/ a large uncertainty
  P_ *= M_PI * M_PI;
  w_c_.setZero();

  // Q = I
  // G * δt * Q * G
  Q_.setIdentity();
  // FIXME
  Q_(0, 0) = sigma_gyro_ * sigma_gyro_;
  Q_(1, 1) = sigma_gyro_ * sigma_gyro_;
  Q_(2, 2) = sigma_gyro_ * sigma_gyro_;
  Q_(3, 3) = sigma_gyro_drift_ * sigma_gyro_drift_;
  Q_(4, 4) = sigma_gyro_drift_ * sigma_gyro_drift_;
  Q_(5, 5) = sigma_gyro_drift_ * sigma_gyro_drift_;

#if 0
  // FIXME
  R_gyr_.setIdentity();
  R_gyr_(0, 0) = sigma_gyro_ * sigma_gyro_;
  R_gyr_(1, 1) = sigma_gyro_ * sigma_gyro_;
  R_gyr_(2, 2) = sigma_gyro_ * sigma_gyro_;
#endif

  R_acc_.setIdentity();
  R_acc_(0, 0) = sigma_accel_ * sigma_accel_;
  R_acc_(1, 1) = sigma_accel_ * sigma_accel_;
  R_acc_(2, 2) = sigma_accel_ * sigma_accel_;

  R_mag_.setIdentity();
  R_mag_(0, 0) = 2.5E-5;
  R_mag_(1, 1) = 2.5E-5;
  R_mag_(2, 2) = 2.5E-5;

  // FIXME
  R_pos_ = sigma_pos_ * sigma_pos_ * I3;

  R_rot_ = sigma_rot_ * sigma_rot_ * I3;

}

void AttitudeESKF::initWithAcc(const vec3& ab) {

  scalar_t ax = ab[0];
  scalar_t ay = ab[1];
  scalar_t az = ab[2];

  // see https://cache.freescale.com/files/sensors/doc/app_note/AN3461.pdf
  // rotation sequence R = Rx * Ry * Rz
  // eq. 38
  scalar_t roll = std::atan2(ay, SGN(-az) * std::sqrt(az * az + 0.01 * ax * ax));
  // eq. 37
  scalar_t pitch = std::atan(-ax / std::sqrt(ay * ay + az * az));

  scalar_t cr05 = std::cos(0.5 * roll );
  scalar_t sr05 = std::sin(0.5 * roll );
  scalar_t cp05 = std::cos(0.5 * pitch);
  scalar_t sp05 = std::sin(0.5 * pitch);

  q_ref_ = quat(cr05, sr05, 0, 0) * quat(cp05, 0, sp05, 0);
}

void AttitudeESKF::initWithAccAndMag(const vec3 &ab, const vec3 &mb) {

  scalar_t ax = ab[0];
  scalar_t ay = ab[1];
  scalar_t az = ab[2];

  // see https://cache.freescale.com/files/sensors/doc/app_note/AN3461.pdf
  // rotation sequence R = Rx * Ry * Rz
  // eq. 38
  scalar_t roll = std::atan2(ay, SGN(-az) * std::sqrt(az * az + 0.01 * ax * ax));
  // eq. 37
  scalar_t pitch = std::atan(-ax / std::sqrt(ay * ay + az * az));

  // see https://www.nxp.com/docs/en/application-note/AN4246.pdf
  // eq. 6 - 10
  scalar_t cr = std::cos(roll );
  scalar_t sr = std::sin(roll );
  scalar_t cp = std::cos(pitch);
  scalar_t sp = std::sin(pitch);

  Eigen::Matrix<scalar_t, 3, 3> RxT;
  RxT << 1,  0 , 0 ,
         0,  cr, sr,
         0, -sr, cr;
  Eigen::Matrix<scalar_t, 3, 3> RyT;
  RyT << cp, 0, -sp,
         0 , 1,  0,
         sp, 0,  cp;

  scalar_t mx = mb[0];
  scalar_t my = mb[1];
  scalar_t mz = mb[2];

  Eigen::Matrix<scalar_t, 3, 1> Bp;
  Bp << mx, my, mz;

  Eigen::Matrix<scalar_t, 3, 1> Bf;
  Bf = RyT * RxT * Bp;

  scalar_t yaw = -atan2(-Bf(1), Bf(0));

  scalar_t cr05 = std::cos(0.5 * roll );
  scalar_t sr05 = std::sin(0.5 * roll );
  scalar_t cp05 = std::cos(0.5 * pitch);
  scalar_t sp05 = std::sin(0.5 * pitch);
  scalar_t cy05 = std::cos(0.5 * yaw  );
  scalar_t sy05 = std::sin(0.5 * yaw  );

  q_ref_ = quat(cr05, sr05, 0, 0) * quat(cp05, 0, sp05, 0) * quat(cy05, 0, 0, sy05);

}
  
AttitudeESKF::vec3 AttitudeESKF::getRPY() {

  auto euler = q_ref_.toRotationMatrix().eulerAngles(0, 1, 2);
  //std::cout << "Euler from quaternion in roll, pitch, yaw"<< std::endl << euler << std::endl; 

  return euler;
}

} //  namespace AttitudeEKF

