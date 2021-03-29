/*
 * AttitudeESKF.hpp
 *
 *  Copyright (c) 2019 Shuyong Chen. Apache 2 License.
 *
 *  This file is part of AttitudeESKF.
 *
 *	Created on: 12/14/2019
 *		  Author: shuyong
 */

#ifndef _ATTITUDE_ESKF_H_
#define _ATTITUDE_ESKF_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "QuaternionJPL.h"

namespace AttitudeEKF {

/**
 *  @class AttitudeESKF
 *  @brief Implementation of an error-state EKF for attitude determination using
 *         quaternions.
 *  @note  Gravity and magnetic field are the supported reference vectors.
 *  @see  'Attitude Error Representations for Kalman Filtering', F. Landis
 *         Markley, JPL
 */

class AttitudeESKF {
public:
  typedef double scalar_t; /**< Type used for all calculations, change as
                              performance requires */

  /**
   * P : Covariance of Kalman filter state (P in common formulation).
   * F : Jacobian matrix of partial derivatives of f(·) with respect to the state vector.
   * G : Jacobian matrix of partial derivatives of f(·) with respect to the process noise vector.
   * Q : Covariance of the process noise (Q in common formulation).
   * H : Jacobian of the measurements (H in common formulation).
   * K : Gain of the Kalman filter (K in common formulation).
   * R : Covariance of the measurement (R in common formulation).
   * S : Covariance of innovation (S in common formulation).
   * 
   * Dimension and description of variables:
   * xk   n × 1 − State vector.
   * wk   n × 1 − Process noise vector.
   * zk   m × 1 − Observation vector.
   * vk   m × 1 − Measurement noise vector.
   * f(·) n × 1 − Process nonlinear vector function. The state-transition function.
   * h(·) m × 1 − Observation nonlinear vector function. The sensor function.
   * Qk   n × n − Process noise covariance matrix.
   * Rk   m × m − Measurement noise covariance matrix.
 */

  typedef Eigen::Matrix<scalar_t, 3, 1> vec3; /// Vector in R3
  typedef Eigen::Matrix<scalar_t, 3, 3> mat3; /// Matrix in R3
  typedef Eigen::QuaternionJPL<scalar_t> quat;   /// Member of S4

  static const int Nsta = 2;	// state values
  static const int Mobs = 3;	// measurements
  static const int V_sz = 3;	// 3-component vector
  static const int ag_i = 0;	// state: index of gibbs vector
  static const int bs_i = 1;	// state: index of gyro drift vector

  typedef Eigen::Matrix<scalar_t, Nsta * V_sz, 1> vec3N; // N-3x1
  typedef Eigen::Matrix<scalar_t, Mobs * V_sz, 1> vec3M; // M-3x1
  typedef Eigen::Matrix<scalar_t, Nsta * V_sz, Nsta * V_sz> mat3_NxN; // NxN
  typedef Eigen::Matrix<scalar_t, Mobs * V_sz, Mobs * V_sz> mat3_MxM; // MxM
  typedef Eigen::Matrix<scalar_t, Nsta * V_sz, Mobs * V_sz> mat3_NxM; // NxM
  typedef Eigen::Matrix<scalar_t, Nsta * V_sz, 1    * V_sz> mat3_Nx1; // Nx1
  typedef Eigen::Matrix<scalar_t, Mobs * V_sz, Nsta * V_sz> mat3_MxN; // MxN
  typedef Eigen::Matrix<scalar_t, 1    * V_sz, Nsta * V_sz> mat3_1xN; // 1xN

  // identity R3
  static const Eigen::Matrix<scalar_t, 3, 3> I3;

  static constexpr scalar_t kOneG = 9.80665;  /// Earth gravity
  static constexpr scalar_t Deg2Rad = M_PI / 180.0;
  
  /**
   *  @brief Ctor, initializes state to all zeros.
   */
  AttitudeESKF();

  void initialize();

  /**
   * @brief predict Perform the prediction step.
   * @param wb Uncorrected gyroscope readings in body frame.
   * @param dt Time step in seconds.
   * FIXME useRK4 If true, use RK4 integration - otherwise euler is used.
   */
  void timePropagation(const vec3& wb, scalar_t dt);

  /**
   * @brief update Perform the update step with accelerometer.
   * @param ab Accelerometer reading in body frame (units of m/s^2).
   * 
   */
  void measurementUpdateAcc(const vec3& ab, scalar_t dt);

  /**
   * @brief update Perform the update step with magnetic.
   * @param mb Measured magnetic field in body frame (units of gauss)
   * 
   */
  void measurementUpdateMag(const vec3& mb, scalar_t dt);
  
  // Called when there is a new measurment from an absolute position reference.
  // Note that this has no body offset, i.e. it assumes exact observation of the center of the IMU.
  //void measurementUpdatePos(const vec3& mp, scalar_t dt);

  // Called when there is a new measurment from an absolute position reference.
  // The measurement is with respect to some location on the body that is not at the IMU center in general.
  // pos_ref_body should specify the reference location in the body frame.
  // For example, this would be the location of the GPS antenna on the body.
  // NOT YET IMPLEMENTED
  // void measurementUpdatePosWithOffset(const vec3& mp, scalar_t dt,
  //        vec3& pos_ref_body);

  // Called when there is a new measurment from an absolute orientation reference.
  // The uncertianty is represented as the covariance of a rotation vector in the body frame
  void measurementUpdateQuat(const vec3& ma, scalar_t dt);

  /**
   * @brief externalYawUpdate Update the z-axis rotation with an external
   * value.
   * @param yaw Yaw angle as the measurement.
   * @param alpha Interpolation fraction between current and external value.
   *
   * @note This method was added for use in Vicon, where only one external
   * angle is used. A complementary filter action is used for this update.
   *
   * The update only takes place if the dip angle is ~ below 30 degrees.
   */
  void externalYawUpdate(scalar_t yaw, scalar_t alpha = 0.5);
  
  /**
   * @brief initWithAccAndMag Initialize the pose.
   * 
   * @param ab Measured body acceleration (units of m/s^2).
   * @param mb Measured magnetic field in body frame (units of gauss).
   * 
   * @note Uses non-linear least squares to formulate initial rotation vector.
   * The inverse covariance matrices are used to weight the input vectors.
   * 
   * @note If magnetometer is disabled, a roll and pitch angles are determined
   * from the gravity vector. The yaw angle is zeroed.
   * 
   */
  void initWithAccAndMag(const vec3& ab, const vec3& mb);
  
  /**
   * @brief initWithAcc Initialize the pose.
   * 
   * @param ab Measured body acceleration (units of m/s^2).
   * 
   * @note Uses non-linear least squares to formulate initial rotation vector.
   * The inverse covariance matrices are used to weight the input vectors.
   * 
   * @note Because magnetometer is disabled, a roll and pitch angles are 
   * determined from the gravity vector. The yaw angle is zeroed.
   * 
   */
  void initWithAcc(const vec3& ab);
  
  /**
   * @brief getQuat Get the state as a quaternion.
   * @return Instance of quaternion.
   */
  const quat& getQuat() const { return q_ref_; }

  /**
   * @brief getAngularVelocity Get angular velocity (corrected for bias).
   * @return Angular velocity in rad/s.
   */
  const vec3& getAngularVelocity() const { return w_ref_; }

  /**
   * @brief getCovariance Get the system covariance on the error state.
   * @return NxN 3x3-block covariance matrix.
   */
  const mat3_NxN& getCovariance() const { return P_; }
  mat3_NxN& getCovariance() { return P_; }
  
#if 0
  /**
   * @brief getPredictedField Get the predicted magnetic field.
   * @return The predicted magnetic field for the current state, units of gauss.
   */
  const vec3 &getPredictedField() const { return predMag_; }
#endif

  /**
   * @brief isStable Determine if the filter is stable.
   * @return True if the Kalman gain was non-singular at the last update.
   */
  bool isStable() const { return isStable_; }

  /**
   * @brief getRPY Utility function, get roll-pitch-yaw.
   * @return Rotations about the x-y-z axes.
   */
  vec3 getRPY();
  
private:

  void predict(scalar_t dt);
  void measurementUpdateWithVector(const mat3& R);
  void reset();

  // Global Orientation, unit reference quaternion
  // eq.(13) q = δq(a) ⊗ q_ref
  quat q_ref_;
  // error quaternion
  // eq.(18b) δq(a) = (1 / sqrt(4 + |a|^2))[ a, 2 ] - Gibbs Vector
  // eq.(19)  δq(a) = [ a / 2, (1 - |a|^2) / 8 ]    - 2-factor error quaternion
  // deviation between the observed and predicted attitudes
  // eq.(13)  q_ref = δq(a)     ⊗ q_ref
  // eq.(49)  q_obs = δq(a_obs) ⊗ q_ref
  quat dq_;
  // Observed Gibbs Vector
  // eq.(34) / eq.(49) and eq.(50)
  vec3 a_obs_;
  quat q_obs_;

  // The angular velocity of the reference attitude
  // eq(23)
  //           1  [ ω_ref ]
  // dq_ref = --- [       ] ⊗ q_ref
  //           2  [   0   ]
  //
  // A priori angular velocity prediction, ω_{k+1|k}
  // eq.(23) & eq.(27) ω_ref = ω
  vec3 w_ref_;
  // The estimated angular velocity ω, ω_{k|k}
  // eq.(30) ω̂ = ω_out − b̂
  vec3 w_hat_;
  // The angular rate vector of gyro output
  vec3 w_out_;
  // gyro drift / bias vector
  vec3 b_hat_;
  #define b_	(x_.block<3, 1>(bs_i * V_sz, 0))
  // The second-order term of the propagation bias correction
  // eq.(55)
  //        1  [ P_c_32 - P_c_23 ]
  // ω_c = --- [ P_c_13 - P_c_31 ]
  //        2  [ P_c_21 - P_c_12 ]
  //
  // eq.(58) ω_ref = ω - ω_c
  vec3 w_c_;
  // Attitude Error Representations
  // eq.(09) & eq.(34) Gibbs Vector
  #define a_	(x_.block<3, 1>(ag_i * V_sz, 0))
  // The Kalman filter estimates the six-component state vector
  // eq.(31) x = [ a, b ] 6x1
  vec3N x_;

  // The System covariance matrix
  // eq.(32)
  //     [ P_a   P_c ]
  // P = [           ]
  //     [ P_c'  P_b ] 6x6
  // P_a: attitude covariance matrix
  // P_b: bias covariance matrix
  // P_c: correlation covariance matrix
  //
  //         [ P_a  ]
  // P_col = [      ]
  //         [ P_c' ] 6x3
  //
  // P_row = [ P_a   P_c ] 3x6
  //
  // The propagation of the covariance matrix
  // eq.(33)
  // dP = F P + P F' + G Q G'
  mat3_NxN P_;
  #define P_a_	(P_.block<3, 3>(ag_i * V_sz, ag_i * V_sz))
  #define P_b_	(P_.block<3, 3>(bs_i * V_sz, bs_i * V_sz))
  #define P_c_	(P_.block<3, 3>(ag_i * V_sz, bs_i * V_sz))
  #define P_col_	(P_.block<Nsta * V_sz, V_sz>(0, 0))
  #define P_row_	(P_.block<V_sz, Nsta * V_sz>(0, 0))
  // Process nonlinear vector function
  // The time derivative of Gibbs vector: f(x,t) .
  // eq(37)
  // f(x) = b̂ − b − η_1 − ω_ref × a
  //
  // F_c is the system matrix.
  // eq.(38)
  //       [ -[ω_ref]×  -I_3×3 ]
  // F_c = [                   ]
  //       [  0_3×3      0_3×3 ]
  //
  // ‘Indirect Kalman Filter for 3D Attitude Estimation’
  // F is the transition matrix, the Jacobian matrix of partial derivatives of f() with respect to state vector x.
  // eq.(187)
  //     [ Θ      Ψ     ]
  // F = [              ]
  //     [ 0_3×3  I_3×3 ]
  //
  // eq.(193)
  // Θ =  I_3×3 − ∆t * [ω_ref]× + ∆t^2 / 2  * [ω_ref]× * [ω_ref]×
  // eq.(201)
  // Ψ = -I_3×3 * ∆t + ∆t^2 / 2  * [ω_ref]× - ∆t^3 / 6  * [ω_ref]× * [ω_ref]×
  //
  mat3_NxN F_;
  #define F_00_	F_.block<3,3>(0, ag_i * V_sz)
  #define F_01_	F_.block<3,3>(0, bs_i * V_sz)
  #define F_10_	F_.block<3,3>(3, ag_i * V_sz)
  #define F_11_	F_.block<3,3>(3, bs_i * V_sz)
  // Threshold for which values of |ω̂| we will use approximate expressions for when |ω̂|→0 . (0.06°)
  scalar_t w_threshold_ = 0.001;

  // Vector Measurement
  vec3 v_B_;
  // Reference vector, representation in the inertial reference frame by the attitude matrix
  vec3 v_I_;
  // Observation nonlinear vector function: h()
  // eq.(43)
  // Independent measurement. Only one measurement is processed at one time.
  //
  // h(v_B) = h(v̄_B) + H_a × a
  //
  #define h_pred(x)	x
  // H_a is the Jacobian matrix of partial derivatives of h() with respect to Gibbs Vector x. 
  // eq.(44)
  //        ∂h
  // H_a = ---- [ v_B_ ]×
  //        ∂a
  //
  #define H_a_	H_.block<3, 3>(0, 0)
  #define H_m_	H_.block<3, 3>(0, 0)
  // H is the Jacobian matrix of partial derivatives of h() with respect to state vector x. 
  // eq.(45)
  // H = [ H_a  0_3×3 ]
  // or
  // H = [ H_m  0_3×3 ]
  //
  // Independent measurement. Only one measurement is processed at one time.
  mat3_1xN  H_;
  // The Kalman gain matrix. 
  // eq.(46)
  //     [ P_a  ]                              -1
  // K = [      ] H_a' [ H_a * P_a * H_a' + R ]
  //     [ P_c' ]
  //
  // Independent measurement. Only one measurement is processed at one time.
  mat3_Nx1 K_;
  // R is the covariance of the measurement white noise.
  // Independent measurement.
  mat3 R_acc_;
  mat3 R_mag_;
  //mat3 R_gyr_;
  // the covariance of the error position vector.
  mat3 R_pos_;
  // the covariance of the error angle vector.
  mat3 R_rot_;
  scalar_t sigma_accel_ = 0.00124; // [m/s^2]  (value derived from Noise Spectral Density in datasheet)
  scalar_t sigma_accel_drift_ = 0.001f * sigma_accel_; // [m/s^2 sqrt(s)] (Educated guess, real value to be measured)
  scalar_t sigma_gyro_ = 0.276; // [rad/s] (value derived from Noise Spectral Density in datasheet)
  scalar_t sigma_gyro_drift_ = 0.001 * sigma_gyro_; // [rad/s sqrt(s)] (Educated guess, real value to be measured)
  scalar_t sigma_pos_ = 0.003; // [m]
  scalar_t sigma_rot_ = 0.03; // [rad]
  /* process noise covariance */
  // ‘Indirect Kalman Filter for 3D Attitude Estimation’, 2.5.2
  // eq.(183)
  //       [ σ_a * σ_a * I_3×3  0_3×3             ]
  // Q_c = [                                      ]
  //       [ 0_3×3              σ_b * σ_b * I_3×3 ]
  //
  // eq.(208)
  //       [ Q_11  Q12 ]
  // Q_d = [           ]
  //       [ Q_21  Q22 ]
  // 
  mat3_NxN Q_;
  #define Q_11_	(Q_.block<3, 3>(ag_i * V_sz, ag_i * V_sz))
  #define Q_12_	(Q_.block<3, 3>(ag_i * V_sz, bs_i * V_sz))
  #define Q_21_	(Q_.block<3, 3>(bs_i * V_sz, ag_i * V_sz))
  #define Q_22_	(Q_.block<3, 3>(bs_i * V_sz, bs_i * V_sz))
  // eq.(47), h_obs is the measurement vector (z) and the predicted value is given by the preupdate expectation of eq.(43).
  // x = x + K [ h_obs - h(v_B) - H_a * a ]
  vec3 h_obs_;
  // The covariance update is eq.(48)
  // P = P - K H_a [ P_a  P_c ]

  bool isStable_;

public:

  /**
   *  @brief getCorrection Get the last correction (error state) generated.
   *  @return The previous error state.
   */
  vec3 getCorrection() const { return a_; }

  /**
   * @brief getGyroBias Get the current gyro bias estimate.
   * @return Gyro bias in units of rad/s.
   */
  vec3 getGyroBias() const { return b_; }

};

} // namespace AttitudeEKF

#endif /* defined(_ATTITUDE_ESKF_H_) */
