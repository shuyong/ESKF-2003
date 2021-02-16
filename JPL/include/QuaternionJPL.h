// Copyright (C) 2021 Shuyong Chen <shuyong.chen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_QUATERNION_JPL_H
#define EIGEN_QUATERNION_JPL_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace Eigen { 
template<typename _Scalar>
class QuaternionJPL : public Quaternion<_Scalar>
{
public:
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef AngleAxis<Scalar> AngleAxisType;

  EIGEN_DEVICE_FUNC inline QuaternionJPL() : Quaternion<Scalar>::Quaternion() {}
  EIGEN_DEVICE_FUNC inline QuaternionJPL(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& w) : Quaternion<Scalar>::Quaternion(w, x, y, z){}
  EIGEN_DEVICE_FUNC inline QuaternionJPL(const Quaternion<Scalar>& other) : Quaternion<Scalar>::Quaternion(other) {}
  EIGEN_DEVICE_FUNC explicit inline QuaternionJPL(const AngleAxisType& aa)
  {
    Quaternion<Scalar>& q = static_cast<Quaternion<Scalar>&>(*this);
    q = aa; 
  }
  EIGEN_DEVICE_FUNC inline QuaternionJPL<Scalar> operator* (const QuaternionJPL<Scalar>& other) const
  {
    const Quaternion<Scalar>& p = static_cast<const Quaternion<Scalar>&>(*this);
    const Quaternion<Scalar>& q = static_cast<const Quaternion<Scalar>&>(other);
    return (q * p);
  }
  EIGEN_DEVICE_FUNC Matrix3 toRotationMatrix() const
  {
    const Quaternion<Scalar>& q = static_cast<const Quaternion<Scalar>&>(*this);
    return q.inverse().toRotationMatrix();
  }
};

typedef QuaternionJPL<float>  QuaternionJPLf;
typedef QuaternionJPL<double> QuaternionJPLd;

} // end namespace Eigen

#endif // EIGEN_QUATERNION_JPL_H
