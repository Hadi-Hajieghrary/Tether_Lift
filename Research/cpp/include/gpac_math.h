#pragma once

/// @file gpac_math.h
/// @brief GPAC geometric math utilities for SO(3) and S² operations.
///
/// Implements the mathematical foundations from the GPAC proposal:
///   - Hat/Vee maps for so(3)
///   - S² projection and error functions
///   - SO(3) attitude error functions
///   - Desired rotation construction
///
/// References:
///   - Lee et al. (2010): Geometric control on SO(3)
///   - GPAC Proposal Section 3: Mathematical preliminaries

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace quad_rope_lift {
namespace gpac {

// =============================================================================
// CONSTANTS
// =============================================================================

constexpr double kEpsilon = 1e-10;
constexpr double kSmallAngle = 1e-6;

// =============================================================================
// BASIC SO(3) OPERATIONS
// =============================================================================

/// @brief Hat map: ℝ³ → so(3) (skew-symmetric matrix)
/// Hat(a) * b = a × b
inline Eigen::Matrix3d Hat(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S <<     0.0, -v(2),  v(1),
         v(2),    0.0, -v(0),
        -v(1),  v(0),    0.0;
  return S;
}

/// @brief Vee map: so(3) → ℝ³ (inverse of Hat)
inline Eigen::Vector3d Vee(const Eigen::Matrix3d& S) {
  return Eigen::Vector3d(S(2, 1) - S(1, 2),
                         S(0, 2) - S(2, 0),
                         S(1, 0) - S(0, 1)) / 2.0;
}

// =============================================================================
// S² (UNIT SPHERE) OPERATIONS [Eq. 4, 8, 16, 18]
// =============================================================================

/// @brief Tangent space projection on S²: P(q) = I - q*q^T
inline Eigen::Matrix3d ProjectionS2(const Eigen::Vector3d& q) {
  const Eigen::Vector3d q_unit = q.normalized();
  return Eigen::Matrix3d::Identity() - q_unit * q_unit.transpose();
}

/// @brief Apply tangent space projection: P(q)*v
inline Eigen::Vector3d ProjectToTangentS2(const Eigen::Vector3d& q,
                                          const Eigen::Vector3d& v) {
  const Eigen::Vector3d q_unit = q.normalized();
  return v - v.dot(q_unit) * q_unit;
}

/// @brief Configuration error on S²: Ψ_q = 1 - q_d · q [Eq. 9]
inline double PsiS2(const Eigen::Vector3d& q, const Eigen::Vector3d& q_d) {
  return 1.0 - q.normalized().dot(q_d.normalized());
}

/// @brief Cable direction error: e_q = P(q)*q_d [Eq. 8]
inline Eigen::Vector3d CableDirectionError(const Eigen::Vector3d& q,
                                           const Eigen::Vector3d& q_d) {
  return ProjectToTangentS2(q, q_d);
}

// =============================================================================
// SO(3) ERROR FUNCTIONS [Eq. 12, 13]
// =============================================================================

/// @brief Configuration error on SO(3): Ψ_R = ½ tr(I - R_d^T R) [Eq. 13]
inline double PsiSO3(const Eigen::Matrix3d& R, const Eigen::Matrix3d& R_d) {
  const Eigen::Matrix3d R_err = R_d.transpose() * R;
  return 0.5 * (3.0 - R_err.trace());
}

/// @brief Attitude error: e_R = ½(R_d^T R - R^T R_d)^∨ [Eq. 12]
inline Eigen::Vector3d AttitudeError(const Eigen::Matrix3d& R,
                                     const Eigen::Matrix3d& R_d) {
  const Eigen::Matrix3d R_err = R_d.transpose() * R - R.transpose() * R_d;
  return 0.5 * Vee(R_err);
}

/// @brief Angular velocity error: e_Ω = Ω - R^T R_d Ω_d [Eq. 12]
inline Eigen::Vector3d AngularVelocityError(
    const Eigen::Vector3d& Omega,
    const Eigen::Matrix3d& R,
    const Eigen::Matrix3d& R_d,
    const Eigen::Vector3d& Omega_d) {
  return Omega - R.transpose() * R_d * Omega_d;
}

// =============================================================================
// DESIRED ROTATION CONSTRUCTION [Eq. 19-20]
// =============================================================================

/// @brief Construct desired rotation from thrust direction and yaw
inline Eigen::Matrix3d DesiredRotation(const Eigen::Vector3d& F_des,
                                       double psi_d) {
  const double F_norm = F_des.norm();
  if (F_norm < kEpsilon) {
    Eigen::Matrix3d R_d = Eigen::Matrix3d::Identity();
    const double cy = std::cos(psi_d);
    const double sy = std::sin(psi_d);
    R_d(0, 0) = cy;  R_d(0, 1) = -sy;
    R_d(1, 0) = sy;  R_d(1, 1) = cy;
    return R_d;
  }

  const Eigen::Vector3d b3_c = F_des / F_norm;
  const Eigen::Vector3d b1_d(std::cos(psi_d), std::sin(psi_d), 0.0);

  const Eigen::Vector3d b3_cross_b1 = b3_c.cross(b1_d);
  const double cross_norm = b3_cross_b1.norm();

  Eigen::Vector3d b2_c, b1_c;

  if (cross_norm < kEpsilon) {
    const Eigen::Vector3d fallback = (std::abs(b3_c(0)) < 0.9)
                                     ? Eigen::Vector3d(1, 0, 0)
                                     : Eigen::Vector3d(0, 1, 0);
    b2_c = b3_c.cross(fallback).normalized();
    b1_c = b2_c.cross(b3_c);
  } else {
    b2_c = b3_cross_b1 / cross_norm;
    b1_c = b2_c.cross(b3_c);
  }

  Eigen::Matrix3d R_d;
  R_d.col(0) = b1_c;
  R_d.col(1) = b2_c;
  R_d.col(2) = b3_c;

  return R_d;
}

/// @brief Derivative of unit vector: d/dt (v/||v||) [Eq. 18]
inline Eigen::Vector3d UnitVectorDerivative(const Eigen::Vector3d& v,
                                            const Eigen::Vector3d& v_dot) {
  const double v_norm = v.norm();
  if (v_norm < kEpsilon) {
    return Eigen::Vector3d::Zero();
  }
  const Eigen::Vector3d v_hat = v / v_norm;
  return (v_dot - v_dot.dot(v_hat) * v_hat) / v_norm;
}

// =============================================================================
// QUATERNION UTILITIES
// =============================================================================

/// @brief Convert rotation matrix to quaternion (w ≥ 0 convention)
inline Eigen::Quaterniond RotationToQuaternion(const Eigen::Matrix3d& R) {
  Eigen::Quaterniond q(R);
  if (q.w() < 0) {
    q.coeffs() = -q.coeffs();
  }
  return q.normalized();
}

/// @brief Convert quaternion to rotation matrix
inline Eigen::Matrix3d QuaternionToRotation(const Eigen::Quaterniond& q) {
  return q.normalized().toRotationMatrix();
}

/// @brief Quaternion error for control (returns ~2x rotation vector)
inline Eigen::Vector3d QuaternionError(const Eigen::Quaterniond& q,
                                       const Eigen::Quaterniond& q_d) {
  Eigen::Quaterniond q_err = q_d.inverse() * q;
  if (q_err.w() < 0) {
    q_err.coeffs() = -q_err.coeffs();
  }
  return 2.0 * Eigen::Vector3d(q_err.x(), q_err.y(), q_err.z());
}

// =============================================================================
// ROTATION MATRIX UTILITIES
// =============================================================================

/// @brief Re-orthogonalize rotation matrix via SVD
inline Eigen::Matrix3d OrthogonalizeRotation(const Eigen::Matrix3d& R) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R_ortho = svd.matrixU() * svd.matrixV().transpose();
  if (R_ortho.determinant() < 0) {
    Eigen::Matrix3d U = svd.matrixU();
    U.col(2) = -U.col(2);
    R_ortho = U * svd.matrixV().transpose();
  }
  return R_ortho;
}

/// @brief Check if matrix is valid rotation
inline bool IsValidRotation(const Eigen::Matrix3d& R, double tol = 1e-6) {
  const Eigen::Matrix3d RtR = R.transpose() * R;
  if ((RtR - Eigen::Matrix3d::Identity()).norm() > tol) {
    return false;
  }
  return std::abs(R.determinant() - 1.0) <= tol;
}

/// @brief Exponential map: so(3) → SO(3) (Rodrigues' formula)
inline Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& omega) {
  const double theta = omega.norm();
  if (theta < kSmallAngle) {
    const Eigen::Matrix3d omega_hat = Hat(omega);
    return Eigen::Matrix3d::Identity() + omega_hat + 0.5 * omega_hat * omega_hat;
  }
  const Eigen::Vector3d axis = omega / theta;
  const Eigen::Matrix3d K = Hat(axis);
  return Eigen::Matrix3d::Identity() +
         std::sin(theta) * K +
         (1.0 - std::cos(theta)) * K * K;
}

/// @brief Logarithmic map: SO(3) → so(3)
inline Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
  const double trace = R.trace();
  const double cos_theta = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
  const double theta = std::acos(cos_theta);

  if (theta < kSmallAngle) {
    return Vee(R - Eigen::Matrix3d::Identity());
  }

  if (std::abs(theta - M_PI) < kSmallAngle) {
    Eigen::Matrix3d B = R + Eigen::Matrix3d::Identity();
    Eigen::Vector3d axis;
    if (B.col(0).norm() > 1e-6) {
      axis = B.col(0).normalized();
    } else if (B.col(1).norm() > 1e-6) {
      axis = B.col(1).normalized();
    } else {
      axis = B.col(2).normalized();
    }
    return M_PI * axis;
  }

  return (theta / (2.0 * std::sin(theta))) * Vee(R - R.transpose());
}

}  // namespace gpac
}  // namespace quad_rope_lift
