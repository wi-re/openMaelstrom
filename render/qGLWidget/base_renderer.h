#pragma once
#include <utility/mathv2/vector_types.h>
#ifndef NO_QT
#include <QOpenGLFunctions_3_3_Core>
/** Polymorphic base class which should be used for all rendering functions as
 * all instances that way can be stored inside of a vector of the openGL
 * widget. **/
class Renderer : protected QOpenGLFunctions_3_3_Core {
public:
  virtual void update() = 0;
  virtual void render() = 0;
  virtual bool valid() = 0;
  virtual void toggle(){};
};
#endif

/// A simple 4x4 Matrix to use with cuda, glm::mat4 tends to be overly slow.
struct Matrix4x4 {
  float data[4 * 4] = { ///< Data, soted column major
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  /// Default constructor, initialized as all zero
  constexpr __device__ __host__ Matrix4x4() {}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Accessor for the matrix
  ///
  /// @param	row   	Row index
  /// @param	column	Column index
  ///
  /// @return	The value of the matrix at (row, column)
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  constexpr __device__ __host__ float &get(int row, int column) {
    return data[row * 4 + column];
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Accessor for the matrix
  ///
  /// @param	row   	Row index
  /// @param	column	Column index
  ///
  /// @return	The value of the matrix at (row, column)
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  constexpr __device__ __host__ float &operator()(int row, int column) {
    return data[row * 4 + column];
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Multiplies a vector with this matrix
  ///
  /// @param	rhs	The vector that should be multiplied with
  ///
  /// @return	The result of the multiplication
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  constexpr __device__ __host__ float4 operator*(const float4 &rhs) {
    float4 res{rhs.x * get(0, 0) + rhs.y * get(0, 1) + rhs.z * get(0, 2) +
                   rhs.w * get(0, 3),
               rhs.x * get(1, 0) + rhs.y * get(1, 1) + rhs.z * get(1, 2) +
                   rhs.w * get(1, 3),
               rhs.x * get(2, 0) + rhs.y * get(2, 1) + rhs.z * get(2, 2) +
                   rhs.w * get(2, 3),
               rhs.x * get(3, 0) + rhs.y * get(3, 1) + rhs.z * get(3, 2) +
                   rhs.w * get(3, 3)};
    return res;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Copy constructs a matrix from another matrix
  ///
  /// @param	rhs	The matrix to be used for initialization
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  constexpr __device__ __host__ Matrix4x4(const Matrix4x4 &rhs) {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        data[i * 4 + j] = rhs.data[i * 4 + j];
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Creates a 4x4 Matrix with 1 for the diagonal elements and 0 everywhere
  /// else.
  ///
  /// @return	Returns a 4x4 Identity Matrix
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  constexpr __device__ __host__ static Matrix4x4 identity() {
    Matrix4x4 m;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        if (i == j)
          m.get(i, j) = 1.f;
    return m;
  }
};

struct DeviceCamera {
  __host__ __device__ constexpr DeviceCamera() {}
  float2 resolution{1920, 1080};     ///< Resolution of the viewport/image
  float3 position = {0.f, 0.f, 0.f}; ///< Camera position ( eye position )
  float3 view = {1.f, 0.f, 0.f};     ///< Viewing direction
  float3 up = {0.f, 0.f, 1.f};       ///< Up direction
  float2 fov = {24.f, 32.f};         ///< Fov of the camera in x and y
  float apertureRadius = 0.f;        ///< Aperture for DOF, unused currently
  float focalDistance = 1.f; ///< Focal distance for DOF, unused currently

   Matrix4x4 ViewInverse{}; ///< Inverted view matrix, generated on host side
   Matrix4x4 PerspInverse{}; ///< Inverted perspective matrix, generated on host side
   Matrix4x4 MVP{}; ///< VP matrix, generated on host side
};

struct SceneInformation {
  __device__ __host__ constexpr SceneInformation(){};
  DeviceCamera m_camera{};
  float width = 1920;
  float height = 1080;
};

extern SceneInformation& hostScene();