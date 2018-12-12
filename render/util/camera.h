#pragma once
#include <QOpenGLShaderProgram>
#include <QtGui/QInputEvent>
#include <render/qGLWidget/base_renderer.h>

#include <QOpenGLFunctions_3_3_Core>

/** This class represents the main camera of the simulation with a first person
 * style camera. The camera is built using a singleton instance for easier
 * access, i.e. in other renderers. The uniforms are created by the openGL
 * Widget.**/
struct Camera : QOpenGLFunctions_3_3_Core {
private:
  Camera() { initializeOpenGLFunctions(); };

public:
  std::vector<std::tuple<QOpenGLShaderProgram *, GLuint, GLuint>> programs;

  static Camera &instance();

  enum CameraType { lookat, firstperson };

  void updateViewMatrix();
  bool moving();
  void setPerspective(float fov, float aspect, float znear, float zfar);
  void updateAspectRatio(float aspect);

  std::pair<bool, DeviceCamera> prepareDeviceCamera();

  void setPosition(QVector3D position);
  void setRotation(QVector3D rotation);
  void rotate(QVector3D delta);
  void setTranslation(QVector3D translation);
  void translate(QVector3D delta);
  void update(float deltaTime);

  virtual void setKeyboardModifiers(QInputEvent *event);
  virtual void resizeEvent(QResizeEvent *event);
  virtual void keyPressEvent(QKeyEvent *event);
  virtual void keyReleaseEvent(QKeyEvent *event);
  virtual void mousePressEvent(QMouseEvent *event);
  virtual void mouseReleaseEvent(QMouseEvent *event);
  virtual void mouseDoubleClickEvent(QMouseEvent *event);
  virtual void mouseMoveEvent(QMouseEvent *event);
  virtual void wheelEvent(QWheelEvent *event);

  float fov;
  float znear, zfar, aspect;

  int32_t width, height;

  CameraType type = CameraType::firstperson;

  QVector3D rotation = QVector3D();
  QVector3D position = QVector3D();
  QVector3D right = QVector3D();
  QVector3D up = QVector3D();

  float rotationSpeed = 0.5f;
  float movementSpeed = 1.0f;

  bool dirty = false;
  struct {
    QMatrix4x4 perspective;
    QMatrix4x4 view;
  } matrices;

  bool lbuttondown = false;
  bool rbuttondown = false;
  bool mbuttondown = false;
  QVector2D mousePos;

  struct {
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
    bool q = false;
    bool e = false;
  } keys;
};
