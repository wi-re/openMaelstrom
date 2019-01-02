#include <QApplication>
#include <QtMath>
#include <iostream>
#include <render/util/camera.h>
#include <utility/helpers/log.h>
#include <utility/identifier/uniform.h>

Camera &Camera::instance() {
  static Camera cam;
  return cam;
}

std::pair<bool, DeviceCamera> Camera::prepareDeviceCamera() {
    // QVector3D forward(matrices.view(2, 0), matrices.view(2, 1), matrices.view(2, 2));
    // QVector3D strafe(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
    // QVector3D up(matrices.view(1, 0), matrices.view(1, 1), matrices.view(1, 2));

  DeviceCamera cam;
  float fovx = fov*1.f;
  fovx = 96.f;
  float2 resolution{(float)width, (float)height};
float2 fov2;
  fov2.x = fovx;
  fov2.y = atan(tan(fovx / 180.f * CUDART_PI_F * 0.5) *
               1.f / aspect) *
          2.0 * 180.f / CUDART_PI_F;
//std::cout << fov2.x << " : " << fov2.y << std::endl;
  cam.resolution = float2{(float)width, (float)height};
  cam.position = float3{position.x(), position.y(), position.z()};
  cam.view = float3{forward.x(), forward.y(), forward.z()};
  cam.up = float3{up.x(), up.y(), up.z()};
  cam.fov = float2{fov2.x, fov2.y};
  cam.apertureRadius = 0.04f;
  cam.focalDistance = 4.f;

  auto MVP = matrices.perspective * matrices.view;

  auto mat_conv = [](auto v_i) {
    Matrix4x4 mat;
    mat(0, 0) = v_i(0, 0);
    mat(1, 0) = v_i(1, 0);
    mat(2, 0) = v_i(2, 0);
    mat(3, 0) = v_i(3, 0);

    mat(0, 1) = v_i(0, 1);
    mat(1, 1) = v_i(1, 1);
    mat(2, 1) = v_i(2, 1);
    mat(3, 1) = v_i(3, 1);

    mat(0, 2) = v_i(0, 2);
    mat(1, 2) = v_i(1, 2);
    mat(2, 2) = v_i(2, 2);
    mat(3, 2) = v_i(3, 2);

    mat(0, 3) = v_i(0, 3);
    mat(1, 3) = v_i(1, 3);
    mat(2, 3) = v_i(2, 3);
    mat(3, 3) = v_i(3, 3);
    return mat;
  };

  cam.ViewInverse = mat_conv(matrices.view.inverted().transposed());
  cam.PerspInverse = mat_conv(matrices.perspective.inverted().transposed());
  cam.MVP = mat_conv(MVP.inverted().transposed());
  return std::make_pair(dirty, cam);
}

void Camera::setKeyboardModifiers([[maybe_unused]] QInputEvent *event) {}
void Camera::resizeEvent([[maybe_unused]] QResizeEvent *event) {}
void Camera::keyPressEvent(QKeyEvent *event) {
  setKeyboardModifiers(event);
  switch (event->key()) {
  case Qt::Key_W:
    keys.up = true;
    break;
  case Qt::Key_S:
    keys.down = true;
    break;
  case Qt::Key_A:
    keys.left = true;
    break;
  case Qt::Key_D:
    keys.right = true;
    break;
  case Qt::Key_E:
    keys.e = true;
    break;
  case Qt::Key_Q:
    keys.q = true;
    break;
  case Qt::Key_F1:
    setPosition(QVector3D{150, 0, 50});
    setRotation(QVector3D{180, 0, -90});
    break;
  case Qt::Key_F2:
    setPosition(QVector3D{0, 150, 50});
    setRotation(QVector3D{180, 0, -180});
    break;
  case Qt::Key_F3:
    setPosition(QVector3D{-150, 0, 50});
    setRotation(QVector3D{180, 0, 90});
    break;
  case Qt::Key_F4:
    setPosition(QVector3D{0, -150, 50});
    setRotation(QVector3D{180, 0, 0});
    break;
  case Qt::Key_F5:
    setPosition(QVector3D{0, 0, -150});
    setRotation(QVector3D{-90, 0, 90});
    break;
  case Qt::Key_F6:
    setPosition(QVector3D{0, 0, 150});
    setRotation(QVector3D{90, 0, 90});
    break;
  }
}
void Camera::keyReleaseEvent(QKeyEvent *event) {
  setKeyboardModifiers(event);
  setKeyboardModifiers(event);
  switch (event->key()) {
  case Qt::Key_W:
    keys.up = false;
    break;
  case Qt::Key_S:
    keys.down = false;
    break;
  case Qt::Key_A:
    keys.left = false;
    break;
  case Qt::Key_D:
    keys.right = false;
    break;
  case Qt::Key_E:
    keys.e = false;
    break;
  case Qt::Key_Q:
    keys.q = false;
    break;
  case Qt::Key_L:
    LOG_DEBUG << "        \"camera_position\": \"" << position.x() << " " << position.y() << " " << position.z()
              << "\"," << std::endl;
    LOG_DEBUG << "        \"camera_angle\": \"" << rotation.x() << " " << rotation.y() << " " << rotation.z() << "\","
              << std::endl;
    break;
  case Qt::Key_R: {
    auto p = get<parameters::camera_position>();
    auto r = get<parameters::camera_angle>();
    QVector3D pos{p.x, p.y, p.z};
    QVector3D rot{r.x, r.y, r.z};
    setPosition(pos);
    setRotation(rot);
  }

  break;
  }
}
void Camera::mousePressEvent(QMouseEvent *event) {
  setKeyboardModifiers(event);
  auto p = event->screenPos();

  double xpos, ypos;
  xpos = p.x();
  ypos = p.y();
  mousePos.setX(xpos);
  mousePos.setY(ypos);
  switch (event->button()) {
  case Qt::LeftButton:
    lbuttondown = true;
    break;
  case Qt::MidButton:
    mbuttondown = true;
    break;
  case Qt::RightButton:
    rbuttondown = true;
    break;
  default:
    break;
  }
  if (lbuttondown || rbuttondown || mbuttondown) {
    QApplication::setOverrideCursor(Qt::BlankCursor);
  } else {
    QApplication::setOverrideCursor(Qt::ArrowCursor);
  }
}
void Camera::mouseReleaseEvent(QMouseEvent *event) {
  setKeyboardModifiers(event);
  switch (event->button()) {
  case Qt::LeftButton:
    lbuttondown = false;
    break;
  case Qt::MidButton:
    mbuttondown = false;
    break;
  case Qt::RightButton:
    rbuttondown = false;
    break;
  default:
    break;
  }
  if (lbuttondown || rbuttondown || mbuttondown) {
    QApplication::setOverrideCursor(Qt::BlankCursor);
  } else {
    QApplication::setOverrideCursor(Qt::ArrowCursor);
  }
}
void Camera::mouseDoubleClickEvent(QMouseEvent *event) { setKeyboardModifiers(event); }
void Camera::mouseMoveEvent(QMouseEvent *event) {
  setKeyboardModifiers(event);
  auto p = event->screenPos();

  if (QPointF(mousePos.x(), mousePos.y()) == p)
    return;

  double xpos, ypos;
  xpos = p.x();
  ypos = p.y();

  if (lbuttondown) {

    QVector3D diff;
    diff.setZ(-rotationSpeed * (mousePos.x() - (float)xpos));
    diff.setX(rotationSpeed * (mousePos.y() - (float)ypos));
    rotate(diff);
    updateViewMatrix();

    QCursor::setPos(QPoint(mousePos.x(), mousePos.y()));
    event->accept();
  }
  if (rbuttondown) {
    // QVector3D forward(matrices.view(0, 2), matrices.view(1, 2), matrices.view(2, 2));
    // QVector3D strafe(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
    // QVector3D up(matrices.view(0, 1), matrices.view(1, 1), matrices.view(2, 1));

    //QVector3D camFront;
    //camFront = forward.normalized();

    position += forward * (mousePos.y() - (float)ypos) * 0.01f;
    updateViewMatrix();

    QCursor::setPos(QPoint(mousePos.x(), mousePos.y()));
    event->accept();
  }
  if (mbuttondown) {
    //QVector3D strafe(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
    //QVector3D up(matrices.view(0, 1), matrices.view(1, 1), matrices.view(2, 1));

    position -= (strafe.normalized()) * (mousePos.x() - (float)xpos) * 0.01f;
    position -= (up.normalized()) * (mousePos.y() - (float)ypos) * 0.01f;
    updateViewMatrix();

    QCursor::setPos(QPoint(mousePos.x(), mousePos.y()));
    event->accept();
  }
}
void Camera::wheelEvent(QWheelEvent *event) { setKeyboardModifiers(event); }
void Camera::updateViewMatrix() {
  float pitch = rotation.x() * M_PI / 180.f;
  float yaw = rotation.z() * M_PI / 180.f;
  float tilt = rotation.z() * M_PI / 180.f;
  //std::cout << yaw << " : " << pitch << std::endl;

  float xDirection = sin(yaw) * cos(pitch);
  float zDirection = sin(pitch);
  float yDirection = cos(yaw) * cos(pitch);
  //std::cout << xDirection << " " << yDirection << " " << zDirection << std::endl;
  float3 centerPosition{position.x(), position.y(), position.z()};
  float3 directionToCamera = float3{xDirection, yDirection, zDirection};
  float3 viewDirection = directionToCamera * (-1.0f);
  float3 eyePosition = centerPosition;

  auto to_vec3 = [](auto v) { return QVector3D(v.x, v.y, v.z); };
  auto p = to_vec3(eyePosition);
  auto v = to_vec3(viewDirection);
  auto horz = QVector3D::crossProduct(v,QVector3D(0,0,1));
  auto vert = QVector3D::crossProduct(p, v);
  //auto u = vert;
  auto u = to_vec3(float3{0, 0, 1});
  if((v + u).length() < 0.1)
    u = to_vec3(float3{1,0,0});
  forward = v;
  forward.normalize();
  up = u;
  up.normalize();
  strafe = QVector3D::crossProduct(forward,up);
  strafe.normalize();


  matrices.view = QMatrix4x4();
  matrices.view.lookAt(p,p + v,u);

  QMatrix4x4 rotM = QMatrix4x4();
  QMatrix4x4 transM;
  rotM.rotate(rotation.x(), 1.f, 0.f, 0.f);
  rotM.rotate(rotation.y(), 0.f, 1.f, 0.f);
  rotM.rotate(rotation.z(), 0.f, 0.f, 1.f);

  transM.translate(position);

  if (type == CameraType::firstperson) {
    //matrices.view = rotM * transM;
  } else {
    //matrices.view = transM * rotM;
  }

  // forward = QVector3D(matrices.view(2, 0), matrices.view(2, 1), matrices.view(2, 2));
  // right = QVector3D(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
  // up = QVector3D(matrices.view(0, 1), matrices.view(1, 1), matrices.view(2, 1));

  QMatrix4x4 matrix;
  matrix.perspective((fov), aspect, znear, zfar);
  //std::cout << fov << std::endl;
  matrices.perspective = matrix;

  for (auto p : programs) {
    auto [program, view, perspective] = p;

    program->bind();
    program->setUniformValue(perspective, matrix);
    program->setUniformValue(view, *((QMatrix4x4 *)&matrices.view));

    program->release();
  }
  dirty = true;
}
bool Camera::moving() { return keys.left || keys.right || keys.up || keys.down || keys.e || keys.q; }
void Camera::setPerspective(float fov, float aspect, float znear, float zfar) {
  this->fov = fov;
  this->znear = znear;
  this->zfar = zfar;
  this->aspect = aspect;
  glViewport(0, 0, width, height);

  QMatrix4x4 matrix;
  matrix.perspective((fov), aspect, znear, zfar);
  matrices.perspective = matrix;

  for (auto p : programs) {
    auto [program, view, perspective] = p;
    program->bind();
    program->setUniformValue(perspective, matrix);
    program->setUniformValue(view, *((QMatrix4x4 *)&matrices.view));

    program->release();
  }
  dirty = true;
}
void Camera::updateAspectRatio(float aspect) { setPerspective(fov, aspect, znear, zfar); }
void Camera::setPosition(QVector3D position) {
  this->position = position;
  updateViewMatrix();
}
void Camera::setRotation(QVector3D rotation) {
  this->rotation = rotation;
  updateViewMatrix();
}
void Camera::rotate(QVector3D delta) {
  this->rotation += delta;
  updateViewMatrix();
}
void Camera::setTranslation(QVector3D translation) {
  this->position = translation;
  updateViewMatrix();
}
void Camera::translate(QVector3D delta) {
  this->position += delta;
  updateViewMatrix();
}
void Camera::update(float deltaTime) {
  if (type == CameraType::firstperson) {
    if (moving()) {
      QVector3D camFront;
      camFront.setX(-cos(qDegreesToRadians(rotation.x())) * sin(qDegreesToRadians(rotation.y())));
      camFront.setY(sin(qDegreesToRadians(rotation.x())));
      camFront.setZ(cos(qDegreesToRadians(rotation.x())) * cos(qDegreesToRadians(rotation.y())));

      camFront.normalize();
    //   QVector3D forward(matrices.view(2, 0), matrices.view(2, 1), matrices.view(2, 2));
    //   QVector3D strafe(matrices.view(0, 0), matrices.view(0, 1), matrices.view(0, 2));
    //   QVector3D up(matrices.view(1, 0), matrices.view(1, 1), matrices.view(1, 2));

    // QVector3D forward(matrices.view(2, 0), matrices.view(2, 1), matrices.view(2, 2));
    // QVector3D strafe(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
    // QVector3D up(matrices.view(1, 0), matrices.view(1, 1), matrices.view(1, 2));

      forward.normalize();
      strafe.normalize();
      QVector3D upl = QVector3D::crossProduct(forward,strafe);
      up.normalize();

      if (deltaTime < 0.02f)
        deltaTime = 0.02f;
      float moveSpeed = deltaTime * movementSpeed;

      if (keys.up)
        position += (forward)*moveSpeed;
      if (keys.down)
        position -= (forward)*moveSpeed;
      if (keys.left)
        position -= (strafe)*moveSpeed;
      if (keys.right)
        position += (strafe)*moveSpeed;
      if (keys.q)
        position += (upl)*moveSpeed;
      if (keys.e)
        position -= (upl)*moveSpeed;

      updateViewMatrix();
    }
  }
}
