#pragma once
#include <QOpenGLFunctions_4_5_Compatibility>
#include <QOpenGLShaderProgram>

/** This class is used to load color maps from files. The color maps are loaded
 * from pngs relative to the resource path. **/
class colorMap : QOpenGLFunctions_4_5_Compatibility {
  GLuint create1DTexture(QVector4D *colors, int32_t elements);
  QVector4D *color_map = nullptr;
  int32_t color_map_elements = 0;
  GLint texunit = -1;

  colorMap() {
    initializeOpenGLFunctions();
    update();
  }

  std::vector<std::tuple<QOpenGLShaderProgram *, GLint, std::string>> bindings;

public:
  static colorMap &instance();

  void update();
  void bind(QOpenGLShaderProgram *prog, GLint texture_id, std::string id);
};
