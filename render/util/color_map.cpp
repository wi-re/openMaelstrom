#include <QImage>
#include <iostream>
#include <render/util/color_map.h>
#include <utility/helpers/log.h>
#include <utility/identifier/uniform.h>

GLuint colorMap::create1DTexture(QVector4D *colorMap, int32_t elements) {
  GLuint textureId_;

  // generate the specified number of texture objects
  glGenTextures(1, &textureId_);
  // assert(glGetError() == GL_NO_ERROR);

  // bind texture
  glBindTexture(GL_TEXTURE_1D, textureId_);
  // assert(glGetError() == GL_NO_ERROR);

  // tells OpenGL how the data that is going to be uploaded is aligned
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  // assert(glGetError() == GL_NO_ERROR);

  glTexImage1D(
      GL_TEXTURE_1D, // Specifies the target texture. Must be GL_TEXTURE_1D or GL_PROXY_TEXTURE_1D.
      0, // Specifies the level-of-detail number. Level 0 is the base image level. Level n is the
         // nth mipmap reduction image.
      GL_RGBA32F, elements,
      0, // border: This value must be 0.
      GL_RGBA, GL_FLOAT, colorMap);
  // assert(glGetError() == GL_NO_ERROR);

  // texture sampling/filtering operation.
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  // assert(glGetError() == GL_NO_ERROR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  // assert(glGetError() == GL_NO_ERROR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // assert(glGetError() == GL_NO_ERROR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  // assert(glGetError() == GL_NO_ERROR);

  glBindTexture(GL_TEXTURE_1D, 0);
  // assert(glGetError() == GL_NO_ERROR);

  return textureId_;
}

#include <utility/helpers/pathfinder.h>

void colorMap::update() {
  static std::string old_colormap = "";
  static size_t old_size = 0;
  if (old_colormap != get<parameters::render_colormap>()) {

    old_colormap = get<parameters::render_colormap>();

	std::string file_name = resolveFile(std::string("cfg/") + get<parameters::render_colormap>() + ".png").string();
    //std::string file_name = root_folder + get<parameters::render_colormap>() + ".png";
    QImage img(1024, 1, QImage::Format_RGB32);
    //std::cout << file_name << " exists: " << fs::exists(file_name) << std::endl;
    for (int32_t it = 0; it < img.width(); ++it)
      img.setPixel(QPoint(it, 0), qRgb((float)it / (float)img.width() * 256.f,
                                       (float)it / (float)img.width() * 256.f,
                                       (float)it / (float)img.width() * 256.f));

    if (std::experimental::filesystem::exists(file_name))
      img.load(QString(file_name.c_str()));

    color_map = (QVector4D *)realloc(color_map, sizeof(QVector4D) * (img.width() + 1));
    for (int32_t it = 0; it < img.width(); ++it) {
      QRgb col = img.pixel(QPoint(it, 0));
      color_map[it] = QVector4D{(float)qRed(col) / 256.f, (float)qGreen(col) / 256.f,
                                (float)qBlue(col) / 256.f, 1.f};
		if(it == img.width()-1)
			color_map[it + 1] = QVector4D{ (float)qRed(col) / 256.f, (float)qGreen(col) / 256.f,
			(float)qBlue(col) / 256.f, 1.f };
    }
	
    color_map_elements = img.width();

    texunit = create1DTexture(color_map, color_map_elements);
    for (auto b : bindings) {
      auto [prog, texture_id, identifier] = b;

      prog->bind();
      GLuint samplerLocation = prog->uniformLocation(identifier.c_str());

      glUniform1i(samplerLocation, texture_id);
      glActiveTexture(GL_TEXTURE0 + texture_id);
      glBindTexture(GL_TEXTURE_1D, texunit);
      prog->release();
    }
    old_size = bindings.size();
  } else if (bindings.size() != old_size) {
    for (auto b : bindings) {
      auto [prog, texture_id, identifier] = b;

      prog->bind();
      GLuint samplerLocation = prog->uniformLocation(identifier.c_str());

      glUniform1i(samplerLocation, texture_id);
      glActiveTexture(GL_TEXTURE0 + texture_id);
      glBindTexture(GL_TEXTURE_1D, texunit);
      prog->release();
    }
    old_size = bindings.size();
  }
}

void colorMap::bind(QOpenGLShaderProgram *prog, GLint texture_id, std::string id) {

  bindings.emplace_back(prog, texture_id, id);
}

colorMap &colorMap::instance() {
  static colorMap inst;
  return inst;
}
