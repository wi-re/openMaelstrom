#pragma once
#include <QOpenGLFunctions_3_3_Core>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <utility/cuda.h>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <iostream>

template <typename T> auto convertToQt(T) { return 0.f; };
template <> inline auto convertToQt<float>(float val) { return val; }
template <> inline auto convertToQt<int>(int val) { return val; }
template <> inline auto convertToQt<float3>(float3 val) {
  return QVector3D(val.x, val.y, val.z);
}
template <> inline auto convertToQt<QMatrix4x4>(QMatrix4x4 val) { return val; }
struct gl_uniform_base {
  std::vector<std::pair<QOpenGLShaderProgram *, GLuint>> programs;

  virtual void update() = 0;
  virtual void add_uniform(QOpenGLShaderProgram *program) = 0;
  virtual void add_uniform(QOpenGLShaderProgram *program, GLuint location) = 0;
};
template <typename T> struct gl_uniform_custom : public gl_uniform_base {
  T *ptr;
  std::function<bool()> update_fn;
  std::string variableName;

  gl_uniform_custom(T *p, std::string identifier)
      : ptr(p), variableName(identifier) {
    update_fn = [this]() {
      static T old = *ptr;
      static auto old_size = programs.size();

      if (old != *ptr || old_size != programs.size()) {
        old = *ptr;
        old_size = programs.size();
        return true;
      }
      return false;
    };
    update_fn();
  }
  virtual void update() {
    for (auto p : programs) {
      auto [program, uniform] = p;
      program->bind();
      std::decay_t<decltype(*ptr)> val = *ptr;
      program->setUniformValue(uniform, convertToQt(val));
      program->release();
    }
  }
  virtual void add_uniform(QOpenGLShaderProgram *prog, GLuint location) {
    programs.emplace_back(prog, location);
  }
  virtual void add_uniform(QOpenGLShaderProgram *prog) {
    auto identifier = prog->uniformLocation(variableName.c_str());
    if (identifier == -1)
      return;
    programs.emplace_back(prog, identifier);
  }
};
template <typename Ty> struct gl_uniform : public gl_uniform_base {
  using T = typename Ty::type;

  std::function<bool()> update_fn;

  gl_uniform() {
    update_fn = [this]() {
      static T old = *Ty::ptr;
      static auto old_size = programs.size();
      if (old != *Ty::ptr || old_size != programs.size()) {
        old = *Ty::ptr;
        old_size = programs.size();
        return true;
      }
      return false;
    };
    update_fn();
  }
  virtual void update() {
    for (auto p : programs) {
      auto [program, uniform] = p;
      program->bind();
      std::decay_t<decltype(*Ty::ptr)> val = *Ty::ptr;
      program->setUniformValue(uniform, convertToQt(val));
      program->release();
    }
  }
  virtual void add_uniform(QOpenGLShaderProgram *prog, GLuint location) {
    programs.emplace_back(prog, location);
  }
  virtual void add_uniform(QOpenGLShaderProgram *prog) {
    auto identifier = prog->uniformLocation(Ty::variableName);
    if (identifier == -1)
      return;
    programs.emplace_back(prog, identifier);
  }
};
struct cuda_buffer_base : protected QOpenGLFunctions_3_3_Core {
  cudaGraphicsResource *resource = nullptr;
  GLuint VBO = UINT_MAX;

  virtual ~cuda_buffer_base(){
    if(resource != nullptr){
      cudaGraphicsUnregisterResource(resource);
      resource = nullptr;
    }
    if(VBO != UINT_MAX){
      glDeleteBuffers(1,&VBO);
      VBO = UINT_MAX;
    }
  }
  virtual void update() = 0;
  virtual GLint bind(GLuint attribute) = 0;
  virtual GLint bind(QOpenGLShaderProgram *prog) = 0;
};
template <typename info>
struct cuda_buffer : public cuda_buffer_base{
  using T = typename info::type;
  bool initialized = false;
  T *dptr;
  size_t num_bytes;

  std::function<bool()> update_fn;

  void initMemory() {
    //initialized = false;
    //return;
    uint32_t max_numptcls = get<parameters::max_numptcls>();
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, max_numptcls * sizeof(T), 0, GL_STREAM_DRAW);
    cudaGraphicsGLRegisterBuffer(&resource, VBO,
                                 cudaGraphicsMapFlagsWriteDiscard);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    initialized = true;
  }
  virtual ~cuda_buffer(){
    if(resource != nullptr){
      cudaGraphicsUnregisterResource(resource);
      resource = nullptr;
    }
    if(VBO != UINT_MAX){
      glDeleteBuffers(1,&VBO);
      VBO = UINT_MAX;
    }
  }
  cuda_buffer(std::function<bool()> valid =[]() {
                    static auto old_frame = INT_MAX;
                    if (old_frame != get<parameters::frame>()) {
                      old_frame = get<parameters::frame>();
                      return true;
                    }
                    return false;
                  }): update_fn(valid) {
    initializeOpenGLFunctions();
  }
  virtual void update() {
    if (!update_fn())
      return;
    if (!initialized)
      return;
    cudaGraphicsMapResources(1, &resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, resource);
    if (*parameters::target::ptr != launch_config::device)
      cudaMemcpy(dptr, info::ptr, info::alloc_size, cudaMemcpyHostToDevice);
    else
      cudaMemcpy(dptr, info::ptr, info::alloc_size, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &resource, 0);
  }
  virtual GLint bind(GLuint attribute) {
    if (!initialized)
      initMemory();
    //return -1;
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(attribute);
    glVertexAttribPointer(attribute, sizeof(T) / sizeof(float), GL_FLOAT,
                          GL_FALSE, 0, NULL);
    glVertexAttribDivisor(attribute, 1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return attribute;
  }
  virtual GLint bind(QOpenGLShaderProgram *prog) {
    auto attribute = prog->attributeLocation(info::variableName);
    if (attribute == -1)
      return -1;
    if (!initialized)
      initMemory();
    //return -1;
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(attribute);
    glVertexAttribPointer(attribute, sizeof(T) / sizeof(float), GL_FLOAT,
                          GL_FALSE, 0, NULL);
    glVertexAttribDivisor(attribute, 1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return attribute;
  }
};
