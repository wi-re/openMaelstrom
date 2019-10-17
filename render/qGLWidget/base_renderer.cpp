#define BOOST_USE_WINDOWS_H
#include <QApplication>
#include <QDebug>
#include <QOpenGLTimerQuery>
#include <iostream>
#include <render/qGLWidget/oglwidget.h>
#include <utility/helpers/arguments.h>
#include <utility/identifier/uniform.h>
#include <render/qGLWidget/base_renderer.h>
#include <render/boundaryRender/bounds.h>
#include <render/particleRender/particle_render.h>
#include <render/volumeRender/volume_render.h>
#include <render/vdbPolyRender/vdb_poly_render.h>
#include <render/quadRender/quadRender.h>
#include <render/mlmRender/mlmRender.h>
#include <utility/template/tuple_for_each.h>
#include <boost/format.hpp>
#include <boost/type_traits/is_assignable.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <chrono>
#include <mutex>
#include <simulation/particleSystem.h>
#include <sstream>
#include <utility/helpers/arguments.h>
#include <utility/helpers/timer.h>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <utility/math.h>
#include <render/qGLWidget/base_renderer.h>

RTXRender::RTXRender(QGLWidget* p) {
	if (get<parameters::rayTracing>() == false) return;
	auto parent = dynamic_cast<OGLWidget*>(p);
	static const char *vertexShaderSource = R"(#version 450
in vec3 vertexPosition_modelspace;
out vec2 UV;

void main(){
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vertexPosition_modelspace.xy+vec2(1,1))/2.0;
	UV.y = 1.f - UV.y;
}
)";

	static const char *fragmentShaderSource = R"(#version 450 
uniform sampler2D renderedTexture;

in vec2 UV;
out vec3 color;

void main(){
	vec4 col = texture( renderedTexture, UV);
	color = vec3(col.xyz) ;
	//gl_FragDepth = col.w;
}
)";

	initialFrame = get<parameters::frame>();
	auto h_scene = hostScene();
	cudaMalloc(&accumulatebuffer, h_scene.width * h_scene.height * sizeof(float3));
	initializeOpenGLFunctions();
	quad_programID = new QOpenGLShaderProgram(parent);
	quad_programID->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
	quad_programID->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);

	glGenVertexArrays(1, &defer_VAO);
	glBindVertexArray(defer_VAO);
	quad_programID->link();
	parent->bind(quad_programID);
	quad_programID->bind();

	auto m_posAttr = quad_programID->attributeLocation("vertexPosition_modelspace");
	glGenTextures(1, &renderedTextureOut);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, renderedTextureOut);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, h_scene.width, h_scene.height, 0, GL_RGBA, GL_FLOAT, 0);
	cudaGraphicsGLRegisterImage(&renderedResourceOut, renderedTextureOut, GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsSurfaceLoadStore);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glActiveTexture(GL_TEXTURE0);
	static const GLfloat g_quad_vertex_bufferdata[] = {
		-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
	};

	GLuint quad_vertexbuffer;
	glGenBuffers(1, &quad_vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_bufferdata), g_quad_vertex_bufferdata, GL_STATIC_DRAW);

	glEnableVertexAttribArray(m_posAttr);
	glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
	quad_programID->setUniformValue("renderedTexture", 0);

	quad_programID->release();
	//prepCUDAscene();
	//update();

}

void RTXRender::update() {
	updateRTX();
}
bool RTXRender::valid() { 
	return bValid; 
}
void RTXRender::render(bool pretty) {
	static std::random_device r;
	static std::default_random_engine e1(r());
	static std::uniform_int_distribution<int32_t> uniform_dist(INT_MIN, INT_MAX);

	if (get<parameters::frame>() == initialFrame) {
		return;
	}
	if (get<parameters::rayTracing>() == false) return;
	auto h_scene = hostScene();
	if (h_scene.dirty || frame != get<parameters::frame>() || dirty) {
		frame = get<parameters::frame>();
		cudaMemset(accumulatebuffer, 1, h_scene.width * h_scene.height * sizeof(float3));
		framenumber = 0;
		dirty = false;
	}
	if (!bValid) return;
	int32_t iterations = 1;
	if (pretty)
		iterations = 50;

	for (int32_t i = 0; i < iterations; ++i) {
		framenumber++;
		renderRTX(pretty, framenumber, uniform_dist(e1));
	}
		glBindVertexArray(defer_VAO);
		quad_programID->bind();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, renderedTextureOut);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
		quad_programID->release();
		glBindVertexArray(0);
}
void RTXRender::prepCUDAscene() {

}