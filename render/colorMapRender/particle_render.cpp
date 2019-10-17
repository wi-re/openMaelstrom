#include <render/ColorMapRender/particle_render.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450
uniform vec4 camera_right;
uniform vec4 camera_up;
uniform mat4 perspective_matrix;
uniform mat4 view_matrix;

in vec3 vertexPosition_modelspace;
out vec2 UV;

void main(){
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vertexPosition_modelspace.xy+vec2(1,1))/2.0;
	UV.y = 1.f - UV.y;
}
)";

static const char *fragmentShaderSource = R"(#version 450 
uniform sampler1D colorRamp;

uniform int transfer_fn;
uniform int mapping_fn;
uniform float color_map_min;
uniform float color_map_max;

uniform int map_flipped;

in vec2 UV;
out vec3 color;

float mapValue(float value, float min, float max, int fn){
	if(fn == 0)
		return (value - min) / (max - min);
	if(fn == 1)
		return (sqrt(value) - sqrt(min)) / (sqrt(max) - sqrt(min));
	if(fn == 2)
		return (value * value - min * min) / (max * max - min * min);
	if(fn == 3)
		return (pow(value,1.f/3.f) - pow(min,1.f/3.f)) / (pow(max,1.f/3.f) - pow(min,1.f/3.f));
	if(fn == 4)
		return (value * value * value - min * min * min) / (max * max * max - min * min * min);
	if(fn == 5)
		return (log(value) - log(min)) / (log(max) - log(min));
	return (value - min) / (max - min);
}

void main(){
	if(UV.x < 0.98f) discard;
	float intensity = 1.f - UV.y;
	if(map_flipped != 0)
		intensity = 1.f - intensity;
	intensity = mapValue(intensity, 0.f, 1.f, mapping_fn);
	vec4 col = texture(colorRamp,intensity);

	color = vec3(col.xyz);
}
)";

bool ColorMapRenderer::valid() { return m_valid; }

#include <experimental/filesystem>
#include <utility/helpers/pathfinder.h>
void ColorMapRenderer::update() {
	auto maps = []() {std::vector <std::string> colorMaps;auto f = resolveFile("cfg/style.css");auto p = f.parent_path().string();if (*(p.end() - 1) == '/' || *(p.end() - 1) == '\\')p = p.substr(0, p.length() - 1);std::replace(p.begin(), p.end(), '\\', '/');for (auto& p : std::experimental::filesystem::directory_iterator(p))if (p.path().extension().string().find(".png") != std::string::npos)colorMaps.push_back(p.path().filename().replace_extension("").string());return colorMaps; }();

	//auto list = []() {std::vector<std::string> names;iterateArraysList([&](auto x) {using Ty = std::decay_t<decltype(x)>;if (Ty::valid() && Ty::kind == memory_kind::particleData)names.push_back(Ty::qualifiedName);});return names;}();
	//for (auto l : list)
	//	std::cout << l << ", ";
	//std::cout << std::endl;
	//std::cout << get<parameters::mapping_mode>() << " -> " << get<parameters::transfer_fn>() << std::endl;
	//std::cout << get<parameters::render_min>() << " - " << get<parameters::render_max>() << std::endl;
	static std::string tMode = "";
	static std::string mMode = "";
	if (tMode != get<parameters::transfer_mode>()) {
		if (get<parameters::transfer_mode>().find("linear") != std::string::npos)
			get<parameters::transfer_fn>() = 0;
		if (get<parameters::transfer_mode>().find("cubic") != std::string::npos)
			get<parameters::transfer_fn>() = 4;
		if (get<parameters::transfer_mode>().find("cubicRoot") != std::string::npos)
			get<parameters::transfer_fn>() = 3;
		if (get<parameters::transfer_mode>().find("square") != std::string::npos)
			get<parameters::transfer_fn>() = 2;
		if (get<parameters::transfer_mode>().find("squareRoot") != std::string::npos)
			get<parameters::transfer_fn>() = 1;
		if (get<parameters::transfer_mode>().find("log") != std::string::npos)
			get<parameters::transfer_fn>() = 5;
		tMode = get<parameters::transfer_mode>();
	}
	if (mMode != get<parameters::mapping_mode>()) {
		if (get<parameters::mapping_mode>().find("linear") != std::string::npos)
			get<parameters::mapping_fn>() = 0;
		if (get<parameters::mapping_mode>().find("cubic") != std::string::npos)
			get<parameters::mapping_fn>() = 4;
		if (get<parameters::mapping_mode>().find("cubicRoot") != std::string::npos)
			get<parameters::mapping_fn>() = 3;
		if (get<parameters::mapping_mode>().find("square") != std::string::npos)
			get<parameters::mapping_fn>() = 2;
		if (get<parameters::mapping_mode>().find("squareRoot") != std::string::npos)
			get<parameters::mapping_fn>() = 1;
		if (get<parameters::mapping_mode>().find("log") != std::string::npos)
			get<parameters::mapping_fn>() = 5;
		mMode = get<parameters::mapping_mode>();
	}
	
	
	colorMap::instance().update(); }

ColorMapRenderer::ColorMapRenderer(OGLWidget *parent) {
  if (get<parameters::rayTracing>() == false)
    return;
  auto h_scene = hostScene();

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

  colorMap::instance().bind(quad_programID, 0, "colorRamp");
  update();
}

void ColorMapRenderer::render(bool pretty) {
  if (!valid())
    return;
  glBindVertexArray(defer_VAO);
  quad_programID->bind();
  glDisable(GL_DEPTH_TEST);
  glActiveTexture(GL_TEXTURE0);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glEnable(GL_DEPTH_TEST);
  quad_programID->release();
  glBindVertexArray(0);
}
