#include <render/particleDirectionRender/particleDirectionRender.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450 
in vec4 posAttr;
in vec4 position;
in vec4 renderArray;
in float volume;

uniform vec4 camera_right;
uniform vec4 camera_up;
uniform mat4 perspective_matrix;
uniform mat4 view_matrix;

uniform vec3 minCoord;
uniform vec3 maxCoord;
uniform vec3 render_clamp;

uniform int transfer_fn;
uniform int mapping_fn;
uniform float color_map_min;
uniform float color_map_max;

uniform float vectorScale;
uniform int vectorScaling;

uniform int map_flipped;

uniform sampler1D colorRamp;

out vec4 direction;
out vec4 colorA;
out int invalidA;

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

void main() {
	float intensity = 0.f;
	float renderIntensity = renderArray.w;
	direction = renderArray;
	if(color_map_min < color_map_max)
		intensity = mapValue(renderIntensity, color_map_min, color_map_max, transfer_fn);
	else
		intensity = mapValue(renderIntensity, color_map_max, color_map_min, transfer_fn);
	//intensity = (float(transfer_fn)) / 5.f;
	intensity = clamp(intensity,0.f,1.f);

	vec3 dir = renderArray.xyz;
	float len = length(dir);
	dir = normalize(dir);
	if(vectorScaling == 0)
		direction = vec4(dir, len * vectorScale);
	else
		direction = vec4(dir, intensity * vectorScale);

	if(map_flipped != 0)
		intensity = 1.f - intensity;
	
		

	intensity = mapValue(intensity, 0.f, 1.f, mapping_fn);
	colorA = texture(colorRamp,intensity);

    gl_Position = vec4(position.xyz, 1.0); 

 	vec3 flowDist = maxCoord-minCoord;
 	vec3 clipVecMin = minCoord + abs (render_clamp) * flowDist;
 	vec3 clipVecMax = maxCoord - abs (render_clamp) * flowDist;
 	if(
    ((render_clamp.x < 0.0f) ? position.x < clipVecMin.x : position.x > clipVecMax.x) ||
    ((render_clamp.y < 0.0f) ? position.y < clipVecMin.y : position.y > clipVecMax.y) ||
    ((render_clamp.z < 0.0f) ? position.z < clipVecMin.z : position.z > clipVecMax.z) )
			invalidA = 1;
	else
			invalidA = 0;
})";

static const char *geometryShaderSource = R"(#version 450 
layout (points) in;
layout (line_strip, max_vertices = 2) out;
in vec4 direction[];
in vec4 colorA[];
flat in int invalidA[];

uniform mat4 perspective_matrix;
uniform mat4 view_matrix;

out vec4 eyeSpacePos;
out vec4 color;
out int invalid;

void main() {
	vec4 worldPos;
    worldPos = gl_in[0].gl_Position + vec4(-0.0, 0.0, 0.0, 0.0); 
	eyeSpacePos = view_matrix * vec4(worldPos.xyz,1.f);
	gl_Position = perspective_matrix * eyeSpacePos;
	color = colorA[0];
	invalid = invalidA[0];
    EmitVertex();

    worldPos = gl_in[0].gl_Position + vec4(direction[0].xyz, 0.0) * direction[0].w; 
	eyeSpacePos = view_matrix * vec4(worldPos.xyz,1.f);
	gl_Position = perspective_matrix * eyeSpacePos;
	color = colorA[0];
	invalid = invalidA[0];
    EmitVertex();

    EndPrimitive();
})";

static const char *fragmentShaderSource = R"(#version 450 
in vec4 color;
in vec4 eyeSpacePos;
flat in int invalid;

out vec4 out_color;

uniform mat4 perspective_matrix;
uniform mat4 view_matrix;

void main() {
	if(invalid == 1) discard;
	out_color = color;
	//out_color = vec4(1,0,0,1);
	vec4 projPos = (perspective_matrix * eyeSpacePos);
	gl_FragDepth = (projPos.z / projPos.w)* 0.5 + 0.5;
})";

bool ParticleDirectionRenderer::valid() { return true; }

#include <experimental/filesystem>
#include <utility/helpers/pathfinder.h>
void ParticleDirectionRenderer::update() {
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

ParticleDirectionRenderer::ParticleDirectionRenderer(OGLWidget *parent) {
  initializeOpenGLFunctions();
  m_program = new QOpenGLShaderProgram(parent);
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  LOG_INFO << "Compiling vertex shader for " << "ParticleDirectionRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  m_program->addShaderFromSourceCode(QOpenGLShader::Geometry, geometryShaderSource);
  LOG_INFO << "Compiling geometry shader for " << "ParticleDirectionRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  LOG_INFO << "Compiling fragment shader for " << "ParticleDirectionRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  m_program->link();
  LOG_INFO << "Linking " << "ParticleDirectionRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;

  m_posAttr = m_program->attributeLocation("posAttr");

  parent->bind(m_program);

  GLfloat vertices[] = {0.f,0.f,0.f};
  glGenBuffers(1, &VXO);
  glBindBuffer(GL_ARRAY_BUFFER, VXO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  colorMap::instance().bind(m_program, 0, "colorRamp");
  update();
}

void ParticleDirectionRenderer::render(bool pretty) {
	if (get<parameters::visualizeDirection>() != 1) return;
  glBindVertexArray(vao);

  m_program->bind();
 // glDisable(GL_LIGHTING);
  glDrawArraysInstanced(GL_POINTS, 0, 1, get<parameters::num_ptcls>());

  m_program->release();
  glBindVertexArray(0);
}
