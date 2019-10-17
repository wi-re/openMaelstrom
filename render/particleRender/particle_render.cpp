#include <render/particleRender/particle_render.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450 
in vec4 posAttr;
in vec2 uvAttr;
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

uniform int map_flipped;

uniform sampler1D colorRamp;

out vec2 uv;
out vec4 color;
out vec4 eyeSpacePos;
flat out int invalid;

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
	uv = uvAttr;
	float intensity = 0.f;
	float renderIntensity = renderArray.w;
	if(color_map_min < color_map_max)
		intensity = mapValue(renderIntensity, color_map_min, color_map_max, transfer_fn);
	else
		intensity = mapValue(renderIntensity, color_map_max, color_map_min, transfer_fn);
	//intensity = (float(transfer_fn)) / 5.f;
	intensity = clamp(intensity,0.f,1.f);

	if(map_flipped != 0)
		intensity = 1.f - intensity;

	intensity = mapValue(intensity, 0.f, 1.f, mapping_fn);
	color = texture(colorRamp,intensity);

	float V = pow(volume / (4.f / 3.f * 3.14159265359f),1.f/3.f);
	eyeSpacePos = view_matrix * vec4(position.xyz,1.f);
	eyeSpacePos += vec4(posAttr.xyz * V * 2.f,0.f);
	gl_Position = perspective_matrix * eyeSpacePos;

 	vec3 flowDist = maxCoord-minCoord;
 	vec3 clipVecMin = minCoord + abs (render_clamp) * flowDist;
 	vec3 clipVecMax = maxCoord - abs (render_clamp) * flowDist;
 	if(
    ((render_clamp.x < 0.0f) ? position.x < clipVecMin.x : position.x > clipVecMax.x) ||
    ((render_clamp.y < 0.0f) ? position.y < clipVecMin.y : position.y > clipVecMax.y) ||
    ((render_clamp.z < 0.0f) ? position.z < clipVecMin.z : position.z > clipVecMax.z) )
			invalid = 1;
	else
			invalid = 0;
})";

static const char *fragmentShaderSource = R"(#version 450 
in vec2 uv;
in vec4 color;
in vec4 eyeSpacePos;
flat in int invalid;


uniform sampler1D           colorRamp;
uniform mat4 perspective_matrix;

void main() {
	if(invalid == 1)
		discard;
	vec3 N;
	N.xy = uv.xy * vec2(2.f, 2.f) - vec2(1.f, 1.f);
	float r2 = dot(N.xy,N.xy);
	if( r2 > 1.f) discard;
	N.z = sqrt(1.f - r2);

	vec3 lightDir = vec3(0, 0, 1);
	float diffuse = abs(dot(N, lightDir));

	gl_FragColor = vec4(diffuse,diffuse,diffuse,1.f) * color;
	//gl_FragColor = color;
	//gl_FragColor = texture(colorRamp,uv.x);
	//gl_FragColor = vec4(N,1.f) ;

vec4 sphereEyeSpacePos;
sphereEyeSpacePos.xyz = eyeSpacePos.xyz + N * eyeSpacePos.w;
sphereEyeSpacePos.w = 1.0;
vec4 projPos = (perspective_matrix * sphereEyeSpacePos);
gl_FragDepth = (projPos.z / projPos.w)* 0.5 + 0.5;
})";

bool ParticleRenderer::valid() { return true; }

#include <experimental/filesystem>
#include <utility/helpers/pathfinder.h>
void ParticleRenderer::update() {
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

ParticleRenderer::ParticleRenderer(OGLWidget *parent) {
  initializeOpenGLFunctions();
  m_program = new QOpenGLShaderProgram(parent);
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  LOG_INFO << "Compiling vertex shader for " << "ParticleRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  LOG_INFO << "Compiling fragment shader for " << "ParticleRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  m_program->link();
  LOG_INFO << "Linking " << "ParticleRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  m_posAttr = m_program->attributeLocation("posAttr");
  m_colAttr = m_program->attributeLocation("uvAttr");

  std::vector<uint32_t> idx = {0, 1, 2, 2, 3, 0};
  glGenBuffers(1, &IBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint32_t), idx.data(), GL_STATIC_DRAW);

  std::vector<QVector4D> instance_pos = {{0.f, 1.f, 0.f, 1.f},
                                         {1.f, 1.f, 0.f, 1.f},
                                         {0.f, -1.f, 0.f, 1.f},
                                         {-1.f, -1.f, 0.f, 1.f},
                                         {0.f, 0.f, 0.f, 1.f}};

  parent->bind(m_program);

  GLfloat vertices[] = {-0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.5f, 0.5f, 0.0f, -0.5f, 0.5f, 0.0f};

  glGenBuffers(1, &VXO);
  glBindBuffer(GL_ARRAY_BUFFER, VXO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  GLfloat uvs[] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

  glGenBuffers(1, &VUV);
  glBindBuffer(GL_ARRAY_BUFFER, VUV);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_colAttr);
  glVertexAttribPointer(m_colAttr, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  colorMap::instance().bind(m_program, 0, "colorRamp");
  update();
}

void ParticleRenderer::render(bool pretty) {
	//return;
  glBindVertexArray(vao);

  m_program->bind();

  glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0,
                          get<parameters::num_ptcls>());

  m_program->release();
  glBindVertexArray(0);
}
