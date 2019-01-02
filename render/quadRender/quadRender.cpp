#define QT_NO_KEYWORDS
#include <render/quadRender/quadRender.h>
#include <simulation/particleSystem.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <render/quadRender/loader.h>
#include <render/util/camera.h>

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

bool QuadRender::valid() { return true; }

void QuadRender::update() {}

QuadRender::QuadRender(OGLWidget *parent) {
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
  prepCUDAscene();
  update();
}
#include <openvdb/openvdb.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/VolumeToMesh.h>



class ParticlesList
{
protected:
	struct Particle
	{
		openvdb::Vec3R p, v;
		openvdb::Real r;
	};

	openvdb::Real mRadiusScale;
	openvdb::Real mVelocityScale;
	std::vector<Particle> mParticleList;

public:
	typedef openvdb::Vec3R  PosType;
	typedef openvdb::Vec3R  value_type;

	ParticlesList(openvdb::Real rScale = 1, openvdb::Real vScale = 1):mRadiusScale(rScale), mVelocityScale(vScale){}

	void add(const openvdb::Vec3R &p, const openvdb::Real &r, const openvdb::Vec3R &v = openvdb::Vec3R(0, 0, 0))
	{
		Particle part;
		part.p = p;
		part.r = r;
		part.v = v;
		mParticleList.push_back(part);
	}

	openvdb::CoordBBox getBBox(const openvdb::GridBase& grid)
	{
		openvdb::CoordBBox bbox;
		openvdb::Coord &min = bbox.min(), &max = bbox.max();
		openvdb::Vec3R pos;
		openvdb::Real rad, invDx = 1 / grid.voxelSize()[0];
		for (size_t n = 0, e = this->size(); n < e; ++n) {
			this->getPosRad(n, pos, rad);
			const openvdb::Vec3d xyz = grid.worldToIndex(pos);
			const openvdb::Real   r = rad * invDx;
			for (int i = 0; i < 3; ++i) {
				min[i] = openvdb::math::Min(min[i], openvdb::math::Floor(xyz[i] - r));
				max[i] = openvdb::math::Max(max[i], openvdb::math::Ceil(xyz[i] + r));
			}
		}
		return bbox;
	}

	openvdb::Vec3R pos(int n)   const { return mParticleList[n].p; }
	openvdb::Vec3R vel(int n)   const { return mVelocityScale * mParticleList[n].v; }
	openvdb::Real radius(int n) const { return mRadiusScale * mParticleList[n].r; }

	size_t size() const { return mParticleList.size(); }

	void getPos(size_t n, openvdb::Vec3R& pos) const { pos = mParticleList[n].p; }

	void getPosRad(size_t n, openvdb::Vec3R& pos, openvdb::Real& rad) const
	{
		pos = mParticleList[n].p;
		rad = mRadiusScale * mParticleList[n].r;
	}

	void getPosRadVel(size_t n, openvdb::Vec3R& pos, openvdb::Real& rad, openvdb::Vec3R& vel) const {
		pos = mParticleList[n].p;
		rad = mRadiusScale * mParticleList[n].r;
		vel = mVelocityScale * mParticleList[n].v;
	}

	void getAtt(size_t n, openvdb::Index32& att) const { att = openvdb::Index32(n); }
};

void QuadRender::render() {
  static std::random_device r;
  static std::default_random_engine e1(r());
  static std::uniform_int_distribution<int32_t> uniform_dist(INT_MIN, INT_MAX);
  static int framenumber = 0;
  auto h_scene = hostScene();
  if (h_scene.dirty) {
    cudaMemset(accumulatebuffer, 1, h_scene.width * h_scene.height * sizeof(float3));
    framenumber = 0;
  }
  framenumber++;

  static auto frame = get<parameters::frame>();
  if (frame != get<parameters::frame>()) {
	  auto numptcls = get<parameters::num_ptcls>();
	  openvdb::Vec4f* positions = new openvdb::Vec4f[numptcls];
	  openvdb::Vec4f* velocities = new openvdb::Vec4f[numptcls];
	  float* volumes = new float[numptcls];
	  float scale = 0.01f;

	  cudaMemcpy(positions, arrays::position::ptr, sizeof(float4) * numptcls, cudaMemcpyDeviceToHost);
	  cudaMemcpy(volumes, arrays::volume::ptr, sizeof(float) * numptcls, cudaMemcpyDeviceToHost);
	  cudaMemcpy(velocities, arrays::velocity::ptr, sizeof(float4) * numptcls, cudaMemcpyDeviceToHost);
	  float* pscale = new float[numptcls];
	  for (int32_t i = 0; i < numptcls; ++i) {
		  //positions[i] *= 0.01f;
		  //velocities[i] *= 0.01f;
		  //pscale[i] = positions[i].w() * 0.5f;
		  pscale[i] = powf(volumes[i] / (4.f / 3.f * CUDART_PI_F), 1.f / 3.f);// *0.01f;
	  }
	  // add points to particle list
	  ParticlesList pl(1, 1);
	  const float minParticleSize = get<parameters::radius>() * scale;
	  for (int32_t i = 0; i < numptcls; ++i) {
		  auto p = positions[i] * scale;
		  auto r = pscale[i] * scale * 0.5f;
		  auto v = velocities[i] * scale;
		 // pl.add(openvdb::Vec3R(p.x(), p.y(), p.z()), openvdb::Real(r), openvdb::Vec3R(v.x(), v.y(), v.z()));
		  pl.add(openvdb::Vec3R(p.x(), p.y(), p.z()), openvdb::Real(r), openvdb::Vec3R(0,0,0));
	  }

	 // openvdb::FloatGrid::Ptr ls = openvdb::FloatGrid::create(
	//	  /*background=*/static_cast<float>(voxelSize));
	  //ls->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
	  //ls->setGridClass(openvdb::GRID_LEVEL_SET);

	  openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(minParticleSize*0.33333f, 3.0);
	  openvdb::tools::particlesToSdf(pl, *ls);
	  std::cout << ls->activeVoxelCount() << std::endl;
	  //openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> raster(*ls);
	  //std::cout << "Rasterizing particles" << std::endl;
	  //raster.setGrainSize(1);//a value of zero disables threading
	  //raster.rasterizeSpheres(pl);
	  //raster.finalize();
	  //raster.rasterizeSpheres(pl);
	  //raster.rasterizeTrails(pl, 0.75);//scale offset between two instances

	  openvdb::tools::LevelSetFilter<openvdb::FloatGrid> filterer(*ls);
	  //filterer.setGrainSize(0);
	  openvdb::tools::LevelSetTracker<openvdb::FloatGrid> tracker(*ls);
	  //tracker.setGrainSize(0);
#define PRINT(x) std::cout << #x << std::endl; x;
	  PRINT(filterer.offset(minParticleSize));
	  PRINT(filterer.track());
	  PRINT(tracker.dilate());
	  PRINT(tracker.track());
	  //PRINT(tracker.resize(3.0));
	  PRINT(filterer.mean(2));
	  PRINT(filterer.mean(2));
	  PRINT(filterer.track());
	  PRINT(tracker.erode());
	  //PRINT(tracker.resize(1.5));
	  PRINT(tracker.track());
	  PRINT(filterer.meanCurvature());
	  PRINT(filterer.meanCurvature());
	  PRINT(filterer.meanCurvature());
	  PRINT(filterer.meanCurvature());
	  PRINT(filterer.track());

	  std::vector<openvdb::Vec3s> points;
	  std::vector<openvdb::Vec4I> quads;
	  std::vector<openvdb::Vec3I> triangles;

	  //openvdb::tools::volumeToMesh(*ls, points, triangles, quads, 0.01, 0);
	  std::cout << "Meshing particles to level set" << std::endl;
	  openvdb::tools::volumeToMesh(*ls, points, triangles, quads, 0.0, 0.083);

	  std::cout << "points " << points.size() << std::endl;
	  std::cout << "triangles " << triangles.size() << std::endl;
	  std::cout << "quads " << quads.size() << std::endl;
	  std::vector<Vertex> verts;
	  std::vector<Triangle> tris;

	  auto getNormal = [](auto e0, auto e1, auto e2) {
		  auto e1_e0 = e1 - e0;
		  auto e2_e0 = e2 - e0;
		  auto n = math::cross(e1_e0, e2_e0);
		  auto l = math::length3(n);
		  if (l > 1e-7f)
			  return n / l;
		  else
			  return float4{ 0.f, 1.f, 0.f,0.f };
	  };
	  for (auto& v : points) {
		  verts.push_back(Vertex{ float3{v.x() / scale, v.y() / scale, v.z() / scale }, float3{0.f, 0.f, 0.f} });
	  }
	  for (auto& t : triangles) {
		  auto i0 = (int32_t)t[0];
		  auto i1 = (int32_t)t[1];
		  auto i2 = (int32_t)t[2];
		  auto e0 = verts[i0].position;
		  auto e1 = verts[i1].position;
		  auto e2 = verts[i2].position;

		  auto n1 = getNormal(e0, e1, e2);
		  verts[i0].normal += n1;
		  verts[i1].normal += n1;
		  verts[i2].normal += n1;

		  tris.push_back(Triangle(t.x(), t.y(), t.z(), verts));
	  }
	  for (auto& q : quads) {
		  auto i0 = (int32_t)q[0];
		  auto i1 = (int32_t)q[1];
		  auto i2 = (int32_t)q[2];
		  auto i3 = (int32_t)q[3];
		  auto e0 = verts[i0].position;
		  auto e1 = verts[i1].position;
		  auto e2 = verts[i2].position;
		  auto e3 = verts[i3].position;

		  auto n1 = getNormal(e0, e1, e2);
		  auto n2 = getNormal(e2, e3, e0);
		  verts[i0].normal += n1 + n2;
		  verts[i1].normal += n1;
		  verts[i2].normal += n1 + n2;
		  verts[i3].normal += n2;

		  tris.push_back(Triangle(q.x(), q.y(), q.z(), verts));
		  tris.push_back(Triangle(q.z(), q.w(), q.x(), verts));
	  }
	  for (auto &v : verts) {
		  v.normal = math::normalize3(v.normal);
	  }
	  mesh m{ verts, tris };
	  fluidLoader.active = true;
	  std::cout << "Resetting fluid BVH" << std::endl;
	  fluidLoader.reset();
	  //fluidLoader.appendObject("C:/dev/source/maelstrom/Configurations/DamBreak/Volumes/Adaptive.vdb");
	  std::cout << "Appending mesh to fluid BVH" << std::endl;
	  fluidLoader.appendMesh(m);
	  std::cout << "Merging meshes in fluid BVH" << std::endl;
	  fluidLoader.mergeMeshes();
	  std::cout << "Building fluid BVH" << std::endl;
	  fluidLoader.buildBVH();
	  std::cout << "Done with fluid BVH construction" << std::endl;

	  frame = get<parameters::frame>();
	  delete[] positions;
	  delete[] velocities;
	  delete[] pscale;
	  delete[] volumes;

	  cudaMemset(accumulatebuffer, 1, h_scene.width * h_scene.height * sizeof(float3));
	  framenumber = 1;
  }

  cudaRender(h_scene, renderedResourceOut, loader, fluidLoader, accumulatebuffer, framenumber, uniform_dist(e1));

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

void QuadRender::prepCUDAscene() {
  auto scenefile = "C:/dev/source/maelstrom/Configurations/DamBreak/Volumes/pillars.vdb";
  loader.appendObject(scenefile);
  objects = loader.mergeMeshes();
  loader.buildBVH();

  //fluidLoader.active = true;
  //fluidLoader.appendObject("C:/dev/source/maelstrom/Configurations/DamBreak/Volumes/Adaptive.vdb");
  ////fluidLoader.appendMesh(m);
  //fluidLoader.buildBVH();
  //fluidLoader.active = true;

  LOG_INFO << "Rendering data initialised and copied to CUDA global memory\n" << std::endl;
}