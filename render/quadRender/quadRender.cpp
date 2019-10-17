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


void QuadRender::updateRTX() {}

QuadRender::QuadRender(OGLWidget *parent) : RTXRender(parent) {
	if (get<parameters::rayTracing>() == false) return;
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

void QuadRender::renderRTX(bool pretty, int32_t fn, int32_t s) {
	if (get<parameters::rayTracing>() == false) return;
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

  static auto frame = -1;
  if (frame != get<parameters::frame>() && get<parameters::frame>() > 0) {
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
	  ParticlesList pl(1.f, 1);
	  const float minParticleSize = get<parameters::radius>() * scale;
	  for (int32_t i = 0; i < numptcls; ++i) {
		  auto p = positions[i] * scale;
		  auto r = pscale[i] * scale;
		  auto v = velocities[i] * scale;
		 // pl.add(openvdb::Vec3R(p.x(), p.y(), p.z()), openvdb::Real(r), openvdb::Vec3R(v.x(), v.y(), v.z()));
		  pl.add(openvdb::Vec3R(p.x(), p.y(), p.z()), openvdb::Real(r), openvdb::Vec3R(0,0,0));
	  }

	 // openvdb::FloatGrid::Ptr ls = openvdb::FloatGrid::create(
	//	  /*background=*/static_cast<float>(voxelSize));
	  //ls->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
	  //ls->setGridClass(openvdb::GRID_LEVEL_SET);
	  float halfbandvoxels = 3.f;
	  float mVoxelSize = minParticleSize * 0.60f;
	  //mVoxelSize = 10.f * minParticleSize;
	  //mVoxelSize = 0.00225f;
	  float background = mVoxelSize * halfbandvoxels;

	  auto transform = openvdb::math::Transform::createLinearTransform(mVoxelSize);

	  auto sdfGrid = openvdb::FloatGrid::create(background);
	  sdfGrid->setGridClass(openvdb::GRID_LEVEL_SET);
	  sdfGrid->setTransform(transform);

	  openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> raster(*sdfGrid);
	  raster.setRmin(0.f);
	  raster.setRmax(1e15f);

	  raster.rasterizeSpheres(pl);
	  raster.finalize(/*prune=*/true);

	  auto numTooSmall = raster.getMinCount();
	  auto numTooLarge = raster.getMaxCount();

	  std::cout << sdfGrid->activeVoxelCount() << std::endl;
	  openvdb::io::File file("fluid.vdb");

	  openvdb::GridPtrVec grids;
	  grids.push_back(sdfGrid);

#define PRINT(x) std::cout << #x << std::endl; x;
	  openvdb::tools::LevelSetFilter<openvdb::FloatGrid> filter(*sdfGrid);
	  filter.setSpatialScheme(openvdb::math::FIRST_BIAS);
	  filter.setTemporalScheme(openvdb::math::TVD_RK1);
	  filter.setTrimming(openvdb::tools::lstrack::TrimMode::kAll);

	  const float voxelSize = float(filter.grid().voxelSize()[0]);

	  PRINT(filter.offset(-mVoxelSize * 2.f));
	  PRINT(filter.laplacian());
	  PRINT(filter.laplacian());
	  PRINT(filter.offset(mVoxelSize * 2.f));
	  //PRINT(filter.meanCurvature());
	  //PRINT(filter.meanCurvature());
	  //PRINT(filter.meanCurvature());

	  //// Write out the contents of the container.
	  file.write(grids);
	  file.close();
	  //exit(1); 


	  std::vector<openvdb::Vec3s> pts;
	  std::vector<openvdb::Vec4I> quads;
	  std::vector<openvdb::Vec3I> triangles;

	  //openvdb::tools::volumeToMesh(*ls, points, triangles, quads, 0.01, 0);
	  float iso = 0.f;
	  float adaptivity = 0.083f;
	  std::cout << "Meshing particles to level set" << std::endl;
	  openvdb::tools::VolumeToMesh mesher(iso, adaptivity);
	  const openvdb::tools::PointList& points = mesher.pointList();
	  openvdb::tools::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();

	  const char exteriorFlag = char(openvdb::tools::POLYFLAG_EXTERIOR);
	  const char seamLineFlag = char(openvdb::tools::POLYFLAG_FRACTURE_SEAM);

	  mesher(*sdfGrid);
	  auto npoints = mesher.pointListSize();
	  std::cout << npoints << std::endl;

	  // index 0 --> interior, not on seam
	  // index 1 --> interior, on seam
	  // index 2 --> surface,  not on seam
	  // index 3 --> surface,  on seam
	  int32_t nquads[4] = { 0, 0, 0, 0 };
	  int32_t ntris[4] = { 0, 0, 0, 0 };
	  for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
		  const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
		  for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
			  int flags = (((polygons.quadFlags(i) & exteriorFlag) != 0) << 1)
				  | ((polygons.quadFlags(i) & seamLineFlag) != 0);
			  ++nquads[flags];
		  }
		  for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
			  int flags = (((polygons.triangleFlags(i) & exteriorFlag) != 0) << 1)
				  | ((polygons.triangleFlags(i) & seamLineFlag) != 0);
			  ++ntris[flags];
		  }
	  }
	  int32_t nverts[4] = {
		  nquads[0] * 4 + ntris[0] * 3,
		  nquads[1] * 4 + ntris[1] * 3,
		  nquads[2] * 4 + ntris[2] * 3,
		  nquads[3] * 4 + ntris[3] * 3
	  };
	  std::vector<int32_t> verts[4];
	  for (int flags = 0; flags < 4; ++flags) {
		  verts[flags].resize(nverts[flags]);
	  }
	  int32_t iquad[4] = { 0, 0, 0, 0 };
	  int32_t itri[4] = { nquads[0] * 4, nquads[1] * 4, nquads[2] * 4, nquads[3] * 4 };

	  for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
		  const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
		  // Copy quads
		  for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
			  const openvdb::Vec4I& quad = polygons.quad(i);
			  int flags = (((polygons.quadFlags(i) & exteriorFlag) != 0) << 1)
				  | ((polygons.quadFlags(i) & seamLineFlag) != 0);
			  verts[flags][iquad[flags]++] = quad[0];
			  verts[flags][iquad[flags]++] = quad[1];
			  verts[flags][iquad[flags]++] = quad[2];
			  verts[flags][iquad[flags]++] = quad[3];
		  }
		  // Copy triangles (adaptive mesh)
		  for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
			  const openvdb::Vec3I& triangle = polygons.triangle(i);
			  int flags = (((polygons.triangleFlags(i) & exteriorFlag) != 0) << 1)
				  | ((polygons.triangleFlags(i) & seamLineFlag) != 0);
			  verts[flags][itri[flags]++] = triangle[0];
			  verts[flags][itri[flags]++] = triangle[1];
			  verts[flags][itri[flags]++] = triangle[2];
		  }


	  }

	  std::vector<Vertex> vtxs;
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
	  for(int32_t i = 0; i < npoints; ++i){
		  auto& v = points[i];
		  vtxs.push_back(Vertex{ float3{v.x() / scale, v.y() / scale, v.z() / scale }, float3{0.f, 0.f, 0.f} });
	  }

	  for (int flags = 0; flags < 4; ++flags) {
		  if (!nquads[flags] && !ntris[flags]) continue;
		  std::cout << "flag : " << flags << " -> num quads = " << nquads[flags] << std::endl;
		  std::cout << "flag : " << flags << " -> num tris  = " << ntris[flags] << std::endl;
		  std::cout << nquads[flags] << " : " << ntris[flags] << std::endl;
		  for (int32_t i = 0; i < nquads[flags]; ++i) {
			  int32_t i0 = verts[flags][i * 4 + 0];
			  int32_t i1 = verts[flags][i * 4 + 1];
			  int32_t i2 = verts[flags][i * 4 + 2];
			  int32_t i3 = verts[flags][i * 4 + 3];

			  auto e0 = vtxs[i0].position;
			  auto e1 = vtxs[i1].position;
			  auto e2 = vtxs[i2].position;
			  auto e3 = vtxs[i3].position;

			  auto n1 = getNormal(e0, e1, e2);
			  auto n2 = getNormal(e2, e3, e0);
			  vtxs[i0].normal += n1 + n2;
			  vtxs[i1].normal += n1;
			  vtxs[i2].normal += n1 + n2;
			  vtxs[i3].normal += n2;

			  tris.push_back(Triangle(i0, i1, i2, vtxs));
			  tris.push_back(Triangle(i2, i3, i0, vtxs));
		  }
		  for (int32_t i = 0; i < ntris[flags]; ++i) {
			  int32_t i0 = verts[flags][i * 3 + nquads[flags] * 4 + 0];
			  int32_t i1 = verts[flags][i * 3 + nquads[flags] * 4 + 1];
			  int32_t i2 = verts[flags][i * 3 + nquads[flags] * 4 + 2];
			  auto e0 = vtxs[i0].position;
			  auto e1 = vtxs[i1].position;
			  auto e2 = vtxs[i2].position;

			  auto n1 = getNormal(e0, e1, e2);
			  vtxs[i0].normal += n1;
			  vtxs[i1].normal += n1;
			  vtxs[i2].normal += n1;
			  tris.push_back(Triangle(i0,i1,i2, vtxs));
		  }
	  }
	  for (auto &v : vtxs) {
		  v.normal = math::normalize3(v.normal);
	  }
	  std::cout << "num vertices : " << vtxs.size() << std::endl;
	  std::cout << "num triangles: " << tris.size() << std::endl;

	  mesh m{ vtxs, tris };
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
	  cuda_particleSystem::instance().running = false;
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
	//auto scenefile = "C:/dev/source/openMaelstrom/Configurations/DamBreak/Volumes/pillars.obj";
	//loader.appendObject(scenefile);

	auto boundaryVolumes = get<parameters::boundaryVolumes>();
	for (auto& b : boundaryVolumes) {
		loader.appendObject(b.fileName.value);
	}
  objects = loader.mergeMeshes();
  loader.tearDownMeshes();
  loader.buildBVH();


  LOG_INFO << "Rendering data initialised and copied to CUDA global memory\n" << std::endl;
}