#define QT_NO_KEYWORDS
#include <render/vrtxRender/vrtxRender.h>
#include <simulation/particleSystem.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <render/util/camera.h>
#include <QApplication>
#include <bitset>
#include <iomanip>
#include <render/qGLWidget/base_renderer.h>

vrtx::Refl_t intToMaterial(int32_t v){
	switch (v) {
	case 0:
		return vrtx::Refl_t::DIFF;
	case 1:
		return vrtx::Refl_t::COAT;
	case 2:
		return vrtx::Refl_t::METAL;
	case 3:
		return vrtx::Refl_t::SPEC;
	case 4:
		return vrtx::Refl_t::REFR;
	default:
		return vrtx::Refl_t::DIFF;
	}
}
std::string materialToString(vrtx::Refl_t  v) {
	switch (v) {
	case vrtx::Refl_t::DIFF:	return "diffuse";
	case vrtx::Refl_t::COAT:	return "coated";
	case vrtx::Refl_t::METAL:	return "metallic";
	case vrtx::Refl_t::SPEC:	return "specular";	
	case vrtx::Refl_t::REFR:	return "refractive";
	default: return "diffuse";
	}
}

std::string vRTXrender::getInformation() {
	if (!valid() || !get<parameters::vrtxDisplayStats>()) return "";
	std::stringstream sstream;
	//switch (renderMode) {
	//case 0: sstream << std::string("vRTX [") + std::to_string(framenumber) + "]\n"; break;
	//case 1: sstream << std::string("vRTX normal debug [") + std::to_string(framenumber) + "]\n"; break;
	//case 2: sstream << std::string("vRTX transparent [") + std::to_string(framenumber) + "]\n"; break;
	//}
	sstream << "vRTX bounces: " << get<parameters::vrtxBounces>() << "\n";
	sstream << "vRTX material: " << materialToString(intToMaterial(get<parameters::vrtxMaterial>())) << "\n";
	sstream << "vRTX fluid: " << (get<parameters::vrtxRenderFluid>() ? "yes" : "no") << "\n";
	sstream << "vRTX normals: " << (get<parameters::vrtxRenderNormals>() ? "yes" : "no") << "\n";
	sstream << "vRTX surface: " << (get<parameters::vrtxRenderSurface>() ? "yes" : "no") << "\n";
	for (auto& [timer,times] : timings) {
		if (std::all_of(times.begin(), times.end(), [](auto v) {return v > 0; })) {
			sstream << "vRTX " << timer << " took ";
			for(auto t: times) sstream << std::setw(8) << std::fixed << std::setprecision(2) << t << "ms\t";
		}
		else {
			sstream << "vRTX " << timer << "      ";
				for (auto t : times) sstream << std::setw(8) << std::fixed << (int32_t) -t << "  \t";
		}
		sstream << "\n";
	}
	return sstream.str();
}
void vRTXrender::keyPressEvent(QKeyEvent *event) {
	switch (event->key()) {
	case Qt::Key_F11: {
		if (QApplication::keyboardModifiers().testFlag(Qt::ControlModifier))
			get<parameters::vrtxRenderFluid>() = !get<parameters::vrtxRenderFluid>();
		else if (QApplication::keyboardModifiers().testFlag(Qt::ShiftModifier))
			get<parameters::vrtxRenderSurface>() = !get<parameters::vrtxRenderSurface>();
		else
			get<parameters::vrtxRenderNormals>() = !get<parameters::vrtxRenderNormals>();
		dirty = true;
		break;
	}
	case Qt::Key_F10: {
		if (QApplication::keyboardModifiers().testFlag(Qt::ControlModifier))
			fluidMaterial = fluidMaterial == vrtx::Refl_t::DIFF ? vrtx::Refl_t::REFR : vrtx::Refl_t::DIFF;
		else if (QApplication::keyboardModifiers().testFlag(Qt::ShiftModifier))
			bounces = std::max(1, bounces--);
		else
			bounces++;
		dirty = true;
		break;
	}
	}
}
#include <utility/helpers/pathfinder.h>

void vRTXrender::updateRTX() {
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

	static std::string old_colormap = "";
	static size_t old_size = 0;
	if (old_colormap != get<parameters::render_colormap>()) {

		old_colormap = get<parameters::render_colormap>();

		//std::string file_name = root_folder + get<parameters::render_colormap>() + ".png";
		QImage img(1024, 1, QImage::Format_RGB32);
		//std::cout << file_name << " exists: " << fs::exists(file_name) << std::endl;
		for (int32_t it = 0; it < img.width(); ++it)
			img.setPixel(QPoint(it, 0), qRgb((float)it / (float)img.width() * 256.f,
			(float)it / (float)img.width() * 256.f,
				(float)it / (float)img.width() * 256.f));
		try {
			std::string file_name = resolveFile(std::string("cfg/") + get<parameters::render_colormap>() + ".png").string();
			if (std::experimental::filesystem::exists(file_name)) {
				//std::cout << "Loading " << file_name << std::endl;
				img = QImage(QString::fromStdString(file_name));
				//img.load(QString(file_name.c_str()));
				//std::cout << img.width() << " : " << img.height() << std::endl;
			}
		}
		catch (...) {}
		color_map = (float4 *)realloc(color_map, sizeof(float4) * (img.width()));
		for (int32_t it = 0; it < img.width(); ++it) {
			QColor col = img.pixelColor(QPoint(it, 1));
			color_map[it] = float4{ (float)(col.red()) / 256.f, (float)(col.green()) / 256.f,
									  (float)(col.blue()) / 256.f, 1.f };
			//if (it == img.width() - 1)
			//	color_map[it + 1] = float4{ (float)(col.red()) / 256.f, (float)(col.green()) / 256.f,
			//	(float)(col.blue()) / 256.f, 1.f };
		}
			color_map_elements = img.width();
			if (cu_color_map != nullptr)
				cudaFree(cu_color_map);
			cudaMalloc(&cu_color_map, sizeof(float4) * color_map_elements);
			cudaMemcpy(cu_color_map, color_map, sizeof(float4) * color_map_elements, cudaMemcpyHostToDevice);
			dirty = true;
		
	}
	static int32_t lastSurfacing = -1;
	static int32_t lastFluid = -1;
	static int32_t lastNormal = -1;
	static int32_t lastBVH = -1;
	static int32_t lastMaterial = -1;
	static int32_t lastBVHMaterial = -1;
	static int32_t lastBounces = -1;
	static float lastIOR = 0.f;
	static float3 lastFluidColor = float3{ 0.f,0.f,0.f };
	static float3 lastDebeerColor = float3{ 0.f,0.f,0.f };
	static float lastDebeerScale = 0.f;
	static float lastBias = -1.f;
	static int32_t lastSurfaceExtraction = -1; ;
	static int32_t lastRenderGrid = -1;
	static int32_t lastFlipped = -1;
	static int32_t oldTransferFn = -1;
	static int32_t oldMapFn = -1;
	static float old_min = -1.f;
	static float old_max = -1.f;
	static float lastR = -1.f;
	static float lastWmin = -1.f;
	static int32_t lastDepth = -1;
	static float lastDepthScale = -1.f;
	static float lastEpsilon = -1.f;
	static float3 lastbvhColor{ -1.f,1.f,2.f };
	if (lastSurfacing != get<parameters::vrtxRenderSurface>() ||
		lastFluid != get<parameters::vrtxRenderFluid>() ||
		lastNormal != get<parameters::vrtxRenderNormals>() ||
		lastMaterial != get<parameters::vrtxMaterial>() ||
		lastBounces != get<parameters::vrtxBounces>() ||
		lastBVH != get<parameters::vrtxRenderBVH>() ||
		lastIOR != get<parameters::vrtxIOR>() ||
		lastFluidColor.x != get<parameters::vrtxFluidColor>().x ||
		lastFluidColor.y != get<parameters::vrtxFluidColor>().y ||
		lastFluidColor.z != get<parameters::vrtxFluidColor>().z ||
		lastDebeerColor.x != get < parameters::vrtxDebeer>().x ||
		lastDebeerColor.y != get < parameters::vrtxDebeer>().y ||
		lastDebeerColor.z != get < parameters::vrtxDebeer>().z ||
		lastDebeerScale != get < parameters::vrtxDebeerScale>() ||
		lastBVHMaterial != get<parameters::vrtxBVHMaterial>() ||
		lastBias != get< parameters::vrtxFluidBias>() ||
		lastSurfaceExtraction != get<parameters::vrtxSurfaceExtraction>() ||
		lastRenderGrid != get<parameters::vrtxRenderGrid>() ||
		lastFlipped != get<parameters::color_map::map_flipped>() ||
		old_min != get<parameters::render_min>() ||
		old_max != get<parameters::render_max>() ||
		oldTransferFn != get<parameters::color_map::transfer_fn>() ||
		oldMapFn != get<parameters::color_map::mapping_fn>() ||
		lastR != get<parameters::vrtxR>() ||
		lastWmin != get<parameters::vrtxWMin>() ||
		lastDepth != get<parameters::vrtxDepth>() ||
		lastDepthScale != get<parameters::vrtxDepthScale>() ||
		lastEpsilon != get<parameters::vrtxDomainEpsilon>() ||
		lastbvhColor != get<parameters::bvhColor>()
		) {
		lastSurfacing = get<parameters::vrtxRenderSurface>();
		lastFluid = get<parameters::vrtxRenderFluid>();
		lastNormal = get<parameters::vrtxRenderNormals>();
		lastMaterial = get<parameters::vrtxMaterial>();
		lastBounces = get<parameters::vrtxBounces>();
		lastIOR = get<parameters::vrtxIOR>();
		lastBVH = get<parameters::vrtxRenderBVH>();
		lastFluidColor = parameters::vrtxFluidColor{};
		lastDebeerColor = parameters::vrtxDebeer{};
		lastDebeerScale = parameters::vrtxDebeerScale{};
		lastBVHMaterial = get<parameters::vrtxBVHMaterial>();
		lastBias = parameters::vrtxFluidBias{};
		lastSurfaceExtraction = get<parameters::vrtxSurfaceExtraction>();
		lastRenderGrid = get<parameters::vrtxRenderGrid>();
		lastFlipped = get<parameters::color_map::map_flipped>();
		old_min = get<parameters::render_min>();
		old_max = get<parameters::render_max>();
		oldTransferFn = get<parameters::color_map::transfer_fn>();
		oldMapFn = get<parameters::color_map::mapping_fn>();
		lastR = get<parameters::vrtxR>();
		lastWmin = get<parameters::vrtxWMin>();
		lastDepth = get<parameters::vrtxDepth>();
		lastDepthScale = get<parameters::vrtxDepthScale>();
		lastEpsilon = get<parameters::vrtxDomainEpsilon>();
		lastbvhColor = get<parameters::bvhColor>();
		//if (lastRenderGrid) lastFluid = 0;
		dirty = true;
	}
	loadBoxes();
	loadSpheres();
}
void vRTXrender::loadSpheres() {
	if (get<parameters::rtxSpheres>().size() == 0) {
		spheres = std::vector<vrtx::Sphere>{
			//vrtx::Sphere{16, {192.0f, 192, 192}, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f}, vrtx::DIFF},
			vrtx::Sphere{32, {-96, 0, 16}, {0, 0, 0}, {1.f, 1.f, 1.f}, vrtx::SPEC},
			vrtx::Sphere{32, {-96, -64, 16}, {0, 0, 0}, {0.5f, 0.f, 0.f}, vrtx::DIFF},
			vrtx::Sphere{32, {-96, 64, 64}, {0, 0, 0}, {1.0f, 1.f, 1.f}, vrtx::REFR},
			vrtx::Sphere{10000, {50.0f, 40.8f, -1060}, {0.55f, 0.55f, 0.55f}, {0.075f, 0.075f, 0.075f}, vrtx::DIFF},
			//vrtx::Sphere{10000, {50.0f, 40.8f, -1060}, {0.55, 0.55, 0.55}, {0.175f, 0.175f, 0.175f}, vrtx::DIFF},
			//vrtx::Sphere{10000, {50.0f, 40.8f, -1060}, {0.f,0.f,0.f}, {0.f,0.f,0.f}, vrtx::DIFF},

			vrtx::Sphere{100000, {0.0f, 0, -100000.f}, {0, 0, 0}, {0.2f, 0.2f, 0.2f}, vrtx::DIFF},
			vrtx::Sphere{100000, {0.0f, 0, -100000.1f}, {0, 0, 0}, {0.3f, 0.3f, 0.3f}, vrtx::DIFF} };
		return;
	}
	auto eq = [](auto lhs, auto rhs) {return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; };
	if(spheres.size() != get<parameters::rtxSpheres>().size())
		spheres.resize(get<parameters::rtxSpheres>().size());
	int32_t i = 0;
	for (const auto& sphere : get<parameters::rtxSpheres>()) {
		float3 color = sphere.color.value;
		float3 emission = sphere.emission.value;
		float3 position = sphere.position.value;
		float radius = sphere.radius.value;
		vrtx::Refl_t material = intToMaterial(sphere.refl_t.value);
		if (!eq(color, spheres[i].col) || !eq(emission, spheres[i].emi) || !eq(position, spheres[i].pos) || radius != spheres[i].rad || material != spheres[i].refl) {
			spheres[i].col = color;
			spheres[i].emi = emission;
			spheres[i].pos = position;
			spheres[i].rad = radius;
			spheres[i].refl = material;
			boxesDirty = true;
			dirty = true;
		}
		i++;
	}
}
#include <sstream>
void vRTXrender::loadBoxes() {
	if (get<parameters::rtxBoxes>().size() == 0) {
		boxes = std::vector<vrtx::Box>{
			//vrtx::Box{{-25.f, -25.f, 96.f},{25.f,25.f, 132.f},{1.f,1.f,1.f}, {0.f,0.f,0.f}, vrtx::DIFF},
			vrtx::Box{{190.f, -192.f, -192.f},{192.f,192.f, 192.f},{1.f,1.f,1.f}, {0.f,0.f,0.f}, vrtx::DIFF}
			,vrtx::Box{{-521, -FLT_MAX, -FLT_MAX},{-51, FLT_MAX, FLT_MAX},{0.f,0.f,0.f}, {1.f, 1.f, 1.f},vrtx::DIFF}
			//,vrtx::Box{ {-FLT_MAX, -25.f, -FLT_MAX},{32, FLT_MAX, FLT_MAX},{0.f,0.f,0.f}, {1.f, 1.f, 1.f}, vrtx::DIFF}
		};
		return;
	}
	auto strToFloat3 = [](auto str) {
		std::istringstream iss(str);
		std::string a, b, c;
		iss >> a >> b >> c;
		float x = (a == "FLT_MAX" ? FLT_MAX : (a == "-FLT_MAX" ? -FLT_MAX : std::stof(a)));
		float y = (b == "FLT_MAX" ? FLT_MAX : (b == "-FLT_MAX" ? -FLT_MAX : std::stof(b)));
		float z = (c == "FLT_MAX" ? FLT_MAX : (c == "-FLT_MAX" ? -FLT_MAX : std::stof(c)));
		return float3{ x, y, z };
	};
	auto eq = [](auto lhs, auto rhs) {return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; };
	if (boxes.size() != get<parameters::rtxBoxes>().size())
		boxes.resize(get<parameters::rtxBoxes>().size());
	int32_t i = 0;
	for (const auto& sphere : get<parameters::rtxBoxes>()) {
		float3 color = sphere.color.value;
		float3 emission = sphere.emission.value;
		float3 minPosition = strToFloat3(sphere.minPosition.value);
		float3 maxPosition = strToFloat3(sphere.maxPosition.value);
		vrtx::Refl_t material = intToMaterial(sphere.refl_t.value);
		if (!eq(color, boxes[i].col) || !eq(emission, boxes[i].emi) || !eq(minPosition, boxes[i].min) || !eq(maxPosition, boxes[i].max) || material != boxes[i].refl) {
			boxes[i].col = color;
			boxes[i].emi = emission;
			boxes[i].min = minPosition;
			boxes[i].max = maxPosition;
			boxes[i].refl = material;
			boxesDirty = true;
			dirty = true;
		}
		i++;
	}

}

vRTXrender::vRTXrender(OGLWidget *parent) : RTXRender(parent) {
  cuda_particleSystem::instance().retainArray("auxHashMap");
  cuda_particleSystem::instance().retainArray("auxCellSpan");
 // cuda_particleSystem::instance().retainArray("auxCellInformation");
  //cuda_particleSystem::instance().retainArray("auxCellSurface");
  //cuda_particleSystem::instance().retainArray("auxIsoDensity");

  cuda_particleSystem::instance().retainArray("renderArray");
  cuda_particleSystem::instance().retainArray("compactCellSpan");
  cuda_particleSystem::instance().retainArray("compactHashMap");
  cuda_particleSystem::instance().retainArray("MLMResolution");
  if (parameters::anisotropicSurface{}) {
	  cuda_particleSystem::instance().retainArray("centerPosition");
	  cuda_particleSystem::instance().retainArray("anisotropicMatrices");
  }
  cuda_particleSystem::instance().retainArray("density");
  //std::cout << get<parameters::renderMode>()
  renderMode = get<parameters::renderMode>() % renderModes;
  if (renderMode == 2) {
	  fluidMaterial = vrtx::Refl_t::REFR;
  }
  else
	  fluidMaterial = vrtx::Refl_t::DIFF;
  updateRTX();
}
void vRTXrender::renderRTX(bool prettyRender, int32_t fn, int32_t s) {
	static std::vector<Matrix4x4> Txs;
	//std::cout << "vrtxRender [" << framenumber << "] @ renderMode " << renderMode << ", simulation frame: " << get<parameters::frame>() << std::endl;
	static bool once = true;
	static int32_t frame = -1;
	if (once || frame != get<parameters::frame>()) {
		bool update = once;
		if (once) {
			for (int32_t b = 0; b < get<parameters::volumeBoundaryCounter>(); ++b) {
				Matrix4x4 Tx;
				cudaMemcpy(&Tx, arrays::volumeBoundaryTransformMatrix::ptr + b, sizeof(Matrix4x4), cudaMemcpyDeviceToHost);
				Txs.push_back(Tx);
			}
		}
		if (frame != get<parameters::frame>()) {
			for (int32_t b = 0; b < get<parameters::volumeBoundaryCounter>(); ++b) {
				Matrix4x4 Tx;
				cudaMemcpy(&Tx, arrays::volumeBoundaryTransformMatrix::ptr + b, sizeof(Matrix4x4), cudaMemcpyDeviceToHost);
				for(int32_t i = 0; i < 16; ++i)
					if(Tx.data[i] != Txs[b].data[i])
						update = true;
			}
		}
		frame = get<parameters::frame>();
		if (update) {
			if (!once) {
				loader.reset();
			}
			for (int32_t b = 0; b < get<parameters::volumeBoundaryCounter>(); ++b) {
				loader.appendObject(b);
			}
			objects = loader.mergeMeshes();
			loader.tearDownMeshes();
			loader.buildBVH();
		}
		once = false;
	}
  vrtxFluidArrays farrays;
  farrays.min_coord = parameters::min_coord{};
  farrays.max_coord = get<parameters::max_coord>() +get<parameters::cell_size>();
  farrays.cell_size = parameters::cell_size{};
  farrays.grid_size = parameters::grid_size{};
  farrays.hash_entries = parameters::hash_entries{};
  farrays.mlm_schemes = parameters::mlm_schemes{};
  farrays.num_ptcls = parameters::num_ptcls{};
  farrays.max_numptcls = parameters::max_numptcls{};
  farrays.timestep = parameters::timestep{};
  farrays.renderRadius = parameters::radius{} *parameters::vrtxWMin{};//(get<parameters::modules::adaptive>() ? 1.f/powf((float)get<parameters::adaptive::resolution>(), 1.f/3.f) : 1.f);
  //farrays.renderRadius = parameters::radius{};
  farrays.rest_density = parameters::rest_density{};

  farrays.compactHashMap = arrays::compactHashMap::ptr;
  farrays.compactCellSpan = arrays::compactCellSpan::ptr;
  farrays.MLMResolution = arrays::MLMResolution::ptr;
  farrays.position = arrays::position::ptr;
  farrays.volume = arrays::volume::ptr;
  farrays.density = arrays::density::ptr;
  farrays.renderArray = arrays::renderArray::ptr;
  farrays.auxIsoDensity = arrays::auxIsoDensity::ptr;

  farrays.centerPosition = arrays::centerPosition::ptr;
  farrays.anisotropicMatrices = arrays::anisotropicMatrices::ptr;

  farrays.minMap = parameters::render_min{};
  farrays.maxMap = parameters::render_max{};
  farrays.transferFn = parameters::transfer_fn{};

  vrtxFluidMemory fmem;
  fmem.grid_size = parameters::grid_size{};
  fmem.cell_size = parameters::cell_size{};
  fmem.min_coord = parameters::min_coord{};
  fmem.max_coord = get<parameters::max_coord>() + get<parameters::cell_size>() * get<parameters::auxScale>();

  fmem.hash_entries = parameters::hash_entries{};
  fmem.mlm_schemes = parameters::mlm_schemes{};

  fmem.num_ptcls = parameters::num_ptcls{};
  fmem.max_numptcls = parameters::max_numptcls{};
  fmem.timestep = parameters::timestep{};
  fmem.renderRadius = parameters::radius{} * parameters::vrtxWMin{};// (get<parameters::modules::adaptive>() ? 1.f / powf((float)get<parameters::adaptive::resolution>(), 1.f / 3.f) : 1.f);
  //fmem.renderRadius = parameters::radius{};
  fmem.rest_density = parameters::rest_density{};
  fmem.fluidBias = parameters::vrtxFluidBias{};

  fmem.cellSpan = arrays::auxCellSpan::ptr;
  fmem.hashMap = arrays::auxHashMap::ptr;

  fmem.vrtxR = parameters::vrtxR{};
  fmem.bounces = parameters::vrtxBounces{};

  fmem.wmin = parameters::vrtxWMin{};
  fmem.wmax = parameters::vrtxWMax{};

  fmem.bvhColor = parameters::bvhColor{};
  fmem.vrtxDepth = parameters::vrtxDepth{};
  fmem.vrtxDepthScale = parameters::vrtxDepthScale{};
  fmem.colorMapFlipped = parameters::color_map::map_flipped{};
  fmem.colorMap = cu_color_map;
  fmem.colorMapLength = color_map_elements - 1;
  fmem.IOR = get<parameters::vrtxIOR>();

  fmem.auxScale = parameters::auxScale{};
  //fmem.cell_size *= fmem.auxScale;
  //fmem.grid_size /= static_cast<int32_t>(fmem.auxScale);
  //fmem.grid_size += 1;
  fmem.cell_size_actual = parameters::cell_size{};
  fmem.grid_size_actual = parameters::grid_size{};

  fmem.maxZ_coordx = (position_to_morton_32(fmem.max_coord, fmem) & 0b001001001001001001001001001001u);
  fmem.maxZ_coordy = (position_to_morton_32(fmem.max_coord, fmem) & 0b010010010010010010010010010010u);
  fmem.maxZ_coordz = (position_to_morton_32(fmem.max_coord, fmem) & 0b100100100100100100100100100100u);

  fmem.vrtxFluidColor = parameters::vrtxFluidColor{};
  fmem.vrtxDebeer = parameters::vrtxDebeer{};
  fmem.vrtxDebeerScale = parameters::vrtxDebeerScale{};
  fmem.bvhMaterial = intToMaterial(get<parameters::vrtxBVHMaterial>());
  fmem.surfaceTechnique = parameters::vrtxSurfaceExtraction{};

  fmem.vrtxDomainMin = parameters::vrtxDomainMin{};
  fmem.vrtxDomainMax = parameters::vrtxDomainMax{};
  fmem.vrtxDomainEpsilon = parameters::vrtxDomainEpsilon{};

  //std::cout << fmem.maxZ_coordx << std::endl;
  //std::cout << fmem.maxZ_coordy << std::endl;
  //std::cout << fmem.maxZ_coordz << std::endl;

  timings = cuVRTXRender(hostScene(), renderedResourceOut, loader, fmem, farrays, accumulatebuffer, 
	  fn, s, renderMode, bounces, fluidRender, get<parameters::vrtxRenderGrid>(), get<parameters::vrtxSurfaceExtraction>(), intToMaterial(get<parameters::vrtxMaterial>()),
	  boxesDirty, spheres, boxes);
  boxesDirty = false;
}