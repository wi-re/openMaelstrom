#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <utility/include_all.h>
#include <utility/iterator.h>
#include <utility/math.h>
#include <utility/unit_math.h>
#include <vector>
  

auto generateHexGrid(float h, float r, bool center = true, float scale = 1.0f) {
	float H = h * kernelSize();
	auto gen_position = [&](auto r, int32_t i, int32_t j, int32_t k) {
		float4 initial{ 2.0f * i + ((j + k) % 2), sqrt(3.f) * (j + 1.0f / 3.0f * (k % 2)), 2.0f * sqrt(6.0f) / 3.0f * k, h / r };
		return initial * r;
	};
	int32_t requiredSlices_x = (int32_t)math::ceilf(scale * H / r);
	int32_t requiredSlices_y = (int32_t)math::ceilf(scale * H / (sqrt(3.0f) * r));
	int32_t requiredSlices_z = (int32_t)math::ceilf(scale * H / r * 3.0f / (sqrt(6.0f) * 2.0f));

	std::vector<float4> positions;
	for (int32_t x_it = -requiredSlices_x; x_it <= requiredSlices_x; x_it++)
		for (int32_t y_it = -requiredSlices_y; y_it <= requiredSlices_y; y_it++)
			for (int32_t z_it = -requiredSlices_z; z_it <= requiredSlices_z; z_it++)
				if (center || (!center && (x_it != 0 || y_it != 0 || z_it != 0)))
					positions.push_back(gen_position(r, x_it, y_it, z_it));
	return positions;
}
#define CALC_CONSTANTS
#ifdef CALC_CONSTANTS
constexpr auto volume = 1.f;
auto radius = powf(volume, 1.f / 3.f) * PI4O3_1;
auto h = support_from_volume(volume);
auto H = h * kernelSize();
auto getPacking() {
	int32_t it = 0;
	auto spacing = math::brentsMethod(
		[&](auto r) {
		auto positions = generateHexGrid(h, r, true, 1.0f);
		auto positionsL = generateHexGrid(h, r, true, 2.0f);
		float error = 0.0f;
		for (const auto& pos : positions) {
			float density = -1.0f;
			for (const auto& posL : positionsL)
				density += volume * spline4_kernel(posL, pos);
			error += density;
		}
		std::cout << r << "[" << it++ << "] -> " << error << std::endl;
		return error;
	},
		radius * 0.75f, radius * 8.0f, 1e-5f, 100);
	return spacing;
}
auto spacing = getPacking();
#else
constexpr auto H = 0x1.2487b0p+1f;
constexpr auto h = 0x1.407358p+0f;
constexpr auto r = 0x1.e8ec8ap-3f;
constexpr auto V = 0x1.000000p+0f;
constexpr auto s = 0x1.1ece3cp-1f;
#endif

constexpr auto lutSize = 1024;
constexpr auto integralSize = 16*1024;

template<typename C>
auto generateLUT(C&& func) {
	float4 c{ 0.f,0.f,0.f,h };
	using res_t = double;// decltype(func(c, std::declval<float4>()));
	std::array<res_t, lutSize> LUT;
	float dd = 2.f * H / ((float)lutSize - 1);
	for (auto di = 0; di < lutSize; ++di) {
		auto n = integralSize;
		double dh = H / ((double)n);
		double d = H - dd * (double)(di);

		res_t integral = vector_t<double, math::dimension_v<res_t>>::zero();
#pragma omp parallel for reduction(+ : integral)
		for (auto ni = 0; ni < n; ++ni) {
			double xl = dh * (double)ni;
			double xh = dh * (double)ni + dh;

			float4 p{ (float) xl + 0.5f * (float)dh, 0.f, 0.f, h };

			double hl = math::clamp(xl - d, 0.f, 2.f * xl);
			double hh = math::clamp(xh - d, 0.f, 2.f * xh);


			double Vl = CUDART_PI_F * hl * hl / 3.f * (3.f * xl - hl);
			double Vh = CUDART_PI_F * hh * hh / 3.f * (3.f * xh - hh);

			double dV = Vh - Vl;

			integral += dV * math::castTo<typename vector_t<double, math::dimension_v<res_t>>::type>(func(c, p, (p.x - d) / H));
		}
		LUT[di] = integral;
	}
	std::reverse(LUT.begin(), LUT.end());
	return LUT;
}

constexpr float t = 0.0001f;
template<typename C>
auto gradientLUT(C&& func) {
	float4 c{ 0.f,0.f,0.f,h };
	using res_t = double;// decltype(func(c, std::declval<float4>()));
	std::array<res_t, lutSize> LUT;
	double dd = 2.f * H / ((double)lutSize - 1);
	for (auto di = 0; di < lutSize; ++di) {
		constexpr auto n = integralSize;
		double dh = H / ((double)n);
		double d = H - dd * (double)(di);

		res_t integralp = vector_t<double, math::dimension_v<res_t>>::zero();
#pragma omp parallel for reduction(+ : integralp)
		for (auto ni = 0; ni < n; ++ni) {
			double xl = dh * (double)ni;
			double xh = dh * (double)ni + dh;

			float4 p{ (float) xl + 0.5f * (float) dh, 0.f, 0.f, h };

			double hl = math::clamp(xl - d + t, 0., 2. * xl);
			double hh = math::clamp(xh - d + t, 0., 2. * xh);


			double Vl = CUDART_PI * hl * hl / 3. * (3. * xl - hl);
			double Vh = CUDART_PI * hh * hh / 3. * (3. * xh - hh);

			double dV = Vh - Vl;

			integralp += dV * math::castTo<typename vector_t<double, math::dimension_v<res_t>>::type>(func(c, p, (p.x - d - t) / H));
		}
		res_t integraln = vector_t<double, math::dimension_v<res_t>>::zero();
#pragma omp parallel for reduction(+ : integraln)
		for (auto ni = 0; ni < n; ++ni) {
			double xl = dh * (double)ni;
			double xh = dh * (double)ni + dh;

			float4 p{ (float)xl + 0.5f * (float)dh , 0.f, 0.f, h };

			double hl = math::clamp(xl - d - t, 0., 2. * xl);
			double hh = math::clamp(xh - d - t, 0., 2. * xh);


			double Vl = CUDART_PI_F * hl * hl / 3. * (3. * xl - hl);
			double Vh = CUDART_PI_F * hh * hh / 3. * (3. * xh - hh);

			double dV = Vh - Vl;

			integraln += dV * math::castTo<typename vector_t<double, math::dimension_v<res_t>>::type>(func(c, p, (p.x - d + t) / H));
		}
		LUT[di] = (integralp - integraln) / (2.0 * t);
	}
	std::reverse(LUT.begin(), LUT.end());
	return LUT;
}
template<typename T>
auto lookup (const std::array<T, lutSize> LUT, float x) {
	auto xRel = ((x + H) / (2.f * H)) * ((float)lutSize - 1.f);
	auto xL = math::floorf(xRel);
	auto xH = math::ceilf(xRel);
	auto xD = xRel - xL;
	int32_t xLi = math::clamp(static_cast<int32_t>(xL), 0, lutSize - 1);
	int32_t xHi = math::clamp(static_cast<int32_t>(xH), 0, lutSize - 1);
	auto lL = LUT[xLi];
	auto lH = LUT[xHi];
	return lL * xD + (1.f - xD) * lH;
};

#include <config/config.h>
#include <fstream>
#ifdef _WIN32
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif
 

auto writeLUT(const std::string& name, const std::string& type, const std::array<double, lutSize>& LUT) {
	fs::path bin_dir(sourceDirectory);
	auto file = bin_dir / "cfg" / name;
	file.replace_extension("lut");

	if (fs::exists(file)) {
		if (fs::exists(__FILE__)) {
			auto input_ts = fs::last_write_time(__FILE__);
			auto output_ts = fs::last_write_time(file);
			if (input_ts <= output_ts) {
				return;
			}
		}
	}
	std::cout << "Writing " << file.string() << std::endl;

	std::ofstream output(file.string());
	//output << "std::vector<" << type << "> " << name << "{ ";
	int32_t ctr = 0;
	for (auto v : LUT) {
		output << std::scientific << std::setprecision(std::numeric_limits<float>::digits10 + 1)
			<< static_cast<float>(v) << " ";
	}
	//output << "};" << std::endl;
	output.close();
}

auto sphericalGradientIntegral(const std::function<double(float4, float4, float)>& func, int32_t phiSlices, int32_t thetaSlices, int32_t radiusSteps) {
	float4 c{ 0.f,0.f,0.f,h };
	using res_t = double;
	std::array<res_t, lutSize> LUT;
	double dd = 2.f * H / ((double)lutSize - 1);
	for (auto di = 0; di < lutSize; ++di) {
		auto n = radiusSteps;
		double dh = H / ((double)n);
		double d = H - dd * (double)(di);
		double dTheta = (2.0 * CUDART_PI) / (double)thetaSlices;
		double dPhi = (CUDART_PI) / (double)phiSlices;

		res_t integralp = vector_t<double, math::dimension_v<res_t>>::zero();
#pragma omp parallel for
		for (auto iR = 0; iR < radiusSteps; ++iR) {
			double xl = dh * (double)iR;
			double xh = dh * (double)iR + dh;
			float r = (float) xl + 0.5f * (float)dh;
			double Vl = 4.0 / 3.0 * CUDART_PI * xl * xl * xl;
			double Vh = 4.0 / 3.0 * CUDART_PI * xh * xh * xh;

			double dV = (Vh - Vl) / (double)thetaSlices / (double)phiSlices;
			res_t thetaSum = vector_t<double, math::dimension_v<res_t>>::zero();
			for (auto iTheta = 0.0; iTheta < 2.0 * CUDART_PI; iTheta += dTheta) {
				res_t phiSum = vector_t<double, math::dimension_v<res_t>>::zero();
				for (auto iPhi = 0.0; iPhi < CUDART_PI; iPhi += dPhi) {
					double theta = iTheta + dTheta * 0.5;
					double phi = iPhi + dPhi * 0.5;
					double x = r * cos(theta) * sin(phi);
					double y = r * sin(theta) * sin(phi);
					double z = r * cos(phi);
					if (x < d + t)
						continue;
					float4 p{ (float)x, (float)y, (float)z, h };
					phiSum += dV * math::castTo<typename vector_t<double, math::dimension_v<res_t>>::type>(func(c, p, d + t));
				}
				thetaSum += phiSum;
			}
			integralp += thetaSum;
		}
		res_t integraln = vector_t<double, math::dimension_v<res_t>>::zero();
#pragma omp parallel for
		for (auto iR = 0; iR < radiusSteps; ++iR) {
			double xl = dh * (double)iR;
			double xh = dh * (double)iR + dh;
			float r = (float)xl + 0.5f * (float)dh;
			double Vl = 4.0 / 3.0 * CUDART_PI * xl * xl * xl;
			double Vh = 4.0 / 3.0 * CUDART_PI * xh * xh * xh;

			double dV = (Vh - Vl) / (double)thetaSlices / (double)phiSlices;
			res_t thetaSum = vector_t<double, math::dimension_v<res_t>>::zero();
			for (auto iTheta = 0.0; iTheta < 2.0 * CUDART_PI; iTheta += dTheta) {
				res_t phiSum = vector_t<double, math::dimension_v<res_t>>::zero();
				for (auto iPhi = 0.0; iPhi < CUDART_PI; iPhi += dPhi) {
					double theta = iTheta + dTheta * 0.5;
					double phi = iPhi + dPhi * 0.5;
					double x = r * cos(theta) * sin(phi);
					double y = r * sin(theta) * sin(phi);
					double z = r * cos(phi);
					if (x < d - t)
						continue;
					float4 p{ (float)x, (float)y, (float)z, h };
					phiSum += dV * math::castTo<typename vector_t<double, math::dimension_v<res_t>>::type>(func(c, p, d - t));
				}
				thetaSum += phiSum;
			}
			integraln += thetaSum;
		}
		LUT[di] = (integralp - integraln) / (2.0 * t);
	}
	std::reverse(LUT.begin(), LUT.end());
	return LUT;
}

void progressBar(int32_t frame, int32_t frameTarget, float progress) {
	std::ios cout_state(nullptr);
	cout_state.copyfmt(std::cout);
	static auto startOverall = std::chrono::high_resolution_clock::now();
	static auto startFrame = startOverall;
	static auto lastTime = startOverall;
	static int32_t lastFrame = frame;
	//if (frame != lastFrame) {
	//	lastFrame = frame;
	if(frame == 0)
		startFrame = std::chrono::high_resolution_clock::now();
	//}
	auto now = std::chrono::high_resolution_clock::now();
	lastTime = now;
	int barWidth = 128;
	std::cout << "Generating " << std::setw(4) << frame;
	if (frameTarget != -1)
		std::cout << "/" << std::setw(4) << frameTarget;
	std::cout << " [";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << std::setw(3) << int(progress * 100.0) << " ";
	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - startFrame);
	if (dur.count() < 100 || progress < 1e-3f) {
		std::cout << " ---/---s  ";
	}
	else {
		auto totalTime = ((float)std::chrono::duration_cast<std::chrono::microseconds>(now - startFrame).count()) / 1000.f / 1000.f;
		std::cout << std::fixed << std::setprecision(0) << " " << std::setw(3) << totalTime << "/" << std::setw(3) << (totalTime / progress) << "s  ";
	}
	std::cout << "\r";
	std::cout.flush();
	std::cout.copyfmt(cout_state);
}

auto sphericalIntegral(const std::function<double(float4, float4, float)>& func, int32_t phiSlices, int32_t thetaSlices, int32_t radiusSteps) {
	float4 c{ 0.f,0.f,0.f,h };
	using res_t = double;
	std::array<res_t, lutSize> LUT;
	double dd = 2.f * H / ((double)lutSize - 1);
	//std::vector<double> thetaSum(thetaSlices);
	//std::vector<double> phiSum(phiSlices);
	for (auto di = 0; di < lutSize; ++di) {
		progressBar(di, lutSize, (double)di / (double)lutSize);
		auto n = radiusSteps;
		double dh = H / ((double)n);
		double d = -H + dd * (double)(di);
		double dTheta = (2.0 * CUDART_PI) / (double)thetaSlices;
		double dPhi = (CUDART_PI) / (double)phiSlices;

		res_t integralp = vector_t<double, math::dimension_v<res_t>>::zero();
		double dVSum = 0.0;
#pragma omp parallel for
		for (auto iR = 0; iR < radiusSteps; ++iR) {
			double xl = dh * (double)iR;
			double xh = dh * (double)iR + dh;
			float r = (float)xl + 0.5f * (float)dh;
			double Vl = 4.0 / 3.0 * CUDART_PI * xl * xl * xl;
			double Vh = 4.0 / 3.0 * CUDART_PI * xh * xh * xh;

			double dV = (Vh - Vl) / (double)thetaSlices / (double)phiSlices;
			res_t thetaSum = vector_t<double, math::dimension_v<res_t>>::zero();
			for (auto iTheta = 0; iTheta < thetaSlices; iTheta++) {
				res_t phiSum = vector_t<double, math::dimension_v<res_t>>::zero();
				for (auto iPhi = 0; iPhi < phiSlices; iPhi++) {
					double theta = ((double)iTheta) * dTheta;// +dTheta * 0.5;
					double phi = ((double)iPhi) * dPhi;// +dPhi * 0.5;
					double x = r * cos(theta) * sin(phi);
					double y = r * sin(theta) * sin(phi);
					double z = r * cos(phi);
					if (x < d)
						phiSum += 0.0;
					else {
						float4 p{ (float)x, (float)y, (float)z, h };
						phiSum += math::castTo<typename vector_t<double, math::dimension_v<res_t>>::type>(func(c, p, (x - d) / (1.0 * H)));
					}
				}
				//std::sort(phiSum.begin(), phiSum.end());
				thetaSum += phiSum;
			}
			//std::sort(thetaSum.begin(), thetaSum.end());
			//integralp += dV * std::reduce(thetaSum.begin(), thetaSum.end());
			integralp += dV * thetaSum;
		}
		LUT[di] = integralp;
		//break;
	}
	std::cout << std::endl;
	//std::reverse(LUT.begin(), LUT.end());
	return LUT;
}

auto approximateGradient(const std::array<double, lutSize>& LUT, const std::array<double, lutSize>& vLUT) {
	std::array<double, lutSize> gLUT;
	for (auto& e : gLUT) e = 0.0;
	for (int32_t i = 0; i < lutSize - 1; ++i) {
		auto vi1 = 1.f; //vLUT[i + 1];
		auto vi = 1.f; //vLUT[i];
		vi1 = vi1 < 1e-5f ? vi : vi1;

		gLUT[i] = (LUT[i + 1] / vi1 - LUT[i] / vi) / (2.0 * H / (double)lutSize) * vi;
	}
	gLUT[lutSize - 1] = gLUT[lutSize - 2];
	return gLUT;
}

auto smoothLUT(const std::array<double, lutSize>& LUT) {
	std::array<double, lutSize> gLUT;
	gLUT = LUT;
	for (int32_t i = 1; i < lutSize - 1; ++i) {
		if(LUT[i-1] < LUT[i])
			gLUT[i] = (LUT[i - 1] + LUT[i + 1])*0.5;
	}
	return gLUT;
}
 
#include <omp.h>
int main(int32_t argc, char** argv) {
	omp_set_num_threads(12);
#ifdef CALC_CONSTANTS
	std::ios cout_state(nullptr);
	cout_state.copyfmt(std::cout);
	std::cout << std::hexfloat << "H = " << H << std::endl;
	std::cout << std::hexfloat << "h = " << h << std::endl;
	std::cout << std::hexfloat << "radius = " << radius << std::endl;
	std::cout << std::hexfloat << "volume = " << volume << std::endl;
	std::cout << std::hexfloat << "spacing = " << spacing << std::endl;
	std::cout.copyfmt(cout_state);
	std::cout << "H = " << H << std::endl;
	std::cout << "h = " << h << std::endl;
	std::cout << "radius = " << radius << std::endl;
	std::cout << "volume = " << volume << std::endl;
	std::cout << "spacing = " << spacing << std::endl;
	std::cout.copyfmt(cout_state);
#endif
	std::cout << "Running LUT generation code." << std::endl;
//#ifdef WIN32
//	HWND console = GetConsoleWindow();
//	RECT _r;
//	GetWindowRect(console, &_r);
//	MoveWindow(console, 0, 0, 1920, 1200, TRUE);
//#endif
	//std::cout << "H = " << H << std::endl;
	//std::cout << "h = " << h << std::endl;
	//std::cout << "radius = " << r << std::endl;
	//std::cout << "volume = " << V << std::endl;
	//std::cout << "spacing = " << s << std::endl;

	//std::cout << "Spherical Integral: " <<  << std::endl;
	//std::cout << "Analytical: " << CUDART_PI * 4.0 / 3.0 * H * H * H << std::endl;

	//std::array densityLUT = sphericalIntegral([](float4 c, float4 p, float d) {return kernel(c, p); }, 128, 128, 128);
	//std::cout << "Generated density LUT" << std::endl;
	float dFactor = 0.f;// 0.f;// 1.5f; 
	std::array densityLUT = generateLUT([&](float4 c, float4 p, float d) { return (1.f + dFactor * d) * kernel(c, p); });
	writeLUT("density", "float", densityLUT);

	auto lookup = [&](auto x) {
		float xRel = ((x + 1.f) / 2.f)* ((float)lutSize - 1.f);
		auto xL = math::floorf(xRel);
		auto xH = math::ceilf(xRel);
		auto xD = xRel - xL;
		int32_t xLi = math::clamp(static_cast<int32_t>(xL), 0, lutSize - 1);
		int32_t xHi = math::clamp(static_cast<int32_t>(xH), 0, lutSize - 1);
		auto lL = densityLUT[xLi];
		auto lH = densityLUT[xHi];
		auto val = lL * xD + (1.f - xD) * lH;
		return val;
	};

	auto findX = [&](auto x) {
		float f = -1.f;
		int32_t n = 8;
		for (int32_t n = 1; n < 11; ++n) {
			auto fx = lookup(f);
			//std::cout << "Starting at " << f << " : " << fx << " with dx = " << powf(0.5f, (float)n) << std::endl;
			while (n % 2 == 1 ? fx > x + 0.001f : fx < x - 0.001f){
				f += (n % 2 == 1 ? 1.f : -1.f) * powf(0.5f, (float)n);
				fx = lookup(f);
				//std::cout << f << " -> " << fx << std::endl;
			}
		}
		//std::cout << f << " - " << lookup(f) << " <-> " << x << std::endl;
		return f;
	};
	std::array<double, lutSize> offsetLUT; 
	for (int32_t i = 0; i < lutSize; ++i) {
		float f = (float) i / (float)lutSize;
		offsetLUT[i] = /*0.24509788f*/ -0.f* findX(f);
	}
	//std::reverse(offsetLUT.begin(), offsetLUT.end());
	writeLUT("offsetLUT", "float", offsetLUT);
	//findX(0.5f);

	std::array spline2LUT = generateLUT([&](float4 c, float4 p, float d) { return (1.f + dFactor * d) * (1.f + dFactor * d) *  math::dot3(gradient(c, p), gradient(c, p)); });
	writeLUT("spline2", "float", spline2LUT);
	std::array spikyLUT = generateLUT([&](float4 c, float4 p, float d) { return (1.f + dFactor * d) * PressureKernel<kernel_kind::spline4>::value(c,p); });
	writeLUT("spiky", "float", spikyLUT);
	std::array splineLUT = generateLUT([&](float4 c, float4 p, float d) { return (1.f + dFactor * d) *  kernel(c, p); });
	writeLUT("spline", "float", splineLUT);
	std::array cohesionLUT = generateLUT([&](float4 c, float4 p, float d) { return (1.f + dFactor * d) *  Kernel<kernel_kind::cohesion>::value(c, p).x; });
	for (int32_t i = 1; i < cohesionLUT.size(); ++i) {
		cohesionLUT[i] = abs(cohesionLUT[i]) < 1e-12 ? cohesionLUT[i - 1] : cohesionLUT[i];
	}
	writeLUT("cohesion", "float", cohesionLUT);
	std::array adhesionLUT = generateLUT([&](float4 c, float4 p, float d) { return (1.f + dFactor * d) *  Kernel<kernel_kind::adhesion>::value(c, p).x; });
	for (int32_t i = 1; i < adhesionLUT.size(); ++i) {
		adhesionLUT[i] = abs(adhesionLUT[i]) < 1e-12 ? adhesionLUT[i - 1] : adhesionLUT[i];
	}
	writeLUT("adhesion", "float", adhesionLUT);

	std::array volumeLUT = generateLUT([&](float4 c, float4 p, float d) { return (1.f + dFactor * d); });
	//for (auto& v : volumeLUT)
	//	v = 1.f;
	writeLUT("volume", "float", volumeLUT);

	////std::cout << adhesion0  << std::endl;
	//std::array spikyGradientLUT = generateLUT([&](float4 c, float4 p, float d) { 
	//	return -((1.f + dFactor * d) *  SpikyKernel<kernel_kind::spline4>::gradient(c, p).x); });

	std::array spikyGradientLUT = approximateGradient(spikyLUT, volumeLUT);
	for (int32_t i = 1; i < spikyGradientLUT.size(); ++i) {
		//spikyGradientLUT[i] = fabsf(spikyGradientLUT[i]) < 1e-12f ? spikyGradientLUT[i - 1] : spikyGradientLUT[i];
	}
	//std::array spikyGradientLUT = generateLUT([&](float4 c, float4 p, float d) { return (1.f) * SpikyKernel<kernel_kind::spline4>::gradient(c, p).x ; });
	//for (auto& v : spikyGradientLUT)
		//v = -v;// math::max(std::decay_t<decltype(v)>{0.f}, v);
	writeLUT("spikyGradient", "float", spikyGradientLUT);
	std::array splineGradientLUT = approximateGradient(splineLUT, volumeLUT);
	for (int32_t i = 1; i < splineGradientLUT.size(); ++i) {
		//splineGradientLUT[i] = fabsf(splineGradientLUT[i]) < 1e-12f ? splineGradientLUT[i - 1] : splineGradientLUT[i];
	}
	//std::array splineGradientLUT = generateLUT([&](float4 c, float4 p, float d) { return (1.f ) * gradient(c, p).x;  });
	//for (auto& v : splineGradientLUT)
	//	v = -v;// math::max(std::decay_t<decltype(v)>{0.f}, v);
	writeLUT("splineGradient", "float", splineGradientLUT);
	
	//std::array densityLUT = sphericalIntegral([&](float4 c, float4 p, float d) {return kernel(c, p); }, 127, 127, 127);
	auto chi = [&](auto d) {
		return 1.f + dFactor * d;
		//return math::clamp(1.0 / lookup(densityLUT, -d * HforV1) + d,1.0,4.0);
	};
	//densityLUT = smoothLUT(densityLUT);
	//std::cout << "Generated density LUT" << std::endl;
	//writeLUT("density", "float", densityLUT);

	//std::array splineLUTN = sphericalIntegral([&](float4 c, float4 p, float d) {return chi(d) * kernel(c, p); }, 127, 127, 127);
	//splineLUTN = smoothLUT(splineLUTN);
	//std::cout << "Generated spline LUT" << std::endl;
	//writeLUT("splinePolar", "float", splineLUTN);
	//std::array splineGradientLUTN = approximateGradient(splineLUTN);
	//std::cout << "Generated spline Gradient LUT" << std::endl;
	//writeLUT("splineGradientPolar", "float", splineGradientLUTN);


	//std::array spikyLUT = sphericalIntegral([&](float4 c, float4 p, float d) {return chi(d) * SpikyKernel<kernel_kind::spline4>::value(c, p); }, 128, 128, 128);
	//spikyLUT = smoothLUT(spikyLUT);
	//std::cout << "Generated spiky LUT" << std::endl;
	//writeLUT("spiky", "float", spikyLUT);
	//std::array spikyGradientLUT = approximateGradient(spikyLUT);
	//std::cout << "Generated spiky Gradient LUT" << std::endl;
	//writeLUT("spikyGradient", "float", spikyGradientLUT);
	//std::array cohesionLUT = sphericalIntegral([&](float4 c, float4 p, float d) {return chi(d) * Kernel<kernel_kind::cohesion>::value(c, p).x; }, 128, 128, 128);
	//std::cout << "Generated cohesion LUT" << std::endl;
	//writeLUT("cohesion", "float", cohesionLUT);
	//std::array adhesionLUT = sphericalIntegral([&](float4 c, float4 p, float d) {return chi(d) * Kernel<kernel_kind::adhesion>::value(c, p).x; }, 128, 128, 128);
	//auto adhesion0 = lookup(adhesionLUT, -HforV1);
	//for (auto& v : adhesionLUT)
	//	v /= adhesion0;
	//std::cout << "Generated adhesion LUT" << std::endl;
	//writeLUT("adhesion", "float", adhesionLUT);
	//std::array volumeLUT = sphericalIntegral([&](float4 c, float4 p, float d) {return 1.0; }, 128, 128, 128);
	//std::cout << "Generated volume LUT" << std::endl;
	//writeLUT("volume", "float", volumeLUT);


	//float dh = 0.001f;
	//float4 dx{ dh,0.f,0.f,0.f };
	//float4 dy{ 0.f,dh,0.f,0.f };
	//float4 dz{ 0.f,0.f,dh,0.f };

	//std::array spikyLUT = generateLUT([&](float4 c, float4 p) { return SpikyKernel<kernel_kind::spline4>::value(c, p); });
	//std::array splineLUT = generateLUT([&](float4 c, float4 p) { return kernel(c, p); });
	//std::array cohesionLUT = generateLUT([&](float4 c, float4 p) { return Kernel<kernel_kind::cohesion>::value(c, p).x; });
	//std::array adhesionLUT = generateLUT([&](float4 c, float4 p) { return Kernel<kernel_kind::adhesion>::value(c, p).x; });
	//std::array spikyGradientLUT = gradientLUT([&](float4 c, float4 p) { return SpikyKernel<kernel_kind::spline4>::value(c, p); });
	//std::array splineGradientLUT = gradientLUT([&](float4 c, float4 p) { return kernel(c, p); });
	//std::array volumeLUT = gradientLUT([&](float4 c, float4 p) { return 1.f; });

	//for (int32_t i = 0; i < lutSize; ++i) {
	//	std::cout << i << "\t" << splineLUT[i] << " - " << splineLUT2[i] << " -> " << splineLUT[i] / splineLUT2[i] << std::endl;
	//}

	//for (float x = -H; x <= H; x += H / 16.f) {
	//	std::cout << x << "\t" << lookup(splineLUT, x) << " @ " << lookup(splineLUT2, x) << std::endl;
	//}

	//for (float x = -H; x <= H; x += H / 16.f) {
	//	std::cout << x << "\t" << lookup(splineLUT, x) << " @ " << lookup(splineGradientLUT, x) << " -> " << lookup(splineLUT, x + dh) << " : " << lookup(splineLUT, x) + dh * lookup(splineGradientLUT, x) << std::endl;
	//}

	//for (auto& v : splineGradientLUT)
	//	v = -v;// math::max(std::decay_t<decltype(v)>{0.f}, v);
	//for (auto& v : spikyGradientLUT)
	//	v = -v;// math::max(std::decay_t<decltype(v)>{0.f}, v);
	//for (auto& v : cohesionLUT)
	//	v = -v;// math::max(std::decay_t<decltype(v)>{0.f}, v);
	////std::cout << adhesion0  << std::endl;


	//auto x0 = lookup(splineLUT, 0.0f);
	//auto xp = lookup(splineLUT, 0.0f + t);
	//auto xn = lookup(splineLUT, 0.0f - t);
	//auto xnum = (xp - xn) / (2.f * t);
	//auto xint = lookup(splineGradientLUT, 0.f);
	//std::cout << x0 << " -> " << xp << std::endl;
	//std::cout << xnum << " <-> " << xint << std::endl;
	
	//float evalP = 0.f * H;
	//std::cout << "Spherical integral" << std::endl;
	//std::cout << "Kernel:   " << lookup(splineLUT, evalP) << std::endl;
	//std::cout << "Gradient: " << lookup(splineGradientLUT, evalP) << std::endl;

	//float4 c{ 0.f,0.f,0.f,h };
	//double sumd = 0.0;
	//double4 gradSumd{ 0.0,0.0,0.0,0.0 };
	//double volume = 0.f;
	//constexpr auto trapz = 512;
	//constexpr auto dt = 2.0 * (double) H / ((double)trapz);
	//constexpr auto dV = dt * dt * dt;
	//auto trapH = support_from_volume(dV);
	//for (int32_t xi = -trapz / 2; xi <= trapz / 2; ++xi) {
	//	for (int32_t yi = -trapz / 2; yi <= trapz / 2; ++yi) {
	//		for (int32_t zi = -trapz / 2; zi <= trapz / 2; ++zi) {
	//			float4 p{ (float)dt * (float)xi, (float)dt * (float)yi, (float)dt * (float)zi, h };
	//			if (p.x > evalP)
	//				continue;
	//			sumd += (double) kernel(c, p) * dV;
	//			gradSumd += math::castTo<double4>(gradient(c, p)) * dV;
	//			volume += kernel(c, p) > 0.f ? dV : 0.f;
	//		}
	//	}
	//}
	//std::cout << "Trapz" << std::endl;
	//std::cout << "Kernel:   " << sumd << std::endl;
	//std::cout << "Gradient: " << gradSumd << std::endl;

	//getchar();
}