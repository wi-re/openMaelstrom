#pragma once
#include <utility/math.h>
__device__ __host__ inline float4 eul2quat(float3 euler) {
	float cy = cosf(euler.x * 0.5f);
	float sy = sinf(euler.x * 0.5f);
	float cp = cosf(euler.y * 0.5f);
	float sp = sinf(euler.y * 0.5f);
	float cr = cosf(euler.z * 0.5f);
	float sr = sinf(euler.z * 0.5f);

	float4 q{ 0.f,0.f,0.f,0.f };
	q.w = cy * cp * cr + sy * sp * sr;
	q.x = cy * cp * sr - sy * sp * cr;
	q.y = sy * cp * sr + cy * sp * cr;
	q.z = sy * cp * cr - cy * sp * sr;
	return q;
}
__device__ __host__ inline float3 quat2eul(const float4& q) {
	// roll (x-axis rotation)
	float sinr_cosp = +2.f * (q.w * q.x + q.y * q.z);
	float cosr_cosp = +1.f - 2.f * (q.x * q.x + q.y * q.y);
	float roll = atan2f(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	float sinp = +2.f * (q.w * q.y - q.z * q.x);
	float pitch = 0.f;
	if (fabsf(sinp) >= 1)
		pitch = copysign(CUDART_PI_F / 2.0f, sinp); // use 90 degrees if out of range
	else
		pitch = asinf(sinp);

	// yaw (z-axis rotation)
	float siny_cosp = +2.f * (q.w * q.z + q.x * q.y);
	float cosy_cosp = +1.f - 2.f * (q.y * q.y + q.z * q.z);
	float yaw = atan2f(siny_cosp, cosy_cosp);
	return float3{ roll, pitch, yaw };
}

struct Matrix4x4 {
	float data[4 * 4] = {
		0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
	constexpr __device__ __host__ inline float &get(int row, int column) {
		return data[row * 4 + column];
	}
	constexpr __device__ __host__ inline const float &get(int row, int column) const{
		return data[row * 4 + column];
	}
	constexpr __device__ __host__ inline float &operator()(int row, int column) {
		return data[row * 4 + column];
	}
	constexpr __device__ __host__ inline const float &operator()(int row, int column) const{
		return data[row * 4 + column];
	}

	constexpr inline Matrix4x4() = default;
	constexpr __device__ __host__ Matrix4x4(
		const float m_00      , const float m_01 = 0.f, const float m_02 = 0.f, const float m_03 = 0.f,
		const float m_10 = 0.f, const float m_11 = 1.f, const float m_12 = 0.f, const float m_13 = 0.f,
		const float m_20 = 0.f, const float m_21 = 0.f, const float m_22 = 1.f, const float m_23 = 0.f,
		const float m_30 = 0.f, const float m_31 = 0.f, const float m_32 = 0.f, const float m_33 = 1.f) {
		data[0 + 0 * 4] = m_00; data[1 + 0 * 4] = m_01; data[2 + 0 * 4] = m_02; data[3 + 0 * 4] = m_03;
		data[0 + 1 * 4] = m_10; data[1 + 1 * 4] = m_11; data[2 + 1 * 4] = m_12; data[3 + 1 * 4] = m_13;
		data[0 + 2 * 4] = m_20; data[1 + 2 * 4] = m_21; data[2 + 2 * 4] = m_22; data[3 + 2 * 4] = m_23;
		data[0 + 3 * 4] = m_30; data[1 + 3 * 4] = m_31; data[2 + 3 * 4] = m_32; data[3 + 3 * 4] = m_33;
	}
	constexpr __device__ __host__ inline Matrix4x4(const Matrix4x4 &rhs) {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				data[i * 4 + j] = rhs.data[i * 4 + j];
	}
	constexpr __device__ __host__ inline static Matrix4x4 identity() {
		Matrix4x4 m;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				if (i == j)
					m.get(i, j) = 1.f;
		return m;
	}
	static __device__ __host__ inline Matrix4x4 fromQuaternion(const float4 quat) {
		float qx = quat.x; float qx2 = qx * qx;
		float qy = quat.y; float qy2 = qy * qy;
		float qz = quat.z; float qz2 = qz * qz; 
		float qw = quat.w;
		return Matrix4x4{ 
			1.f - 2.f * qy2 - 2.f * qz2,		2.f * qx * qy - 2.f * qz * qw,			2.f * qx * qz + 2.f * qy * qw,		0.f,
			2.f * qx * qy + 2.f * qz * qw,		1.f - 2.f * qx2 - 2.f * qz2,			2.f * qy * qz - 2.f * qx * qw,		0.f,
			2.f * qx * qz - 2.f * qy * qw,		2.f * qy * qz + 2.f * qx * qw,			1.f - 2.f * qx2 - 2.f * qy2,		0.f,
			0.f,								0.f,									0.f,								1.f 
		};
		//1 - 2 * qy2 - 2 * qz2	2 * qx*qy - 2 * qz*qw	2 * qx*qz + 2 * qy*qw
		//	2 * qx*qy + 2 * qz*qw	1 - 2 * qx2 - 2 * qz2	2 * qy*qz - 2 * qx*qw
		//	2 * qx*qz - 2 * qy*qw	2 * qy*qz + 2 * qx*qw	1 - 2 * qx2 - 2 * qy2
	}
	static __device__ __host__ inline Matrix4x4 fromQuaternion(const float x, const float y, const float z, const float w) {
		return fromQuaternion(float4{ x,y,z,w });
	}
	static __device__ __host__ inline Matrix4x4 fromEuler(const float3 euler) {
		return fromQuaternion(eul2quat(euler));
	}
	static __device__ __host__ inline Matrix4x4 fromEuler(const float roll, const float pitch, const float yaw) {
		return fromEuler(float3{ roll,pitch,yaw });
	}
	static constexpr __device__ __host__ inline Matrix4x4 fromTranspose(const float3 p) {
		return Matrix4x4{
			1.f,0.f,0.f,p.x,
			0.f,1.f,0.f,p.y,
			0.f,0.f,1.f,p.z,
			0.f,0.f,0.f,1.f
		};
	}
	static constexpr __device__ __host__ inline Matrix4x4 fromTranspose(const float4 p) {
		return fromTranspose(float3{ p.x,p.y,p.z });
	}
	static constexpr __device__ __host__ inline Matrix4x4 fromTranspose(const float x, const float y, const float z) {
		return fromTranspose(float3{ x,y,z });
	}
	__device__ __host__ inline void print() const{
		printf("%6.4f %6.4f %6.4f %6.4f\n%6.4f %6.4f %6.4f %6.4f\n%6.4f %6.4f %6.4f %6.4f\n%6.4f %6.4f %6.4f %6.4f\n",
			data[0], data[1], data[2], data[3],
			data[4], data[5], data[6], data[7],
			data[8], data[9], data[10], data[11],
			data[12], data[13], data[14], data[15]);
	}
	
	constexpr __device__ __host__ inline float det() const {
		return
			get(0, 3) * get(1, 2) * get(2, 1) * get(3, 0) - get(0, 2) * get(1, 3) * get(2, 1) * get(3, 0) -
			get(0, 3) * get(1, 1) * get(2, 2) * get(3, 0) + get(0, 1) * get(1, 3) * get(2, 2) * get(3, 0) +
			get(0, 2) * get(1, 1) * get(2, 3) * get(3, 0) - get(0, 1) * get(1, 2) * get(2, 3) * get(3, 0) -
			get(0, 3) * get(1, 2) * get(2, 0) * get(3, 1) + get(0, 2) * get(1, 3) * get(2, 0) * get(3, 1) +
			get(0, 3) * get(1, 0) * get(2, 2) * get(3, 1) - get(0, 0) * get(1, 3) * get(2, 2) * get(3, 1) -
			get(0, 2) * get(1, 0) * get(2, 3) * get(3, 1) + get(0, 0) * get(1, 2) * get(2, 3) * get(3, 1) +
			get(0, 3) * get(1, 1) * get(2, 0) * get(3, 2) - get(0, 1) * get(1, 3) * get(2, 0) * get(3, 2) -
			get(0, 3) * get(1, 0) * get(2, 1) * get(3, 2) + get(0, 0) * get(1, 3) * get(2, 1) * get(3, 2) +
			get(0, 1) * get(1, 0) * get(2, 3) * get(3, 2) - get(0, 0) * get(1, 1) * get(2, 3) * get(3, 2) -
			get(0, 2) * get(1, 1) * get(2, 0) * get(3, 3) + get(0, 1) * get(1, 2) * get(2, 0) * get(3, 3) +
			get(0, 2) * get(1, 0) * get(2, 1) * get(3, 3) - get(0, 0) * get(1, 2) * get(2, 1) * get(3, 3) -
			get(0, 1) * get(1, 0) * get(2, 2) * get(3, 3) + get(0, 0) * get(1, 1) * get(2, 2) * get(3, 3);
	}	
	constexpr __device__ __host__ inline float minor(int32_t r0, int32_t r1, int32_t r2, int32_t c0, int32_t c1, int32_t c2) const
	{
		return data[4 * r0 + c0] * (data[4 * r1 + c1] * data[4 * r2 + c2] - data[4 * r2 + c1] * data[4 * r1 + c2]) -
			data[4 * r0 + c1] * (data[4 * r1 + c0] * data[4 * r2 + c2] - data[4 * r2 + c0] * data[4 * r1 + c2]) +
			data[4 * r0 + c2] * (data[4 * r1 + c0] * data[4 * r2 + c1] - data[4 * r2 + c0] * data[4 * r1 + c1]);
	}
	constexpr __device__ __host__ inline Matrix4x4 adjoint() const{
		return Matrix4x4{
			 minor(1, 2, 3, 1, 2, 3), -minor(0, 2, 3, 1, 2, 3),  minor(0, 1, 3, 1, 2, 3), -minor(0, 1, 2, 1, 2, 3),
			-minor(1, 2, 3, 0, 2, 3),  minor(0, 2, 3, 0, 2, 3), -minor(0, 1, 3, 0, 2, 3),  minor(0, 1, 2, 0, 2, 3),
			 minor(1, 2, 3, 0, 1, 3), -minor(0, 2, 3, 0, 1, 3),  minor(0, 1, 3, 0, 1, 3), -minor(0, 1, 2, 0, 1, 3),
			-minor(1, 2, 3, 0, 1, 2),  minor(0, 2, 3, 0, 1, 2), -minor(0, 1, 3, 0, 1, 2),  minor(0, 1, 2, 0, 1, 2)
		};
	}
	constexpr __device__ __host__ inline Matrix4x4 inverse() const {
		auto adj = adjoint();
		auto inv_det = 1.f / det();
		for (int32_t i = 0; i < 16; ++i)
			adj.data[i] = inv_det * adj.data[i];
		return adj;
	}
	constexpr __device__ __host__ inline Matrix4x4 transpose() const {
		return Matrix4x4{
			get(0, 0), get(1, 0), get(2, 0), get(3, 0),
			get(0, 1), get(1, 1), get(2, 1), get(3, 1),
			get(0, 2), get(1, 2), get(2, 2), get(3, 2),
			get(0, 3), get(1, 3), get(2, 3), get(3, 3)
		};
	}

	constexpr __device__ __host__ inline Matrix4x4 operator*(const float &rhs) const {
		Matrix4x4 C = *this;
		for (int32_t i = 0; i < 16; i++)
			C.data[i] *= rhs;
		return C;
	}
	constexpr __device__ __host__ inline Matrix4x4 operator/(const float &rhs) const {
        Matrix4x4 C = *this;
        for (int32_t i = 0; i < 16; i++)
        C.data[i] /= rhs;
        return C;
    }
	constexpr __device__ __host__ inline Matrix4x4 operator*(const Matrix4x4 &rhs) const {
		Matrix4x4 C;
		for (int32_t i = 0; i < 4; i++) {
			for (int32_t j = 0; j < 4; j++) {
				float num = 0.f;
				for (int32_t k = 0; k < 4; k++) {
					num += get(i, k) * rhs(k, j);
				}
				C(i, j) = num;
			}
		}
		return C;
	}
	constexpr __device__ __host__ inline Matrix4x4 operator+(const Matrix4x4 &rhs) const {
		Matrix4x4 C;
		for (int32_t i = 0; i < 16; ++i)
			C.data[i] = data[i] + rhs.data[i];
		return C;
	}
	constexpr __device__ __host__ inline Matrix4x4 operator-(const Matrix4x4 &rhs) const {
		Matrix4x4 C;
		for (int32_t i = 0; i < 16; ++i)
			C.data[i] = data[i] - rhs.data[i];
		return C;
	}
	constexpr __device__ __host__ inline Matrix4x4& operator*=(const float &rhs) {
		for (int32_t i = 0; i < 16; i++)
			data[i] *= rhs;
		return *this;
	}
	constexpr __device__ __host__ inline Matrix4x4& operator/=(const float &rhs) {
		for (int32_t i = 0; i < 16; i++)
			data[i] /= rhs;
		return  *this;
	}
	constexpr __device__ __host__ inline Matrix4x4& operator*=(const Matrix4x4 &rhs) {
		*this = *this * rhs;
		return *this;
	}
	constexpr __device__ __host__ inline Matrix4x4& operator+=(const Matrix4x4 &rhs) {
		for (int32_t i = 0; i < 16; ++i)
			data[i] = data[i] + rhs.data[i];
		return *this;
	}
	constexpr __device__ __host__ inline Matrix4x4& operator-=(const Matrix4x4 &rhs) {
		for (int32_t i = 0; i < 16; ++i)
			data[i] = data[i] - rhs.data[i];
		return *this;
	}
	constexpr __device__ __host__ inline float4 operator*(const float4& rhs) const{
		float4 res{ rhs.x * get(0, 0) + rhs.y * get(0, 1) + rhs.z * get(0, 2) +
					   rhs.w * get(0, 3),
				   rhs.x * get(1, 0) + rhs.y * get(1, 1) + rhs.z * get(1, 2) +
					   rhs.w * get(1, 3),
				   rhs.x * get(2, 0) + rhs.y * get(2, 1) + rhs.z * get(2, 2) +
					   rhs.w * get(2, 3),
				   rhs.x * get(3, 0) + rhs.y * get(3, 1) + rhs.z * get(3, 2) +
					   rhs.w * get(3, 3) };
		return res;
	}
};
