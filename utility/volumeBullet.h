#pragma once
#include <utility/math.h>
#include <btBulletDynamicsCommon.h>
#include <BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h>
#include <map>

namespace bt {
	float4 f4frombt3(btVector3 v);
	btVector3 bt3fromf3(float3 v);
	btVector3 bt3fromf4(float4 v);
	btQuaternion btqfromf4(float4 v);
	float4 f4frombtq(btQuaternion v);
	float4 quatConjugate(float4 v);
	float4 quatNormalize(float4 v);
	float4 quatMul(float4 q1, float4 q2);
	float4 toAngularVelocity(const float4& start, const float4& end, float delta_sec);

	struct World {
	private:
		World();
	public:
		static World& instance();
		~World();
		int32_t counter;
		btDefaultCollisionConfiguration* collisionConfiguration = nullptr;
		btCollisionDispatcher* dispatcher = nullptr;
		btBroadphaseInterface* overlappingPairCache = nullptr;
		btSequentialImpulseConstraintSolver* solver = nullptr;
		btDiscreteDynamicsWorld* dynamicsWorld = nullptr;

		//keep track of the shapes, we release memory at exit.
		//make sure to re-use collision shapes among rigid bodies whenever possible!
		btAlignedObjectArray<btCollisionShape*> collisionShapes;

		std::map<int32_t, int32_t> boundaryMap;
		void step(float dt);
		void resetWorld();
	};

}