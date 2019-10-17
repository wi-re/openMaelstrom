#include <utility/volumeBullet.h>


#include <SPH/boundary/volumeBoundary.cuh>
#include <utility/generation.h>
#include <utility/include_all.h>
// At some point this will have to be replaced with <filesystem>
#include <fstream>

namespace bt {
	float4 f4frombt3(btVector3 v) {
		return float4{ (float)v.x(), (float)v.y(), (float)v.z() , 0.f };
	}
	btVector3 bt3fromf3(float3 v) {
		return btVector3(v.x, v.y, v.z);
	}
	btVector3 bt3fromf4(float4 v) {
		return btVector3(v.x, v.y, v.z);
	}
	btQuaternion btqfromf4(float4 v) {
		return btQuaternion(v.x, v.y, v.z, v.w);
	}
	float4 f4frombtq(btQuaternion v) {
		return float4{ (float)v.x(), (float)v.y(), (float)v.z(), (float)v.w() };
	}
	float4 quatConjugate(float4 v) {
		return float4{ -v.x, -v.y, -v.z, v.w };
	}
	float4 quatNormalize(float4 v) {
		return math::normalize(v);
	}
	float4 quatMul(float4 q1, float4 q2) {
		float x = q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
		float y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
		float z = q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
		float w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;
		return float4{ x,y,z,w };
	}
	float4 toAngularVelocity(const float4& start, const float4& end, float delta_sec) {
		auto[p_s, r_s, y_s] = quat2eul(start);
		auto[p_e, r_e, y_e] = quat2eul(end);

		float p_rate = (p_e - p_s) / delta_sec;
		float r_rate = (r_e - r_s) / delta_sec;
		float y_rate = (y_e - y_s) / delta_sec;

		//TODO: optimize below
		//Sec 1.3, https://ocw.mit.edu/courses/mechanical-engineering/2-154-maneuvering-and-control-of-surface-and-underwater-vehicles-13-49-fall-2004/lecture-notes/lec1.pdf
		float wx = r_rate + 0 - y_rate * sinf(p_e);
		float wy = 0 + p_rate * cosf(r_e) + y_rate * sinf(r_e) * cosf(p_e);
		float wz = 0 - p_rate * sinf(r_e) + y_rate * cosf(r_e) * cosf(p_e);

		return float4{ wx , wy, wz,0.f };
	}
	World::World() {
		///-----initialization_start-----
		///collision configuration contains default setup for memory, collision setup. Advanced users can create their own configuration.
		collisionConfiguration = new btDefaultCollisionConfiguration();
		///use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see Extras/BulletMultiThreaded)
		dispatcher = new btCollisionDispatcher(collisionConfiguration);
		btGImpactCollisionAlgorithm::registerAlgorithm(dispatcher);
		///btDbvtBroadphase is a good general purpose broadphase. You can also try out btAxis3Sweep.
		overlappingPairCache = new btDbvtBroadphase();
		///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
		solver = new btSequentialImpulseConstraintSolver;
		dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, overlappingPairCache, solver, collisionConfiguration);
		dynamicsWorld->setGravity(bt3fromf3(math::castTo<float3>(get<parameters::external_force>())));
		///-----initialization_end-----
		{
			float3 max = get<parameters::max_domain>();
			float3 min = get<parameters::min_domain>();
			float r = -get<parameters::radius>();
			char prev = 'x';
			for (auto c : get<parameters::domainWalls>()) {
				if (c == 'x' || c == 'y' || c == 'z')
					prev = c;
				float3 dim;
				float sign = -1.f;
				if (c == '+')
					dim = max+0.01f ;
				if (c == '-') {
					dim = -(min-0.01f);
					sign = 1.f;
				}
				btCollisionShape* shape = nullptr;
				if (c == '+' || c == '-') {
					if (prev == 'x')
						shape = (new btBoxShape(btVector3(btScalar(1.f), btScalar(1000.f), btScalar(1000.f))));
					if (prev == 'y')
						shape = (new btBoxShape(btVector3(btScalar(1000.f), btScalar(1.f), btScalar(1000.f))));
					if (prev == 'z')
						shape = (new btBoxShape(btVector3(btScalar(1000.f), btScalar(1000.f), btScalar(1.f))));
				}
				if (shape != nullptr) {
					collisionShapes.push_back(shape);
					btTransform groundTransform;
					groundTransform.setIdentity();
					if (prev == 'x')
						groundTransform.setOrigin(btVector3(-1 * sign - dim.x * sign, 0, 0));
					if (prev == 'y')
						groundTransform.setOrigin(btVector3(0, -1 * sign - dim.y * sign, 0));
					if (prev == 'z')
						groundTransform.setOrigin(btVector3(0, 0, -1 * sign - dim.z * sign));
					btScalar mass(0.);
					bool isDynamic = (mass != 0.f);
					btVector3 localInertia(0, 0, 0);
					//if (isDynamic)
					//	shape->calculateLocalInertia(mass, localInertia);
					//using motionstate is optional, it provides interpolation capabilities, and only synchronizes 'active' objects
					btDefaultMotionState* myMotionState = new btDefaultMotionState(groundTransform);
					btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, myMotionState, shape, localInertia);
					btRigidBody* body = new btRigidBody(rbInfo);
					dynamicsWorld->addRigidBody(body);
				}
			}
		}
		counter = dynamicsWorld->getCollisionObjectArray().size();
		using v_tuple = std::tuple<cudaTextureObject_t, float4, float4, int4, float4, Matrix4x4>;
		std::map<std::string, generation::obj_tuple> boundaryObjects;
		std::map<std::string, v_tuple> volumeObjects;
		for (int32_t b = 0; b < get<parameters::volumeBoundaryCounter>(); ++b) {
			auto boundaryVolume = get<parameters::boundaryVolumes>()[b];
			auto file = resolveFile(boundaryVolume.fileName.value, { get<parameters::config_folder>() });
			auto f_obj = file;
			f_obj.replace_extension(".obj");

			generation::obj_tuple robj;
			v_tuple vobj;

			if (boundaryObjects.find(f_obj.string()) != boundaryObjects.end()) {
				robj = boundaryObjects[f_obj.string()];
			}
			else {
				robj = generation::fileToObj(std::experimental::filesystem::exists(f_obj) ? f_obj : file);
				boundaryObjects[f_obj.string()] = robj;
			}
			if (volumeObjects.find(file.string()) != volumeObjects.end()) {
				vobj = volumeObjects[file.string()];
			}
			else {
				vobj = generation::cudaVolume(boundaryVolume.fileName.value);
				volumeObjects[file.string()] = vobj;
			}
			auto[objVertices, objTriangles, edges, minp, maxp] = robj;
			auto[texture, min, max, dimension, centerOfMass, inertia] = vobj;
			auto V = centerOfMass.w;
			auto mass = V * boundaryVolume.density.value;

			auto numVtx = objVertices.size();
			auto numTri = objTriangles.size();
			std::vector<float4> vertices;

			for (auto& v : objVertices) {
				v -= float3{ centerOfMass.x, centerOfMass.y, centerOfMass.z };
			}
			btTriangleMesh* trimesh = new btTriangleMesh();
			for (int32_t i = 0; i < numTri; ++i) {
				auto[idx1, idx2, idx3] = objTriangles[i];
				auto v0 = objVertices[idx1];
				auto v1 = objVertices[idx2];
				auto v2 = objVertices[idx3];
				btVector3 bv0(v0.x, v0.y, v0.z);
				btVector3 bv1(v1.x, v1.y, v1.z);
				btVector3 bv2(v2.x, v2.y, v2.z);
				trimesh->addTriangle(bv0, bv1, bv2);
			}


			btGImpactMeshShape *colShape = new btGImpactMeshShape(trimesh);
			colShape->updateBound();

			collisionShapes.push_back(colShape);

			/// Create Dynamic Objects
			btTransform startTransform;
			startTransform.setIdentity();
			startTransform.setOrigin(bt3fromf3(boundaryVolume.position.value + math::castTo<float3>(centerOfMass)));
			auto angle = boundaryVolume.angle.value;
			angle = angle / 180.f * CUDART_PI_F;
			startTransform.setRotation(btqfromf4(eul2quat(angle)));

			bool isDynamic = (mass != 0.f);
			if (boundaryVolume.kind.value != 0) {
				isDynamic = false;
				mass = 0.f;
			}
			btMatrix3x3 inertiaMat(
				inertia(0, 0), inertia(0, 1), inertia(0, 2),
				inertia(1, 0), inertia(1, 1), inertia(1, 2),
				inertia(2, 0), inertia(2, 1), inertia(2, 2));
			//inertiaMat.diagonalize();
			btVector3 localInertia(0, 0, 0);

			if (isDynamic)
				colShape->calculateLocalInertia(mass, localInertia);

			//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
			btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);
			btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, myMotionState, colShape, localInertia);
			btRigidBody* body = new btRigidBody(rbInfo);
			if (isDynamic) {
				body->setDamping(0.075f, 0.075f);
				//body->setSleepingThresholds(0.0, 0.0);
				//body->setAngularFactor(0.0);
				//body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
				//body->setActivationState(4);
			}
			body->setActivationState(DISABLE_DEACTIVATION);

			dynamicsWorld->addRigidBody(body);
			boundaryMap[b] = dynamicsWorld->getCollisionObjectArray().size() - 1;

		}
		///-----stepsimulation_end-----
	}
	World::~World() {

		//cleanup in the reverse order of creation/initialization

		///-----cleanup_start-----

		//remove the rigidbodies from the dynamics world and delete them
		for (int32_t i = dynamicsWorld->getNumCollisionObjects() - 1; i >= 0; i--)
		{
			btCollisionObject* obj = dynamicsWorld->getCollisionObjectArray()[i];
			btRigidBody* body = btRigidBody::upcast(obj);
			if (body && body->getMotionState())
			{
				delete body->getMotionState();
			}
			dynamicsWorld->removeCollisionObject(obj);
			delete obj;
		}

		//delete collision shapes
		for (int j = 0; j < collisionShapes.size(); j++)
		{
			btCollisionShape* shape = collisionShapes[j];
			collisionShapes[j] = 0;
			delete shape;
		}

		//delete dynamics world
		delete dynamicsWorld;

		//delete solver
		delete solver;

		//delete broadphase
		delete overlappingPairCache;

		//delete dispatcher
		delete dispatcher;

		delete collisionConfiguration;

		//next line is optional: it will be cleared by the destructor when the array goes out of scope
		collisionShapes.clear();
	}
	void World::step(float dt) {
		dynamicsWorld->stepSimulation(dt, get<parameters::timestep_max>() / get<parameters::timestep_min>() * 4.f, get<parameters::timestep_min>()*0.25f);
	}
	void World::resetWorld() {
		int32_t n = get<parameters::volumeBoundaryCounter>();
		if (n == 0) return;
		std::vector<float4> positions(n), quaternions(n);
		std::vector<float4> velocities(n), angularVelocities(n);
		std::vector<float4> forces(n), torques(n);
		std::vector<float> volumes(n), densities(n);
		std::vector<int32_t> kinds(n);
		std::vector<Matrix4x4> Txs(n), Txs_1(n);

		cudaMemcpy(positions.data(), arrays::volumeBoundaryPosition::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemcpy(quaternions.data(), arrays::volumeBoundaryQuaternion::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemcpy(velocities.data(), arrays::volumeBoundaryVelocity::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemcpy(angularVelocities.data(), arrays::volumeBoundaryAngularVelocity::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
 		cudaMemcpy(forces.data(), arrays::volumeBoundaryAcceleration::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemcpy(torques.data(), arrays::volumeBoundaryAngularAcceleration::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemcpy(volumes.data(), arrays::volumeBoundaryVolume::ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(densities.data(), arrays::volumeBoundaryDensity::ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Txs.data(), arrays::volumeBoundaryTransformMatrix::ptr, n * sizeof(Matrix4x4), cudaMemcpyDeviceToHost);
		cudaMemcpy(Txs_1.data(), arrays::volumeBoundaryTransformMatrixInverse::ptr, n * sizeof(Matrix4x4), cudaMemcpyDeviceToHost);

		auto vols = get<parameters::boundaryVolumes>();
		for (int32_t b = 0; b < get<parameters::volumeBoundaryCounter>(); ++b) {
			auto q = btqfromf4(quaternions[b]);
			auto p = bt3fromf4(positions[b]);
			auto c_obj = dynamicsWorld->getCollisionObjectArray()[boundaryMap[b]];
			auto body = btRigidBody::upcast(c_obj);
			body->getWorldTransform().setRotation(q);
			body->getWorldTransform().setOrigin(p);
			body->setAngularVelocity(bt3fromf4(angularVelocities[b]));
			body->setLinearVelocity(bt3fromf4(velocities[b]));
		}
	}
	World& World::instance() {
		static World bw;
		return bw;
	}

}
