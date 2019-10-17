#pragma once


//headers for bullet
//#include "ImportObjExample.h"
#include <btBulletDynamicsCommon.h>
#include <vector>
#include <utility/include_all.h>

class DynamicsWorld
{
    private:
        /* Here will be the instance stored. */
        //static DynamicsWorld* instance;

        btDiscreteDynamicsWorld* m_dynamicsWorld;

        btBroadphaseInterface* m_broadphase;
        btCollisionDispatcher* m_dispatcher;
        btConstraintSolver* m_solver;
        btDefaultCollisionConfiguration* m_collisionConfiguration;
        std::vector<std::string> rigid_body_file;
        std::vector<float> rigid_body_density;
        std::vector<float3> rigid_body_shift;
        std::vector<float4> rigid_centers;
        
        
        std::vector<std::map<std::string, float3>> maxmin;
        /* Private constructor to prevent instancing. */
        DynamicsWorld(): m_broadphase(0),
		  m_dispatcher(0),
		  m_solver(0),
		  m_collisionConfiguration(0),
		  m_dynamicsWorld(0){};
    public:
        std::vector<btRigidBody*> rigid_bodies;
        std::vector<btQuaternion> rigid_quaternion;
        std::vector<float4> rigid_origins;
        /* Static access method. */
        static DynamicsWorld* getInstance();
        
        btAlignedObjectArray<btCollisionObject*> getRigids();

        std::map<std::string, float3> addBody(float vol, int32_t index);

        float_u<SI::density> getDensity(int32_t rigid_num) const;

        std::map<std::string, float3>  addInfoBody(const std::string fileName, const float density, const float3 shift);

        void createWorld();

        void applyForce(float4 force, int32_t rigid_num, float4 pos);

        void stepSimulation(float step);

        int32_t getRigidBodiesCount();

        btTransform getRigidTransform(int32_t num);

        float4 moveRigidParticle(float4 pos, int32_t rigid_num);

        void updateCenter(int32_t rigid_num);

        void createBoundingBox();

        void addPlane(float x, float y, float z, float xx, float yy, float zz);

        void addBoundary(const std::string fileName);

        void clearForces(int32_t rigid_num);

        void getTotalForce(int32_t rigid_num);

        void writeRotation();

        float4 getTorque(float4 force, int32_t rigid_num, float4 pos);

        void applyForceFake(float4 force, float4 torque);
        
        float4 getLinearVelocity(int32_t rigid_num);
};