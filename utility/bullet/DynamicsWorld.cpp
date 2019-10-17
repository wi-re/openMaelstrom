#include "DynamicsWorld.h"

#include <fstream>
#include <iostream>
#include <utility/generation.h>
#include <utility/include_all.h>
#include <utility/mathv2.h>
#include <vector>

#define NO_EXTERNAL
#ifndef NO_EXTERNAL
#include <utility/bullet/Wavefront/tiny_obj_loader.h>
#endif
#include <btBulletDynamicsCommon.h>
//#include <utility/bullet/Utils/b3ResourcePath.h>
#ifndef NO_EXTERNAL
#include <utility/bullet/Utils/b3BulletDefaultFileIO.h>
#endif
#include <Bullet3Common/b3FileUtils.h>
#include <Bullet3Common/b3HashMap.h>
#include <BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h>
#include <BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h>
#include <BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h>
#include <LinearMath/btQuaternion.h>
// #include "BulletDynamics/Dynamics/btRigidBody.h"

//#include "stb_image/stb_image.h"
#ifndef NO_EXTERNAL
#include <utility/bullet/CommonInterfaces/CommonRigidBodyBase.h>
#endif
#include <utility/generation/base_generation.h>

// #include <utility/bullet/ImportMeshUtility/b3ImportMeshUtility.h>

class btRigidBody *m_pickedBody;


struct CachedObjResult {
  std::string m_msg;
#ifndef NO_EXTERNAL
  std::vector<tinyobj::shape_t> m_shapes;
#endif
};

static b3HashMap<b3HashString, CachedObjResult> gCachedObjResults;

std::string LoadFromCachedOrFromObj(
#ifndef NO_EXTERNAL
    std::vector<tinyobj::shape_t> &shapes, // [output]
#endif
    const char *filename, const char *mtl_basepath, struct CommonFileIOInterface *fileIO) {
  CachedObjResult *resultPtr = gCachedObjResults[filename];
  if (resultPtr) {
    const CachedObjResult &result = *resultPtr;
#ifndef NO_EXTERNAL
    shapes = result.m_shapes;
    return result.m_msg;
#endif
  }

#ifndef NO_EXTERNAL
  std::string err = tinyobj::LoadObj(shapes, filename, mtl_basepath, fileIO);
  CachedObjResult result;
  result.m_msg = err;
  result.m_shapes = shapes;
  // if (gEnableFileCaching)
  // {
  // 	gCachedObjResults.insert(filename, result);
  // }
  return err;
#else
	return std::string("");
#endif

}

//DynamicsWorld* DynamicsWorld::instance = 0;

DynamicsWorld* DynamicsWorld::getInstance()
{
	static DynamicsWorld instance;
	return &instance;
    //if (instance == 0)
    //{
    //    instance = new DynamicsWorld();
    //}

    //return instance;
}

float_u<SI::density> DynamicsWorld::getDensity(int32_t rigid_num) const
{
    float_u<SI::density> dens;
    dens.val = rigid_body_density.at(rigid_num - 1);
    return dens;
}

void DynamicsWorld::applyForce(float4 force, int32_t rigid_num, float4 pos){
    btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[rigid_num - 1];
	btRigidBody* body = btRigidBody::upcast(obj);
    btTransform trr;
    body->getMotionState()->getWorldTransform(trr);
    auto orig = trr.getOrigin();
    
    float4 center_x = {(float)orig.getX(), (float)orig.getY(), (float)orig.getZ(), 0};
    auto rel_pos =  pos - (center_x);
    
    
    auto force_x = btVector3(force.x, force.y, force.z);
    auto rel_pos_x = btVector3(rel_pos.x, rel_pos.y, rel_pos.z);
    
    // rigid_bodies.at(rigid_num - 1)->applyTorque(btVector3(100000000, 0, 0));
    // rigid_bodies.at(rigid_num - 1)->applyForce(force_x, btVector3(0, 0, 0));
	static std::mutex m; {
		std::lock_guard<std::mutex> lg(m);
		rigid_bodies.at(rigid_num - 1)->applyForce(force_x, rel_pos_x);
	}    
    // rigid_bodies.at(rigid_num - 1)->applyForce(btVector3(-100000, 0, 0), btVector3(0, 0, 0));
    
}

void DynamicsWorld::applyForceFake(float4 force, float4 torque){
    auto frc = btVector3(force.x, force.y, force.z);
    // auto trq = btVector3(0, 0, 0);
    auto trq = btVector3(torque.x, torque.y, torque.z);
    rigid_bodies.at(0)->applyCentralForce(frc);
    rigid_bodies.at(0)->applyTorque(trq);
    // rigid_bodies.at(rigid_num - 1)->applyForce(btVector3(-100000, 0, 0), btVector3(0, 0, 0));
    
}


float4 DynamicsWorld::getTorque(float4 force, int32_t rigid_num, float4 pos){
    auto center = rigid_centers.at(rigid_num - 1);
    auto center_tmp = rigid_bodies.at(rigid_num - 1)->getCenterOfMassPosition();
    // float4 center_x = {center_tmp.getX(), center_tmp.getY(), center_tmp.getZ(), 0};
    // auto rel_pos =  pos - (center_x);
    // auto cross = math::cross(rel_pos, force);
    
    btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[rigid_num - 1];
	btRigidBody* body = btRigidBody::upcast(obj);
    btTransform trr;
    body->getMotionState()->getWorldTransform(trr);
    auto orig = trr.getOrigin();
    auto quart = trr.getRotation();
    
    float4 center_x = { (float)orig.getX(), (float)orig.getY(), (float)orig.getZ(), 0};
    auto rel_pos =  pos - (center_x);
    auto cross = math::cross(rel_pos, force);
    float len = sqrt(rel_pos.x*rel_pos.x + rel_pos.y*rel_pos.y + rel_pos.z*rel_pos.z);

    return cross;
}

void DynamicsWorld::addPlane(float x, float y, float z, float xx, float yy, float zz)
{
    btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(x), btScalar(y), btScalar(z)));

    btTransform groundTransform;
    groundTransform.setIdentity();
    groundTransform.setOrigin(btVector3(xx, yy, zz));

    btScalar mass(0.);

    btVector3 localInertia(0, 0, 0);

    //using motionstate is optional, it provides interpolation capabilities, and only synchronizes 'active' objects
    btDefaultMotionState* myMotionState = new btDefaultMotionState(groundTransform);
    btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, myMotionState, groundShape, localInertia);
    btRigidBody* body = new btRigidBody(rbInfo);

    //add the body to the dynamics world
    m_dynamicsWorld->addRigidBody(body);
}

void DynamicsWorld::createBoundingBox()
{
    float3 max = get<parameters::max_domain>();
	float3 min = get<parameters::min_domain>();

    float width = 0.00001;
    // float width = 4;
    addPlane(width, max.y - min.y, max.z - min.z, min.x, (max.y + min.y)/2, (max.z + min.z)/2);
    addPlane(width, max.y - min.y, max.z - min.z, max.x, (max.y + min.y)/2, (max.z + min.z)/2);
    addPlane(max.x - min.x, width, max.z - min.z, (max.x + min.x)/2, min.y, (max.z + min.z)/2);
    addPlane(max.x - min.x, width, max.z - min.z, (max.x + min.x)/2, max.y, (max.z + min.z)/2);
    addPlane(max.x - min.x, max.y - min.y, width, (max.x + min.x)/2, (max.y + min.y)/2, min.z);
    addPlane(max.x - min.x, max.y - min.y, width, (max.x + min.x)/2, (max.y + min.y)/2, max.z);
}

void DynamicsWorld::createWorld()
{
	throw std::exception("REEEEEEEEEEEEEEEEEEE");
    
    ///collision configuration contains default setup for memory, collision setup. Advanced users can create their own configuration.
	m_collisionConfiguration = new btDefaultCollisionConfiguration();

	///use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see Extras/BulletMultiThreaded)
	m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);
    btGImpactCollisionAlgorithm::registerAlgorithm(m_dispatcher);
	
    ///btDbvtBroadphase is a good general purpose broadphase. You can also try out btAxis3Sweep.
	m_broadphase = new btDbvtBroadphase();

	///the default constraint32_t solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
	m_solver = new btSequentialImpulseConstraintSolver;

	m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher, m_broadphase, m_solver, m_collisionConfiguration);

    auto force = get<parameters::external_force>(); 
	// m_dynamicsWorld->setGravity(btVector3(force.x, force.y, force.z));
	m_dynamicsWorld->setGravity(btVector3(0, 0, 0));
}

void DynamicsWorld::stepSimulation(float step)
{
    m_dynamicsWorld->stepSimulation(step);
}

int32_t DynamicsWorld::getRigidBodiesCount()
{
	static std::mutex m;
	std::lock_guard<std::mutex> lg(m);
    return rigid_bodies.size();
}

btTransform DynamicsWorld::getRigidTransform(int32_t num)
{
    auto transform = rigid_bodies.at(num)->getCenterOfMassTransform();
    auto center = rigid_bodies.at(num)->getCenterOfMassPosition();
    return transform;
}

void DynamicsWorld::updateCenter(int32_t rigid_num)
{
	static std::mutex m;
	std::lock_guard<std::mutex> lg(m);
    btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[rigid_num];
	btRigidBody* body = btRigidBody::upcast(obj);
    btTransform trr;
    body->getMotionState()->getWorldTransform(trr);
    auto orig = trr.getOrigin();
    auto quart = trr.getRotation();

    float4 orig_offset = { (float)orig.getX(), (float)orig.getY(), (float)orig.getZ(), 0};

    float4 outc = rigid_centers.at(rigid_num);
    // std::cout << "centerso: " << outc.x << " " << outc.y << " " << outc.z << std::endl;
    // std::cout << "centersn: " << orig_offset.x << " " << orig_offset.y << " " << orig_offset.z << std::endl;
    rigid_origins.at(rigid_num) = orig_offset;
    rigid_quaternion.at(rigid_num) = quart;

    auto vel = body->getLinearVelocity();
    arrays::rigidLinearVelocities::ptr[rigid_num] = {(float)vel.getX(), (float)vel.getY(), (float)vel.getZ(), 0};
    auto avel = body->getAngularVelocity();
    arrays::rigidAVelocities::ptr[rigid_num] = { (float)avel.getX(), (float)avel.getY(), (float)avel.getZ()};
    arrays::rigidOrigins::ptr[rigid_num] = { (float)orig.getX(), (float)orig.getY(), (float)orig.getZ()};
    arrays::rigidQuaternions::ptr[rigid_num] = { (float)quart.getX(), (float)quart.getY(), (float)quart.getZ(), (float)quart.getW()};
    LOG_VERBOSE << "move origin: " << orig.getX() << " " << orig.getY() << " " << orig.getZ() << std::endl;
	LOG_VERBOSE << "move quater: " << quart.getX() << " " << quart.getY() << " " << quart.getZ() << " " << quart.getW() << std::endl;
    // rigid_centers.at(rigid_num) += orig_offset;
}

void DynamicsWorld::clearForces(int32_t rigid_num)
{
	static std::mutex m;
	std::lock_guard<std::mutex> lg(m);
    btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[rigid_num];
	btRigidBody* body = btRigidBody::upcast(obj);
    body->clearForces();
}

void DynamicsWorld::getTotalForce(int32_t rigid_num)
{
    btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[rigid_num];
	btRigidBody* body = btRigidBody::upcast(obj);
    auto frc = body->getTotalForce();
    auto trq = body->getTotalTorque();
	LOG_VERBOSE << "Total Force: " << frc.getX() << " " << frc.getY() << " " << frc.getZ() << std::endl;
	LOG_VERBOSE << "Total Torque: " << trq.getX() << " " << trq.getY() << " " << trq.getZ() << std::endl;
}

void DynamicsWorld::writeRotation()
{
    btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[0];
	btRigidBody* body = btRigidBody::upcast(obj);
    btTransform trr;
    body->getMotionState()->getWorldTransform(trr);
    auto quart = trr.getRotation();
    auto orig = trr.getOrigin();
    auto torque = body->getTotalTorque();
    
    std::ofstream myfile;
    // myfile.open("force.txt", std::ios::app | std::ios::out | std::ios::in);
    // // myfile << quart.getX() << " " << quart.getY() << " " << quart.getZ() << " " << quart.getW() << std::endl;
    // myfile << debugSum.val.x << " " << debugSum.val.y << " " << debugSum.val.z << std::endl;
    // myfile.close();


    // std::ofstream myfile2;
    // myfile2.open("origin.txt", std::ios::app | std::ios::out | std::ios::in);
    // myfile2 << orig.getX() << " " << orig.getY() << " " << orig.getZ() << std::endl;
    // myfile2.close();
}



float4 DynamicsWorld::moveRigidParticle(float4 pos, int32_t rigid_num)
{
	static std::mutex m;
	std::lock_guard<std::mutex> lg(m);
    // printf("pos   x: %f y: %f z: %f\n", pos.x, pos.y, pos.z);
    
    btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[rigid_num];
	btRigidBody* body = btRigidBody::upcast(obj);


    //auto transform = getRigidTransform(rigid_num);
    btTransform trr;
    body->getMotionState()->getWorldTransform(trr);
    auto quart = trr.getRotation();
    // btQuaternion quart = btQuaternion(-0.006234, -0.0202283, 7.45331e-07, 0.994135);
    auto orig = trr.getOrigin();
    
    auto old_quat = rigid_quaternion.at(rigid_num);
    auto masscenter = rigid_origins.at(rigid_num);

    // subtract masscenter pos from abs pos to get relative pos 
    float4 mscenter = { (float)orig.getX(), (float)orig.getY(),(float)orig.getZ(), 0};
    // auto pos1 = pos - mscenter;
    auto pos1 = pos - masscenter;
    //build quarterion from relative pos
    btQuaternion vec = btQuaternion(pos1.x, pos1.y, pos1.z, 0);
    // auto resquart = vec;
    // auto resquart = quart * vec * quart.inverse();
    auto resquart = quart * (old_quat.inverse() * vec * old_quat) * quart.inverse();
    //printf("x: %f y: %f z: %f\n", resquart.getX(), resquart.getY(), resquart.getZ());
    // float4 transformed_point = {pos1.x, pos1.y, pos1.z, 0};


    float4 transformed_point = { (float)resquart.getX(), (float)resquart.getY(), (float)resquart.getZ(), 0};
    float4 orig_offset = { (float)orig.getX(), (float)orig.getY(), (float)orig.getZ(), 0};
    // std::cout<< "transformed1 " << transformed_point.x << " "  << transformed_point.y << " "  << transformed_point.z << std::endl;
    //transformed_point += (orig_offset - masscenter);
    transformed_point += mscenter;
    float4 return_point = {transformed_point.x, transformed_point.y, transformed_point.z, pos.w};
    
    // std::cout<< "transformed2 " << transformed_point.x << " "  << transformed_point.y << " "  << transformed_point.z << std::endl;
    // std::cout<< "old point32_t   " << pos.x << " "  << pos.y << " "  << pos.z << std::endl;
    // std::cout<< "old masscenter   " << masscenter.x << " "  << masscenter.y << " "  << masscenter.z << std::endl;
    // std::cout<< "orig   " << orig.getX() << " "  << orig.getY() << " "  << orig.getZ() << std::endl;
    // std::cout<< "old quart   " << old_quat.getX() << " "  << old_quat.getY() << " "  << old_quat.getZ() << " "  << old_quat.getW() << std::endl;
    // std::cout<< "cur quart   " << quart.getX() << " "  << quart.getY() << " "  << quart.getZ() << " "  << quart.getW() << std::endl;
    return return_point;

    // return {10, 10, 10, 0};
}

void findMaxMinValues(float& max_x, float& max_y, float& max_z, float& min_x, float& min_y, float& min_z, 
    float* pnt0, float* pnt1, float* pnt2, bool& first)
{
    if (!first)
        max_x = std::max(std::max(std::max(max_x, pnt0[0]), pnt1[0]), pnt2[0]);
    else max_x = std::max(std::max(pnt0[0], pnt1[0]), pnt2[0]);
    
    if (!first)
        max_y = std::max(std::max(std::max(max_y, pnt0[1]), pnt1[1]), pnt2[1]);
    else max_y = std::max(std::max(pnt0[1], pnt1[1]), pnt2[1]);

    if (!first)
        max_z = std::max(std::max(std::max(max_z, pnt0[2]), pnt1[2]), pnt2[2]);
    else max_z = std::max(std::max(pnt0[2], pnt1[2]), pnt2[2]);
    
    if (!first)
        min_x = std::min(std::min(std::min(min_x, pnt0[0]), pnt1[0]), pnt2[0]);
    else min_x = std::min(std::min(pnt0[0], pnt1[0]), pnt2[0]);

    if (!first)
        min_y = std::min(std::min(std::min(min_y, pnt0[1]), pnt1[1]), pnt2[1]);
    else min_y = std::min(std::min(pnt0[1], pnt1[1]), pnt2[1]);

    if (!first)
        min_z = std::min(std::min(std::min(min_z, pnt0[2]), pnt1[2]), pnt2[2]);
    else min_z = std::min(std::min(pnt0[2], pnt1[2]), pnt2[2]);

    first = false;
            
}

std::map<std::string, float3> DynamicsWorld::addInfoBody(const std::string fileName, const float density, const float3 shift)
{
    rigid_body_file.push_back(fileName);
    rigid_body_density.push_back(density);
    rigid_body_shift.push_back(shift);

#ifndef NO_EXTERNAL
    b3BulletDefaultFileIO fileIO;

    char relativeFileName[1024];
    auto path = resolveFile(fileName, {get<parameters::config_folder>()});
    if (fileIO.findResourcePath(path.string().c_str(), relativeFileName, 1024))
    {
        char pathPrefix[1024];

        b3FileUtils::extractPath(relativeFileName, pathPrefix, 1024);
        btVector3 shift(0, 0, 0);

        std::vector<tinyobj::shape_t> shapes;
        {
            B3_PROFILE("tinyobj::LoadObj");
            std::string err = LoadFromCachedOrFromObj(shapes, relativeFileName, pathPrefix, &fileIO);
            //std::string err = tinyobj::LoadObj(shapes, relativeFileName, pathPrefix);
        }


        tinyobj::shape_t& shape = shapes[0];
        int32_t faceCount = shape.mesh.indices.size();

        btTriangleMesh* trimesh = new btTriangleMesh();
            
        float max_x = 0, max_y = 0, max_z = 0, min_x = 0, min_y = 0, min_z = 0;
        bool first = true;
        for (int32_t f = 0; f < faceCount; f += 3)
        {
            btVector3 normal(0, 1, 0);
            
            if (f < 0 && f >= int(shape.mesh.indices.size()))
            {
                continue;
            }

            float pnt0[3], pnt1[3], pnt2[3];
            pnt0[0] = shape.mesh.positions[shape.mesh.indices[f] * 3 + 0];
            pnt0[1] = shape.mesh.positions[shape.mesh.indices[f] * 3 + 1];
            pnt0[2] = shape.mesh.positions[shape.mesh.indices[f] * 3 + 2];

            pnt1[0] = shape.mesh.positions[shape.mesh.indices[f + 1] * 3 + 0];
            pnt1[1] = shape.mesh.positions[shape.mesh.indices[f + 1] * 3 + 1];
            pnt1[2] = shape.mesh.positions[shape.mesh.indices[f + 1] * 3 + 2];

            pnt2[0] = shape.mesh.positions[shape.mesh.indices[f + 2] * 3 + 0];
            pnt2[1] = shape.mesh.positions[shape.mesh.indices[f + 2] * 3 + 1];
            pnt2[2] = shape.mesh.positions[shape.mesh.indices[f + 2] * 3 + 2];

            findMaxMinValues(max_x, max_y, max_z, min_x, min_y, min_z, pnt0, pnt1, pnt2, first);
            
            btVector3 vertex0(pnt0[0], pnt0[1], pnt0[2]);
            btVector3 vertex1(pnt1[0], pnt1[1], pnt1[2]);
            btVector3 vertex2(pnt2[0], pnt2[1], pnt2[2]);
            
            trimesh->addTriangle(vertex0, vertex1, vertex2);
            first = false;
        }
        
        std::map<std::string, float3> tmpmm;
        tmpmm["max"] = {max_x, max_y, max_z};
        tmpmm["min"] = {min_x, min_y, min_z};

        maxmin.push_back(tmpmm);

        return tmpmm;
    }
#else
    return std::map<std::string, float3>{};
#endif
}

std::map<std::string, float3> DynamicsWorld::addBody(float vol, int32_t index)
{
    btVector3 center = btVector3((maxmin[index]["max"].x + maxmin[index]["min"].x)/2, 
        (maxmin[index]["max"].y + maxmin[index]["min"].y)/2, (maxmin[index]["max"].z + maxmin[index]["min"].z)/2);
    auto fileName = rigid_body_file[index];
    // auto mass = ridi
    float mass = vol * rigid_body_density[index];
    auto shiftr = rigid_body_shift[index];
    LOG_INFO << "mass: " << mass << std::endl;

#ifndef NO_EXTERNAL
    b3BulletDefaultFileIO fileIO;

    char relativeFileName[1024];
    auto path = resolveFile(fileName, {get<parameters::config_folder>()});
    if (fileIO.findResourcePath(path.string().c_str(), relativeFileName, 1024))
    {
        char pathPrefix[1024];

        b3FileUtils::extractPath(relativeFileName, pathPrefix, 1024);
        btVector3 shift(0, 0, 0);

        std::vector<tinyobj::shape_t> shapes;
        {
            B3_PROFILE("tinyobj::LoadObj");
            std::string err = LoadFromCachedOrFromObj(shapes, relativeFileName, pathPrefix, &fileIO);
            //std::string err = tinyobj::LoadObj(shapes, relativeFileName, pathPrefix);
        }


        tinyobj::shape_t& shape = shapes[0];
        int32_t faceCount = shape.mesh.indices.size();

        btTriangleMesh* trimesh = new btTriangleMesh();
            
        float max_x = 0, max_y = 0, max_z = 0, min_x = 0, min_y = 0, min_z = 0;
        bool first = true;
        for (int32_t f = 0; f < faceCount; f += 3)
        {
            btVector3 normal(0, 1, 0);
            
            if (f < 0 && f >= int(shape.mesh.indices.size()))
            {
                continue;
            }

            float pnt0[3], pnt1[3], pnt2[3];
            pnt0[0] = shape.mesh.positions[shape.mesh.indices[f] * 3 + 0];
            pnt0[1] = shape.mesh.positions[shape.mesh.indices[f] * 3 + 1];
            pnt0[2] = shape.mesh.positions[shape.mesh.indices[f] * 3 + 2];

            pnt1[0] = shape.mesh.positions[shape.mesh.indices[f + 1] * 3 + 0];
            pnt1[1] = shape.mesh.positions[shape.mesh.indices[f + 1] * 3 + 1];
            pnt1[2] = shape.mesh.positions[shape.mesh.indices[f + 1] * 3 + 2];

            pnt2[0] = shape.mesh.positions[shape.mesh.indices[f + 2] * 3 + 0];
            pnt2[1] = shape.mesh.positions[shape.mesh.indices[f + 2] * 3 + 1];
            pnt2[2] = shape.mesh.positions[shape.mesh.indices[f + 2] * 3 + 2];

            findMaxMinValues(max_x, max_y, max_z, min_x, min_y, min_z, pnt0, pnt1, pnt2, first);
            
            btVector3 vertex0(pnt0[0], pnt0[1], pnt0[2]);
            btVector3 vertex1(pnt1[0], pnt1[1], pnt1[2]);
            btVector3 vertex2(pnt2[0], pnt2[1], pnt2[2]);
            
            trimesh->addTriangle(vertex0 - center, vertex1 - center, vertex2 - center);
            first = false;
        }

        btGImpactMeshShape *tmpshape = new btGImpactMeshShape(trimesh);
        //btBvhTriangleMeshShape *tmpshape = new btBvhTriangleMeshShape(trimesh, false);
        tmpshape->updateBound();
        // btCollisionShape* tmpshape = new btBoxShape(btVector3((max_x - min_x)/2, (max_y - min_y)/2, (max_z - min_z)/2));

		LOG_INFO << "max min: " << max_x << " " << max_y << " " << max_z << " " << min_x << " " << min_y << " " << min_z
            << "\n avg: " << (max_z + min_z)/2 << std::endl;
        btVector3 localInertia(0, 0, 0);
        tmpshape->calculateLocalInertia(mass, localInertia);
        btTransform trans;
        trans.setIdentity();
        if (arrays::rigidQuaternions::ptr != nullptr)
        {
            float4 quartt = arrays::rigidQuaternions::ptr[index];
            trans.setRotation(btQuaternion(quartt.x, quartt.y, quartt.z, quartt.w));
        }
        auto cnt = btVector3((max_x + min_x)/2 + shiftr.x, (max_y + min_y)/2 + shiftr.y, (max_z + min_z)/2 + shiftr.z);
        // trans.setOrigin(cnt);
        if (arrays::rigidOrigins::ptr != nullptr)
        {
            float3 originn = arrays::rigidOrigins::ptr[index];
            cnt = btVector3(originn.x, originn.y, originn.z);
        }

        trans.setOrigin(cnt);
        
        btDefaultMotionState* motionState = new btDefaultMotionState(trans);
        btRigidBody::btRigidBodyConstructionInfo info(mass, motionState, tmpshape, localInertia);
        btRigidBody* body = new btRigidBody(info);
        body->setActivationState(DISABLE_DEACTIVATION);

        if (arrays::rigidLinearVelocities::ptr != nullptr)
        {
            float4 vell = arrays::rigidLinearVelocities::ptr[index];
            body->setLinearVelocity(btVector3(vell.x, vell.y, vell.z));
            float3 avell = arrays::rigidAVelocities::ptr[index];
            body->setAngularVelocity(btVector3(avell.x, avell.y, avell.z));
        }
        // body->setContactProcessingThreshold(BT_LARGE_FLOAT);
        // body->setCcdMotionThreshold(.5);
        // body->setCcdSweptSphereRadius(0);

        
        rigid_bodies.push_back(body);
        
        m_dynamicsWorld->addRigidBody(body);

        auto center = body->getCenterOfMassPosition();
        float4 cent = { (float)cnt.getX(), (float)cnt.getY(), (float)cnt.getZ(), 0};
        float4 orig = { (float)center.getX(), (float)center.getY(), (float)center.getZ(), 0};
		LOG_INFO << "center: " << center.getX() << " " << center.getY() << " " << center.getZ() << std::endl;
        rigid_centers.push_back(cent);
        rigid_origins.push_back(orig);
        rigid_quaternion.push_back(trans.getRotation());

        std::map<std::string, float3> ret;
        ret["max"] = {max_x, max_y, max_z};
        ret["min"] = {min_x, min_y, min_z};

        return ret;
    }
#else
    return std::map<std::string, float3>{};
#endif

}

btAlignedObjectArray<btCollisionObject*> DynamicsWorld::getRigids()
{
    auto tmp = m_dynamicsWorld->getCollisionObjectArray(); 
    return tmp;
}

float4 DynamicsWorld::getLinearVelocity(int32_t rigid_num)
{
	static std::mutex m;
	std::lock_guard<std::mutex> lg(m);
    btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[rigid_num];
	btRigidBody* body = btRigidBody::upcast(obj);
    auto frc = body->getLinearVelocity();
    return { (float)frc.getX(), (float)frc.getY(), (float)frc.getZ(), 0.f};
}

void DynamicsWorld::addBoundary(const std::string fileName)
{
    auto path = resolveFile(fileName, {get<parameters::config_folder>()});
    //std::vector<float3>, std::vector<Triangle>, std::vector<Edge>, float3, float3
    auto[vertices, triangles, edges, minaabb, maxaabb] = generation::fileToObj(path);
    
	LOG_INFO << "cnt: " << triangles.size() << " min max : " << minaabb.x <<
        " " << minaabb.y << " " << minaabb.z << " " << maxaabb.x << " " << maxaabb.y << " " << maxaabb.z << std::endl;
    
    btTriangleMesh* trimesh = new btTriangleMesh();
        
    for(std::vector<generation::Triangle>::iterator it = triangles.begin(); it != triangles.end(); ++it) 
    {
        float3 pnt0, pnt1, pnt2;
        pnt0 = vertices[(*it).i0];
        pnt1 = vertices[(*it).i1];
        pnt2 = vertices[(*it).i2];
        

        btVector3 vertex0(pnt0.x, pnt0.y, pnt0.z);
        btVector3 vertex1(pnt1.x, pnt1.y, pnt1.z);
        btVector3 vertex2(pnt2.x, pnt2.y, pnt2.z);
        
        trimesh->addTriangle(vertex0, vertex1, vertex2);
    }

    btBvhTriangleMeshShape *tmpshape = new btBvhTriangleMeshShape(trimesh, false);
    // btGImpactMeshShape * tmpshape = new btGImpactMeshShape(trimesh);
	// tmpshape->updateBound();	

    btVector3 localInertia(0, 0, 0);
    tmpshape->calculateLocalInertia(0.f, localInertia);
    btTransform trans;
    trans.setIdentity();
    trans.setOrigin(btVector3(0, 0, 0));
    btDefaultMotionState* motionState = new btDefaultMotionState(trans);
    btRigidBody::btRigidBodyConstructionInfo info(0.f, motionState, tmpshape, localInertia);
    btRigidBody* body = new btRigidBody(info);
    
    m_dynamicsWorld->addRigidBody(body);
}
