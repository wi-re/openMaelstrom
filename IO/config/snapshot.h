#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <utility/identifier.h>
#include <utility/cuda/memory.h>
#include <vector>
#include <utility/bullet/DynamicsWorld.h>

namespace IO {
namespace config {
struct SnapShot {
  virtual void save() = 0;
  virtual void load() = 0;
  virtual void clear() = 0;
}; 

template <typename Ty> struct UniformSnap : public SnapShot {
  using T = typename Ty::type;
  T *original_ptr = Ty::ptr;
  T lastValue = *Ty::ptr;

  virtual void save() override { lastValue = *original_ptr; }
  virtual void load() override { *original_ptr = lastValue; }
  virtual void clear() override{};
};

template <typename Ty> struct ArraySnap : public SnapShot {
  using T = typename Ty::type;
  T *original_ptr = nullptr;
  T *last_ptr = nullptr;
  size_t allocSize;
  virtual void load() override {
    original_ptr = Ty::ptr;
    if (last_ptr == nullptr) {
      return;
    }
    cuda::memcpy(original_ptr, last_ptr, allocSize, cudaMemcpyHostToDevice);
  }
  virtual void save() override {
    original_ptr = Ty::ptr;
    if (original_ptr == nullptr)
      return;
    if (last_ptr == nullptr) {
      last_ptr = (T *)malloc(Ty::alloc_size);
      allocSize = Ty::alloc_size;
    }
    cuda::memcpy(last_ptr, original_ptr, allocSize, cudaMemcpyDeviceToHost);
  }
  virtual void clear() override {
    original_ptr = nullptr;
    if (last_ptr != nullptr)
      free(last_ptr);
    last_ptr = nullptr;
    allocSize = 0;
  }
};

struct RigidSnap : public SnapShot {
  
  std::vector<btTransform> trans;
  std::vector<btVector3> vel;
  std::vector<btVector3> avel;
  std::vector<btQuaternion> rquat;

  virtual void load() override {
    auto wrld = DynamicsWorld::getInstance();
    auto rgds = wrld->getRigids();
    auto cnt = wrld->getRigidBodiesCount();

    for (int i = 0; i < cnt; i++)
    {
      auto vl = vel[i];
      auto avl = avel[i];
      auto tr = trans[i];
      auto rqt = rquat[i];
      std::cout << "load: " << tr.getOrigin().getX() << " " << tr.getOrigin().getY() << " " << tr.getOrigin().getZ() << std::endl;
      
      auto rg = rgds[i];
      btRigidBody* body = btRigidBody::upcast(rg);
      body->setLinearVelocity(vl);
      body->setAngularVelocity(avl);

      body->setCenterOfMassTransform(tr);
      body->clearForces();
      
      auto orig = tr.getOrigin();
      wrld->rigid_origins.at(i) = {(float)orig.getX(), (float)orig.getY(), (float)orig.getZ(), 0};
      wrld->rigid_quaternion.at(i) = rqt;
    }
  }
  virtual void save() override {
    trans.clear();
    vel.clear();
    avel.clear();
    rquat.clear();
    auto wrld = DynamicsWorld::getInstance();
    auto rgds = wrld->getRigids();
    auto cnt = wrld->getRigidBodiesCount();

    for (int i = 0; i < cnt; i++)
    {
      auto rg = rgds[i];
      btRigidBody* body = btRigidBody::upcast(rg);
      vel.push_back(body->getLinearVelocity());
      avel.push_back(body->getAngularVelocity());
      auto tr = body->getCenterOfMassTransform();
      trans.push_back(tr);
      rquat.push_back(wrld->rigid_quaternion.at(i));
      std::cout << "save: " << tr.getOrigin().getX() << " " << tr.getOrigin().getY() << " " << tr.getOrigin().getZ() << std::endl;
    }

  }
  virtual void clear() override {
    
  }
};

extern std::vector<SnapShot *> snaps;
} // namespace config
} // namespace IO
