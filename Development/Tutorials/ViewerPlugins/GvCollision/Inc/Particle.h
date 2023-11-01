/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * BSD 3-Clause License:
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the organization nor the names  of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @version 1.0
 */

#ifndef _PARTICLE_H_
#define _PARTICLE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cuda_runtime.h>
#include <vector_types.h>
#include <curand_kernel.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>

// OpenGL
#include <GL/glew.h>

// Project
#include "CollisionDetectorKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/
class Particle;
namespace Particles {

	/**
	 * Whether to run or not the application.
	 */
	__constant__ bool cRunAnimation = false;

	/**
	 * Gravity
	 */
	__constant__ float cGravity = 0.001f;

	/**
	 * Rebound
	 */
	__constant__ float cRebound = 0.8f;

	/**
	 * Fill the array of particles.
	 */
	__global__ 
	void createParticlesKernel( unsigned long long seed );

	void createParticles( unsigned long long seed, int nParticles );

	/**
	 * Array of particles.
	 */
	__device__
	Particle *particles;

	/**
	 * Number of particles in the array.
	 */
	__device__
	unsigned int nParticles;

	/**
	 * Index of the VBO containing the position of the particle (for the display).
	 */
	GLuint positionBuffer;

	/**
	 * VBO containing the position of the particle (for the display).
	 */
	struct cudaGraphicsResource *cuda_vbo_resource;

	/**
	 * Animate the particles.
	 */
	template< class TVolTreeKernelType, class GPUCacheType >
	void animation( const TVolTreeKernelType pVolumeTree,
			GPUCacheType pGPUCache,
			float3 *vboCollision,
			unsigned int nParticles );

	/**
	 * Animate the particles (device part).
	 */
	template< class TVolTreeKernelType, class GPUCacheType >
	__global__
	void animationKernel( const TVolTreeKernelType pVolumeTree,
			GPUCacheType pGPUCache,
			float3 *vboCollision );
}

/**
 * @class Particle
 *
 * @brief TODO
 *
 * TODO
 *
 */
struct Particle
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	/**
	 * Position (center of the particle)
	 */
	float3 _position;

	/**
	 * Size of the x, y and z side.
	 */
	float3 _extents;

	/**
	 * Speed vector of the particle.
	 */
	float3 _speed;

	/**
	 * Basis.
	 */
	float4x4 _rotation;

	/**
	 * Angular speeds of the particle.
	 */
	float3 _angularSpeed;

	/******************************** METHODS *********************************/
	/**
	 * Initialize a particle with random values.
	 */
	__device__
	void init( curandState &state );

	/**
	 *
	 */
	void collisionDetection();

	/**
	 *
	 */
	__device__
	void collisionReaction( float3 normal );

	/**
	 * TODO
	 */
	template< class TVolTreeKernelType, class GPUCacheType >
	__device__
	float3 collision_BBOX_VolTree_Kernel (
				const TVolTreeKernelType pVolumeTree,
				GPUCacheType pGPUCache ) const;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/


	/******************************** METHODS *********************************/


	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

#include "Particle.inl"

#endif // !_PARTICLE_H_
