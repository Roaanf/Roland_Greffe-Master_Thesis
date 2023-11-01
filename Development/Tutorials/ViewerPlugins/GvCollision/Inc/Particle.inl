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

#include "Particle.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/
// GigaVoxels
#include <GvStructure/GsVolumeTreeKernel.h>
#include <GvStructure/GsNode.h>


#define MAX_THREADS 64000

void Particles::createParticles( unsigned long long seed, int nParticles ) 
{
	unsigned int nThreadPerBlocs = 64;
	unsigned int nBlocs = nParticles / nThreadPerBlocs + 1;
	if( nBlocs * nThreadPerBlocs > MAX_THREADS ){
		nBlocs = MAX_THREADS / nThreadPerBlocs;
	}
	createParticlesKernel<<< nBlocs, nThreadPerBlocs >>>( seed );
}

/**
 * Fill the array of particles.
 */
__global__ 
void Particles::createParticlesKernel( unsigned long long seed ) 
{
	// Thread id
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// One thread may compute several collision.
	unsigned int it = tid;

	// Init random generator
	curandState state;
	curand_init( seed, it, 0, &state );

	// Each tread may have more than one particle to create.
	while( it < nParticles ) {
		particles[it].init( state );

		// Next particle
		it += blockDim.x * gridDim.x;
	}
}
template< class TVolTreeKernelType, class GPUCacheType >
void Particles::animation( const TVolTreeKernelType pVolumeTree,
			    GPUCacheType pGPUCache,
	   			float3 *vboCollision,
	   			unsigned int nParticles ) {
	unsigned int nThreadPerBlocs = 64;
	unsigned int nBlocs = nParticles / nThreadPerBlocs + 1;
	if( nBlocs * nThreadPerBlocs > MAX_THREADS ){
		nBlocs = MAX_THREADS / nThreadPerBlocs;
	}

	animationKernel<<< nBlocs, nThreadPerBlocs >>>(
			pVolumeTree,
			pGPUCache,
			vboCollision );
}

template< class TVolTreeKernelType, class GPUCacheType >
__global__
void Particles::animationKernel( const TVolTreeKernelType pVolumeTree,
			    GPUCacheType pGPUCache,
	   			float3 *vboCollision ) {
	// Thread id
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// One thread may compute several collision.
	unsigned int it = tid;

	// Set the root node as used (TODO useful ?)
	// TODO : test perfs of the if
	// if( tid == 0 )
	pGPUCache._nodeCacheManager.setElementUsage( 0 );

	// Each tread may have more than one collision to test.
	while( it < Particles::nParticles ) {
		Particle *p = &Particles::particles[it];

		// Displace particle
		if( cRunAnimation ) {
			// Accelerate the particule (gravity)
			p->_speed.z -= Particles::cGravity;

			// Displace particle
			p->_position += p->_speed;

			// Find collision
			float3 normal = p->collision_BBOX_VolTree_Kernel< TVolTreeKernelType, GPUCacheType >( pVolumeTree, pGPUCache );

			// Resolve collision
			p->collisionReaction( normal );
		}

		// Write results in vbo
		vboCollision[it] = p->_position;

		// Next particle
		it += blockDim.x * gridDim.x;
	}
}

/**
 * Initialize a particle with random values.
 */
__device__
void Particle::init( curandState &state ) {
	float4x4 basis;
	basis.m[0].x = 1.f;
	basis.m[0].y = 0.f;
	basis.m[0].z = 0.f;

	basis.m[1].x = 0.f;
	basis.m[1].y = 1.f;
	basis.m[1].z = 0.f;

	basis.m[2].x = 0.f;
	basis.m[2].y = 0.f;
	basis.m[2].z = 1.f;

	this->_speed.x = 0.f;
	this->_speed.y = 0.f;
	//this->_speed.x = -.5f + curand_uniform( &state );
	//this->_speed.y = -.5f + curand_uniform( &state );
	this->_speed.z = 0.f;

	//this->_angularSpeed.x = 0.f;
	//this->_angularSpeed.y = 0.f;
	//this->_angularSpeed.z = 0.f;
	this->_angularSpeed.x = -.5f + curand_uniform( &state );
	this->_angularSpeed.y = -.5f + curand_uniform( &state );
	this->_angularSpeed.z = -.5f + curand_uniform( &state );

	this->_rotation = basis;

	this->_position.x = -.5f + curand_uniform( &state );
	this->_position.y = -.5f + curand_uniform( &state );
	//this->_position.z = -.5f + curand_uniform( &state );
	this->_position.z = 0.7f;

	//this->_position.x = 0.f;
	//this->_position.y = 0.3f;
	//this->_position.z = 0.7f;


	this->_extents.x = 0.01f;
	this->_extents.y = 0.01f;
	this->_extents.z = 0.01f;
}


__device__
void Particle::collisionReaction( float3 normal ) {
	// Collisions
	if( length( normal ) > 0.f ) {
		// Replace the particle outside of the voxel
		if( dot( normal, this->_speed ) <= 0 ) {
			this->_position += normal * length( this->_speed );
		}
		// Change particle speed according to the normal
		this->_speed = Particles::cRebound * ( this->_speed - dot( this->_speed, normal ) * normal );
	}

	// Invisible box
	float3 invisibleBoxSize = make_float3( 1.f, 1.f, 1.5f );
	if( this->_position.z <= -invisibleBoxSize.z ) {
		this->_position.z = -invisibleBoxSize.z;
		// Frictions on the floor
		this->_speed.x *= 0.99f;
		this->_speed.y *= 0.99f;
	}
	if( this->_position.x <= -invisibleBoxSize.x ) {
		this->_speed.x *= -0.4f;
		this->_position.x = -invisibleBoxSize.x;
	}
	if( this->_position.x >= invisibleBoxSize.x ) {
		this->_speed.x *= -0.4f;
		this->_position.x = invisibleBoxSize.x;
	}
	if( this->_position.y <= -invisibleBoxSize.y ) {
		this->_speed.y *= -0.4f;
		this->_position.y = -invisibleBoxSize.y;
	}
	if( this->_position.y >= invisibleBoxSize.y ) {
		this->_speed.y *= -0.4f;
		this->_position.y = invisibleBoxSize.y;
	}
}

/**
 *
 */
class Node {
	public :
	float _size;
	float3 _pos;
	uint _nodeTileAddress;
	uint _nextNode;
};

/******************************************************************************
 * Determine if there is a collision between a BBOX and a Gigavoxel box.
 *
 * @param pVolumeTree The gigavoxel data structure
 * @param pPrecision A given precision
 * @param position The position of the BBOX
 * @param extents The size of the BBOX
 * @param basis An orthonormal basis reflecting the orientation of the BBox
 ******************************************************************************/
template< class TVolTreeKernelType, class GPUCacheType >
__device__
float3 Particle::collision_BBOX_VolTree_Kernel (
			const TVolTreeKernelType pVolumeTree,
			GPUCacheType pGPUCache ) const {
	float3 normal = make_float3( 0.f );
	float3 localNormal;

	int precision = 1; // TODO

	// Stack for traversing the tree
	Node stack[ 32 ]; // TODO use constant

	// Pointer on the head of the stack
	Node * __restrict__ stackPtr = stack + 1;

	// Current GsNode
	GvStructure::GsNode gvNode;

	// Root node
	Node * __restrict__ currentNode;
	currentNode = stackPtr;
	pVolumeTree.fetchNode( gvNode, pVolumeTree._rootAddress );
	currentNode->_size = .5f;
	currentNode->_pos = make_float3( 0.f );
	currentNode->_nodeTileAddress = gvNode.getChildAddress().x;
	currentNode->_nextNode = 0;

	// Basis of the Gvbox.
	float3 extentsGv = make_float3( currentNode->_size );
	float3 positionGv = currentNode->_pos;
	float4x4 basisGv;
	basisGv.m[0].x = 1.f;
	basisGv.m[0].y = 0.f;
	basisGv.m[0].z = 0.f;

	basisGv.m[1].x = 0.f;
	basisGv.m[1].y = 1.f;
	basisGv.m[1].z = 0.f;

	basisGv.m[2].x = 0.f;
	basisGv.m[2].y = 0.f;
	basisGv.m[2].z = 1.f;

	// SAT algo need a matrix representing the rotation between A and B.
	// As A and B are constants throughout the iterations,
	// we can pre-compute it.
	float4x4 rotation, rotationAbs;
	rotation.m[0].x = dot( make_float3( basisGv.m[0] ), make_float3( this->_rotation.m[0] ));
	rotation.m[0].y = dot( make_float3( basisGv.m[0] ), make_float3( this->_rotation.m[1] ));
	rotation.m[0].z = dot( make_float3( basisGv.m[0] ), make_float3( this->_rotation.m[2] ));
	rotation.m[1].x = dot( make_float3( basisGv.m[1] ), make_float3( this->_rotation.m[0] ));
	rotation.m[1].y = dot( make_float3( basisGv.m[1] ), make_float3( this->_rotation.m[1] ));
	rotation.m[1].z = dot( make_float3( basisGv.m[1] ), make_float3( this->_rotation.m[2] ));
	rotation.m[2].x = dot( make_float3( basisGv.m[2] ), make_float3( this->_rotation.m[0] ));
	rotation.m[2].y = dot( make_float3( basisGv.m[2] ), make_float3( this->_rotation.m[1] ));
	rotation.m[2].z = dot( make_float3( basisGv.m[2] ), make_float3( this->_rotation.m[2] ));

	// Compute the absolute value of the rotation matrix
	rotationAbs.m[0].x = fabsf( rotation.m[0].x );
	rotationAbs.m[0].y = fabsf( rotation.m[0].y );
	rotationAbs.m[0].z = fabsf( rotation.m[0].z );
	rotationAbs.m[1].x = fabsf( rotation.m[1].x );
	rotationAbs.m[1].y = fabsf( rotation.m[1].y );
	rotationAbs.m[1].z = fabsf( rotation.m[1].z );
	rotationAbs.m[2].x = fabsf( rotation.m[2].x );
	rotationAbs.m[2].y = fabsf( rotation.m[2].y );
	rotationAbs.m[2].z = fabsf( rotation.m[2].z );

	// BBox size (used to limit the precision).
	float sizeLimit = max( max( this->_extents.x, this->_extents.y ), this->_extents.z ) / precision;
	float bboxVolume = 8 * this->_extents.x * this->_extents.y * this->_extents.z;


	// Test the root node
	if( !GvCollision::sat( positionGv, extentsGv, basisGv, this->_position, this->_extents, rotation, rotationAbs )) {
		// No collision.
		return make_float3( 0.f );
	}

	if( !gvNode.hasSubNodes()) {
		// Root node as no subnodes yet.
		pGPUCache.subDivRequest( pVolumeTree._rootAddress );
		return make_float3(1.f,0.f,0.f); // TODO
	}

	// Main loop : depth-first search for all nodes colliding with the BBox.
	do {
		// Update current node
		currentNode = stackPtr;

		Node son;
		son._size = currentNode->_size * 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );

		// Compute the position of the son in the world coordinates
		// TODO utiliser la fonction symétrique de toFloat1 ?
		uint x = currentNode->_nextNode & 1;
		uint y = ( currentNode->_nextNode & 2 ) >> 1;
		uint z = ( currentNode->_nextNode & 4 ) >> 2;
		son._pos.x = currentNode->_pos.x + x * currentNode->_size - son._size;
		son._pos.y = currentNode->_pos.y + y * currentNode->_size - son._size;
		son._pos.z = currentNode->_pos.z + z * currentNode->_size - son._size;

		// Try to retrieve the node from the node pool given its address
		uint nodeAddress = currentNode->_nodeTileAddress + currentNode->_nextNode;
		pVolumeTree.fetchNode( gvNode, nodeAddress);

		son._nodeTileAddress = gvNode.getChildAddress().x;

		if( ++currentNode->_nextNode >= 8 ) { // TODO 8 => ?
			// No more son to test, unstack the current node
			--stackPtr;
		}

		if( gvNode.isBrick() ) {
			// Non empty node, launch collision tests.
			positionGv = son._pos;
			extentsGv = make_float3( son._size );

			if( GvCollision::sat( positionGv, extentsGv, basisGv, this->_position, this->_extents, rotation, rotationAbs )) {
				// Collision
				// Flag node as used
				pGPUCache._nodeCacheManager.setElementUsage( currentNode->_nodeTileAddress );

				if( !gvNode.hasSubNodes() || son._size < sizeLimit ) {
					// This node as no son or we reached the precision limit.
					if( son._size >= sizeLimit && !gvNode.isTerminal() ) {
						// Emit request to generate son
						pGPUCache.subDivRequest( nodeAddress );
					}
					// Compute the normal of the voxels in the node
					if( gvNode.hasBrick() ) {
						// TODO set brick usage
						float3 brickAddress = make_float3( GvStructure::GsNode::unpackBrickAddress( gvNode.getBrickAddressEncoded() ) ) * pVolumeTree.brickCacheResINV;
						float3 positionInBrick = make_float3( 0.5f ) * pVolumeTree.brickCacheResINV;
						localNormal = make_float3( pVolumeTree.getSampleValueTriLinear<1>( brickAddress, positionInBrick ) );
					} else {
						localNormal = make_float3( 1.f );
						if( x ) localNormal.x *= -1;
						if( y ) localNormal.y *= -1;
						if( z ) localNormal.z *= -1;

						// TODO generation request.
					}
					normal += localNormal;//* son._size;
				} else {
					// Put the son on the top of the stack
					++stackPtr;
					son._nextNode = 0;
					*stackPtr = son;
				}
			}
		}
	} while( stackPtr != stack );

	if( bboxVolume == 0.f || ( normal.x == 0 && normal.y == 0 && normal.z == 0 )) {
		return make_float3( 0.f );
	} else {
		return normalize( normal );
	}
}
