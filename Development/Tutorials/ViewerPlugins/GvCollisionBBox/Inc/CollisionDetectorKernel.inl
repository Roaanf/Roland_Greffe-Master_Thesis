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

// Gigavoxel
#include <GvStructure/GsVolumeTreeKernel.h>
#include <GvStructure/GsNode.h>

namespace GvCollision {

/******************************************************************************
 *
 *
 * @param pVolumeTree the data structure
 * @param pPoint A given position in space
 * @param pPrecision A given precision
 ******************************************************************************/
template< class TVolTreeKernelType >
__global__
void collision_Point_VolTree_Kernel( TVolTreeKernelType pVolumeTree,
		    float3 pPoint,
		    float pPrecision )
{
	if( pPoint.x > 1 || pPoint.x < 0 ||
		pPoint.y > 1 || pPoint.y < 0 ||
		pPoint.z > 1 || pPoint.z < 0 ) {
			collision = false;
			return;
	}

	// Useful variables initialization
	uint nodeDepth = 0;
	GvStructure::GsNode pNode;
	float3 nodePosTree = make_float3( 0.0f );
	float pNodeSizeTree = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	float nodeSizeTreeInv = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	float voxelSizeTree = pNodeSizeTree / static_cast< float >( TVolTreeKernelType::BrickResolution::maxRes );

	// Initialize the address of the first node in the "node pool".
	// While traversing the data structure, this address will be
	// updated to the one associated to the current traversed node.
	// It will be used to fetch info of the node stored in the "node pool".
	uint nodeTileAddress = TVolTreeKernelType::NodeResolution::getNumElements();

	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	bool descentSizeCriteria;
	do
	{
		// [ 1 ] - Update size parameters
		pNodeSizeTree		*= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current node size
		voxelSizeTree		*= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current voxel size
		nodeSizeTreeInv		*= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );			// current node resolution (nb nodes in a dimension)

		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"
		uint3 nodeChildCoordinates = make_uint3( nodeSizeTreeInv * ( pPoint - nodePosTree ) );
		uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );
		uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
		nodePosTree = nodePosTree + pNodeSizeTree * make_float3( nodeChildCoordinates );

		// Try to retrieve node from the node pool given its address
		pVolumeTree.fetchNode( pNode, nodeAddress );

		// Update descent condition
		descentSizeCriteria = ( voxelSizeTree > pPrecision ) && ( nodeDepth < k_maxVolTreeDepth );

		// Update octree depth
		++nodeDepth;

		nodeTileAddress = pNode.getChildAddress().x;
	}
	while ( descentSizeCriteria && pNode.hasSubNodes() );	// END of the data structure traversal


	collision = pNode.isBrick();
}

class Node {
	public :
	float _size;

	float3 _pos;

	uint _nodeTileAddress;

	uint _nextNode;
};

/**
 * @return the absolute value of x.
 */
__device__ __forceinline__ float fabsf(const float &x) {
	unsigned int i = *( unsigned int* ) &x;
	i &= 0x7FFFFFFF; // Force sign bit to 0.
	return *( float* ) &i;
}

/**
 * Implementation of the SAT algorithm (collision detection between two OBB).
 * @param Pa position of the first bounding box
 * @param a extents of the first bounding box
 * @param A orthonormal basis reflecting the first bounding box orientation
 * @param Pb position of the second bounding box
 * @param b extents of the second bounding box
 * @param TODO
 */
__device__
bool sat(
		const float3 &Pa,
		const float3 &a,
		const float4x4 &A,
		const float3 &Pb,
		const float3 &b,
		const float4x4 &R,
		const float4x4 &Rabs

		) {
	// Translation, in parent frame
	float4 v = make_float4(Pb - Pa, 0);

	// Translation, in A's frame
	float3 T = make_float3 ( dot(v, A.m[0]), dot(v, A.m[1]), dot(v, A.m[2]) );

	// B's basis with respect to A's local frame
	float ra, rb, t;

	// ALGORITHM: Use the separating axis test for all 15 potential
	// separating axes. If a separating axis could not be found, the two
	// boxes overlap.

	// A's basis vectors
	ra = a.x;
	rb = b.x*Rabs.m[0].x + b.y*Rabs.m[0].y + b.z*Rabs.m[0].z;
	t = fabsf( T.x );
	if( t > ra + rb ) {
		return false;
	}

	ra = a.y;
	rb = b.x*Rabs.m[1].x + b.y*Rabs.m[1].y + b.z*Rabs.m[1].z;
	t = fabsf( T.y );
	if( t > ra + rb ) {
		return false;
	}

	ra = a.z;
	rb = b.x*Rabs.m[2].x + b.y*Rabs.m[2].y + b.z*Rabs.m[2].z;
	t = fabsf( T.z );
	if( t > ra + rb ) {
		return false;
	}

	// B's basis vectors
	ra = a.x*Rabs.m[0].x + a.y*Rabs.m[1].x + a.z*Rabs.m[2].x;
	rb = b.x;
	t = fabsf( T.x*R.m[0].x + T.y*R.m[1].x + T.z*R.m[2].x );
	if( t > ra + rb ) {
		return false;
	}

	ra = a.x*Rabs.m[0].y + a.y*Rabs.m[1].y + a.z*Rabs.m[2].y;
	rb = b.y;
	t = fabsf( T.x*R.m[0].y + T.y*R.m[1].y + T.z*R.m[2].y );
	if( t > ra + rb ) {
		return false;
	}

	ra = a.x*Rabs.m[0].z + a.y*Rabs.m[1].z + a.z*Rabs.m[2].z;
	rb = b.z;
	t = fabsf( T.x*R.m[0].z + T.y*R.m[1].z + T.z*R.m[2].z );
	if( t > ra + rb ) {
		return false;
	}

	// 9 cross products
	// L = A0 x B0
	ra = a.y*Rabs.m[2].x + a.z*Rabs.m[1].x;
	rb = b.y*Rabs.m[0].z + b.z*Rabs.m[0].y;
	t = fabsf( T.z*R.m[1].x - T.y*R.m[2].x );
	if( t > ra + rb ) {
		return false;
	}

	// L = A0 x B1
	ra = a.y*Rabs.m[2].y + a.z*Rabs.m[1].y;
	rb = b.x*Rabs.m[0].z + b.z*Rabs.m[0].x;
	t = fabsf( T.z*R.m[1].y - T.y*R.m[2].y );
	if( t > ra + rb ) {
		return false;
	}

	// L = A0 x B2
	ra = a.y*Rabs.m[2].z + a.z*Rabs.m[1].z;
	rb = b.x*Rabs.m[0].y + b.y*Rabs.m[0].x;
	t = fabsf( T.z*R.m[1].z - T.y*R.m[2].z );
	if( t > ra + rb ) {
		return false;
	}

	// L = A1 x B0
	ra = a.x*Rabs.m[2].x + a.z*Rabs.m[0].x;
	rb = b.y*Rabs.m[1].z + b.z*Rabs.m[1].y;
	t = fabsf( T.x*R.m[2].x - T.z*R.m[0].x );
	if( t > ra + rb ) {
		return false;
	}

	// L = A1 x B1
	ra = a.x*Rabs.m[2].y + a.z*Rabs.m[0].y;
	rb = b.x*Rabs.m[1].z + b.z*Rabs.m[1].x;
	t = fabsf( T.x*R.m[2].y - T.z*R.m[0].y );
	if( t > ra + rb ) {
		return false;
	}

	// L = A1 x B2
	ra = a.x*Rabs.m[2].z + a.z*Rabs.m[0].z;
	rb = b.x*Rabs.m[1].y + b.y*Rabs.m[1].x;
	t = fabsf( T.x*R.m[2].z - T.z*R.m[0].z );
	if( t > ra + rb ) {
		return false;
	}

	// L = A2 x B0
	ra = a.x*Rabs.m[1].x + a.y*Rabs.m[0].x;
	rb = b.y*Rabs.m[2].z + b.z*Rabs.m[2].y;
	t = fabsf( T.y*R.m[0].x - T.x*R.m[1].x );
	if( t > ra + rb ) {
		return false;
	}

	// L = A2 x B1
	ra = a.x*Rabs.m[1].y + a.y*Rabs.m[0].y;
	rb = b.x *Rabs.m[2].z + b.z*Rabs.m[2].x;
	t = fabsf( T.y*R.m[0].y - T.x*R.m[1].y );
	if( t > ra + rb ) {
		return false;
	}

	// L = A2 x B2
	ra = a.x*Rabs.m[1].z + a.y*Rabs.m[0].z;
	rb = b.x*Rabs.m[2].y + b.y*Rabs.m[2].x;
	t = fabsf( T.y*R.m[0].z - T.x*R.m[1].z );
	if( t > ra + rb ) {
		return false;
	}

	// No separating axis found, the two boxes overlap
	return true;
}

/******************************************************************************
 * Determine if there is a collision between a BBOX and a Gigavoxel box.
 *
 * @param pVolumeTree The gigavoxel data structure
 * @param pPrecision A given precision
 * @param position The position of the BBOX
 * @param extents The size of the BBOX
 * @param basis An orthonormal basis reflecting the orientation of the BBox
 ******************************************************************************/
template< class TVolTreeKernelType >
__global__
void collision_BBOX_VolTree_Kernel (
			const TVolTreeKernelType pVolumeTree,
		    const unsigned int *precision,
	   		const float3 *position,
			const float3 *extents,
			const float4x4 *basis,
	   		float *results,
			uint arraysSize ) {
	// Thread id
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// One thread may compute several collision.
	unsigned int it = tid;

	while( it < arraysSize ) {

		results[it] = 0;

		// Stack for traversing the tree
		Node stack[32];

		// Pointer on the head of the stack
		Node *stackPtr = stack + 1;

		// Current GsNode
		GvStructure::GsNode node;

		// Root node
		Node *currentNode;
		currentNode = stackPtr;
		pVolumeTree.fetchNode( node, pVolumeTree._rootAddress );
		currentNode->_size = .5f;
		currentNode->_pos = make_float3( 0.f );
		currentNode->_nodeTileAddress = node.getChildAddress().x;
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
		// As A and B are constants, we can pre-compute it.
		float4x4 rotation, rotationAbs;
		rotation.m[0].x = dot(basisGv.m[0], basis[it].m[0]);
		rotation.m[0].y = dot(basisGv.m[0], basis[it].m[1]);
		rotation.m[0].z = dot(basisGv.m[0], basis[it].m[2]);
		rotation.m[1].x = dot(basisGv.m[1], basis[it].m[0]);
		rotation.m[1].y = dot(basisGv.m[1], basis[it].m[1]);
		rotation.m[1].z = dot(basisGv.m[1], basis[it].m[2]);
		rotation.m[2].x = dot(basisGv.m[2], basis[it].m[0]);
		rotation.m[2].y = dot(basisGv.m[2], basis[it].m[1]);
		rotation.m[2].z = dot(basisGv.m[2], basis[it].m[2]);

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


		// Test the root node
		if( !sat( positionGv, extentsGv, basisGv, position[it], extents[it], rotation, rotationAbs )) {
			return;
		}

		// BBox size (used to limit the precision).
		float sizeLimit = max( max( extents[it].x, extents[it].y ), extents[it].z ) / precision[it];
		float bboxVolume = 8 * extents[it].x * extents[it].y * extents[it].z;

		// Main loop : depth-first search for all nodes colliding with the BBox.
		do {
			// Update current node
			currentNode = stackPtr;

			Node son;
			son._size = currentNode->_size * 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );

			// TODO utiliser la fonction symétrique de toFloat1 ?
			uint x = currentNode->_nextNode & 1;
			uint y = ( currentNode->_nextNode & 2 ) >> 1;
			uint z = ( currentNode->_nextNode & 4 ) >> 2;

			// Node position
			son._pos.x = currentNode->_pos.x + x * currentNode->_size - son._size;
			son._pos.y = currentNode->_pos.y + y * currentNode->_size - son._size;
			son._pos.z = currentNode->_pos.z + z * currentNode->_size - son._size;

			// Try to retrieve the node from the node pool given its address
			pVolumeTree.fetchNode( node, currentNode->_nodeTileAddress, currentNode->_nextNode );

			son._nodeTileAddress = node.getChildAddress().x;

			if( ++currentNode->_nextNode >= 8 ) { // TODO 8 => ?
				// Unstack the current node
				--stackPtr;
			}

			if( node.isBrick() ) {
				// Non empty node, launch collision tests.
				positionGv = son._pos;
				extentsGv = make_float3( son._size );

				if( sat( positionGv, extentsGv, basisGv, position[it], extents[it], rotation, rotationAbs)) {
					// Collision
					if( son._size < sizeLimit || son._nodeTileAddress == 0 ) {
						// No more son or reach the precision limit.
						results[it] += currentNode->_size * currentNode->_size * currentNode->_size; // ( equivalent to son->size^3 * 8 )
						// TODO ? : put the node in a list of colliding node.
					} else {
						// Put the son on the stack
						++stackPtr;
						son._nextNode = 0;
						*stackPtr = son;
					}
				}
			}
		} while( stackPtr != stack );

		if( bboxVolume == 0 ) {
			results[it] = 0;
		} else {
			results[it] /= bboxVolume;
		}

		it += blockDim.x * gridDim.x;
	}

}

}; // GvCollision
