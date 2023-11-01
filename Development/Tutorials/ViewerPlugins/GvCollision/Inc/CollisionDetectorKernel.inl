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

/**
 * Overload fabs to use int operation (may be useful if the float pipeline is
 * full?).
 * @return the absolute value of x.
 */
//__device__ __forceinline__ float fabsf(const float &x) {
//	unsigned int i = *( unsigned int* ) &x;
//	i &= 0x7FFFFFFF; // Force sign bit to 0.
//	return *( float* ) &i;
//}

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
	float3 v = Pb - Pa;

	// Translation, in A's frame
	float3 T = make_float3 ( dot(v, make_float3( A.m[0] )),
			                 dot(v, make_float3( A.m[1] )),
			                 dot(v, make_float3( A.m[2] )));

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
	if( Rabs.m[2].x != 0.f || Rabs.m[1].x != 0.f ) {
		ra = a.y*Rabs.m[2].x + a.z*Rabs.m[1].x;
		rb = b.y*Rabs.m[0].z + b.z*Rabs.m[0].y;
		t = fabsf( T.z*R.m[1].x - T.y*R.m[2].x );
		if( t > ra + rb ) {
			return false;
		}
	}

	// L = A0 x B1
	if( Rabs.m[2].y != 0.f || Rabs.m[1].y != 0.f ) {
		ra = a.y*Rabs.m[2].y + a.z*Rabs.m[1].y;
		rb = b.x*Rabs.m[0].z + b.z*Rabs.m[0].x;
		t = fabsf( T.z*R.m[1].y - T.y*R.m[2].y );
		if( t > ra + rb ) {
			return false;
		}
	}

	// L = A0 x B2
	if( Rabs.m[2].z != 0.f || Rabs.m[1].z != 0.f ) {
		ra = a.y*Rabs.m[2].z + a.z*Rabs.m[1].z;
		rb = b.x*Rabs.m[0].y + b.y*Rabs.m[0].x;
		t = fabsf( T.z*R.m[1].z - T.y*R.m[2].z );
		if( t > ra + rb ) {
			return false;
		}
	}

	// L = A1 x B0
	if( Rabs.m[2].x != 0.f || Rabs.m[0].x != 0.f ) {
		ra = a.x*Rabs.m[2].x + a.z*Rabs.m[0].x;
		rb = b.y*Rabs.m[1].z + b.z*Rabs.m[1].y;
		t = fabsf( T.x*R.m[2].x - T.z*R.m[0].x );
		if( t > ra + rb ) {
			return false;
		}
	}

	// L = A1 x B1
	if( Rabs.m[2].y != 0.f || Rabs.m[0].y != 0.f ) {
		ra = a.x*Rabs.m[2].y + a.z*Rabs.m[0].y;
		rb = b.x*Rabs.m[1].z + b.z*Rabs.m[1].x;
		t = fabsf( T.x*R.m[2].y - T.z*R.m[0].y );
		if( t > ra + rb ) {
			return false;
		}
	}

	// L = A1 x B2
	if( Rabs.m[2].z != 0.f || Rabs.m[0].z != 0.f ) {
		ra = a.x*Rabs.m[2].z + a.z*Rabs.m[0].z;
		rb = b.x*Rabs.m[1].y + b.y*Rabs.m[1].x;
		t = fabsf( T.x*R.m[2].z - T.z*R.m[0].z );
		if( t > ra + rb ) {
			return false;
		}
	}

	// L = A2 x B0
	if( Rabs.m[1].x != 0.f || Rabs.m[0].x != 0.f ) {
		ra = a.x*Rabs.m[1].x + a.y*Rabs.m[0].x;
		rb = b.y*Rabs.m[2].z + b.z*Rabs.m[2].y;
		t = fabsf( T.y*R.m[0].x - T.x*R.m[1].x );
		if( t > ra + rb ) {
			return false;
		}
	}

	// L = A2 x B1
	if( Rabs.m[1].y != 0.f || Rabs.m[0].y != 0.f ) {
		ra = a.x*Rabs.m[1].y + a.y*Rabs.m[0].y;
		rb = b.x *Rabs.m[2].z + b.z*Rabs.m[2].x;
		t = fabsf( T.y*R.m[0].y - T.x*R.m[1].y );
		if( t > ra + rb ) {
			return false;
		}
	}

	// L = A2 x B2
	if( Rabs.m[1].z != 0.f || Rabs.m[0].z != 0.f ) {
		ra = a.x*Rabs.m[1].z + a.y*Rabs.m[0].z;
		rb = b.x*Rabs.m[2].y + b.y*Rabs.m[2].x;
		t = fabsf( T.y*R.m[0].z - T.x*R.m[1].z );
		if( t > ra + rb ) {
			return false;
		}
	}

	// No separating axis found, the two boxes overlap
	return true;
}


}; // GvCollision
