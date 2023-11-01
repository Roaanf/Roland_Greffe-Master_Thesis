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

#ifndef _COLLISION_DETECTOR_KERNEL_H_
#define _COLLISION_DETECTOR_KERNEL_H_

namespace GvCollision {

/**
 * Indicate whether or not their is a collision.
 */
__device__ bool collision;

/**
 * TODO
 *
 * @param pVolumeTree the data structure
 * @param pPoint A given position in space
 * @param pPrecision A given precision
 */
template< class TVolTreeKernelType >
__global__
void collision_Point_VolTree_Kernel( TVolTreeKernelType pVolumeTree,
		    float3 pPoint,
		    float pPrecision );

/**
 * TODO
 */
template< class TVolTreeKernelType >
__global__
void collision_BBOX_VolTree_Kernel (
			const TVolTreeKernelType pVolumeTree,
		    const unsigned int *precision,
	   		const float3 *position,
			const float3 *extents,
			const float4x4 *basis,
	   		float *results,
			uint arraysSize );

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
		);

}; // GvCollision

#include "CollisionDetectorKernel.inl"

#endif // !_COLLISION_DETECTOR_KERNEL_H_
