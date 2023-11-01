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

#ifndef _VOLUME_TREE_RENDERER_CUDA_KERNEL_H_
#define _VOLUME_TREE_RENDERER_CUDA_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <helper_math.h>
#include <texture_types.h>

// GigaVoxels
#include <GvCore/GsPool.h>
#include <GvCore/GsRendererTypes.h>
#include <GvRendering/GsRendererHelpersKernel.h>
#include <GvRendering/GsSamplerKernel.h>
#include <GvRendering/GsRendererContext.h>
#include <GvPerfMon/GsPerformanceMonitor.h>
#include <GvStructure/GsNode.h>
//#include <GvPerfMon/GsPerformanceMonitor.cu>

// Project
#include "VolumeTreeSamplingKernel.h"
#include "VolumeTreeTraversalKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

//namespace GsRenderer
//{ 
//
//	/**
//	 * Model matrix (CUDA constant memory)
//	 */
//	__constant__ float4x4 k_modelMatrix;
//
//	/**
//	 * Inverse model matrix (CUDA constant memory)
//	 */
//	__constant__ float4x4 k_modelMatrixInv;
//
//}

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Initialize the pixel coordinates.
 *
 * @param Pid the input thread identifiant
 * @param blockPos the computed block position
 * @param pixelCoords the computed pixel coordinates
 ******************************************************************************/
template< class TBlockResolution >
__device__
inline void initPixelCoords( const uint Pid, /*uint2& blockPos,*/ uint2& pixelCoords )
{
#if RENDER_USE_SWIZZLED_THREADS==0
	pixelCoords.x = threadIdx.x + __uimul( blockIdx.x, TBlockResolution::x );
	pixelCoords.y = threadIdx.y + __uimul( blockIdx.y, TBlockResolution::y );
#else // Z-curve scheduling

	// Deinterleave bits
	GvRendering::deinterleaveBits32( Pid, pixelCoords );

// As "blockPos" parameter seems to be unused anymore, the following are commented.
// TO DO : are following lines useful ?
//#if 1
//	blockPos = make_uint2( blockIdx.x, blockIdx.y );
//#else
//	uint Bid = blockIdx.x + blockIdx.y * gridDim.x;
//	deinterleaveBits32( Bid, blockPos );
//#endif

	pixelCoords.x += /*blockPos.x*/blockIdx.x * TBlockResolution::x;
	pixelCoords.y += /*blockPos.y*/blockIdx.y * TBlockResolution::y;
#endif
}

__device__ void multMatrix(float * mat,float3* xx) {
	float3 yy;
	yy.x=xx->x- 0.5f;
	yy.y=xx->y- 0.5f;
	yy.z=xx->z- 0.5f;
	float3 zz;
	zz.x = mat[0]*(yy.x)+mat[1]*(yy.y)+mat[2]*(yy.z);
	zz.y = mat[3]*(yy.x)+mat[4]*(yy.y)+mat[5]*(yy.z);
	zz.z = mat[6]*(yy.x)+mat[7]*(yy.y)+mat[8]*(yy.z);

	xx->x = zz.x+0.5f;
	xx->y = zz.y+0.5f;
	xx->z = zz.z+0.5f;
}
__device__ void multMatrix2(float * mat,float3* xx) {

	float3 zz;
	zz.x = mat[0]*(xx->x)+mat[1]*(xx->y)+mat[2]*(xx->z);
	zz.y = mat[3]*(xx->x)+mat[4]*(xx->y)+mat[5]*(xx->z);
	zz.z = mat[6]*(xx->x)+mat[7]*(xx->y)+mat[8]*(xx->z);

	xx->x = zz.x;
	xx->y = zz.y;
	xx->z = zz.z;

}


/******************************************************************************
 * This function is used to :
 * - traverse the data structure (and emit requests if necessary)
 * - render bricks
 *
 * @param pDataStructure data structure
 * @param pShader shader
 * @param pCache cahce
 * @param pPixelCoords pixel coordinates in window
 * @param pRayStartTree ray start point
 * @param pRayDirTree ray direction
 * @param ptMaxTree max distance from camera found after box intersection test and comparing with input z (from the scene)
 * @param ptTree the distance from camera found after box intersection test and comparing with input z (from the scene)
 ******************************************************************************/
template< bool TFastUpdateMode, bool TPriorityOnBrick, class VolTreeKernelType, class SampleShaderType, class TCacheType >
__device__
void renderVolTree_Std( VolTreeKernelType& pDataStructure, SampleShaderType& pShader, TCacheType& pCache,
					   uint2 pPixelCoords, const float3 pRayStartTree, const float3 pRayDirTree, const float ptMaxTree, float& ptTree )
{
	
	CUDAPM_KERNEL_DEFINE_EVENT( 2 );
	CUDAPM_KERNEL_DEFINE_EVENT( 3 );
	CUDAPM_KERNEL_DEFINE_EVENT( 4 );
	CUDAPM_KERNEL_DEFINE_EVENT( 5 );

	CUDAPM_KERNEL_START_EVENT( pPixelCoords, 5 );
	float3 rayDirTree = pRayDirTree;
	float3 rayStartTree = pRayStartTree;
	float epsilon = 1e-3;

	// Keep root node in cache
	pCache._nodeCacheManager.setElementUsage( 0 );
	
	// Initialize the brick sampler, a helper class used to sample data in the data structure
	GvRendering::GsSamplerKernel< VolTreeKernelType > brickSampler;
	brickSampler._volumeTree = &pDataStructure;

	// Initialize the position at wich data will be sampled
	float3 samplePosTree = pRayStartTree + ptTree * pRayDirTree;
	
 float Id[9] = {1,0,0,0,1,0,0,0,1}; //inv Id
 float R0[9] = {0,-1,0,1,0,0,0,0,1}; // inv R1
 float R1[9] = {0,1,0,-1,0,0,0,0,1}; // inv R0
 float R2[9] = {1,0,0,0,0,-1,0,1,0};// inv R3
 float R3[9] = {1,0,0,0,0,1,0,-1,0}; // inv R2
 float R4[9] = {0,0,1,0,1,0,-1,0,0};// inv R5
 float R5[9] = {0,0,-1,0,1,0,1,0,0}; // inv R4
 float R6[9] = {0,0,1,1,0,0,0,1,0}; // inv R13
 float R7[9] = {0,0,-1,-1,0,0,0,1,0}; // inv R12
 float R8[9] = {0,-1,0,0,0,-1,1,0,0}; // inv R10
 float R9[9] = {0,1,0,0,0,-1,-1,0,0}; // inv R11
 float R10[9] = {0,0,1,-1,0,0,0,-1,0}; // inv R8
 float R11[9] = {0,0,-1,1,0,0,0,-1,0}; // inv R9
 float R12[9] = {0,-1,0,0,0,1,-1,0,0}; // inv R7
 float R13[9] = {0,1,0,0,0,1,1,0,0};  // inv R6
 float R14[9] = {1,0,0,0,-1,0,0,0,-1}; // inv R14
 float R15[9] = {-1,0,0,0,1,0,0,0,-1}; // inv R15
 float R16[9] = {-1,0,0,0,-1,0,0,0,1}; // inv R16
 float R17[9] = {0,0,1,0,-1,0,1,0,0}; // inv R17
 float R18[9] = {0,0,-1,0,-1,0,-1,0,0}; // inv R18
 float R19[9] = {0,-1,0,-1,0,0,0,0,-1};  // inv R19
 float R20[9] = {0,1,0,1,0,0,0,0,-1}; // inv R20
 float R21[9] = {-1,0,0,0,0,-1,0,-1,0};  // inv R21
 float R22[9] = {-1,0,0,0,0,1,0,1,0}; // inv R22

	// Shader pre-shade process
	pShader.preShade( pRayStartTree, pRayDirTree, ptTree );
	float offsetPtree = 0.f;
	float3 brickIndex = cNbCameraReflections;
	
	//printf ("%f,%f,%f\n",brickIndex.x,brickIndex.y,brickIndex.z);
	//float3 brickOffset = make_float3(0.f,0.f,0.f);

	bool reflection = false; 
	unsigned int reflection_total = 0;
	//float * matrix = Id;
	float * antiMatrix = Id;
	float * matrix  = Id;
	switch (  abs(( (int)((brickIndex.x-0.5f)*2.f + (brickIndex.y-0.5f)*3.f + (brickIndex.z-0.5f)*5.f  )) % 24))
		{
			case 0 : antiMatrix = Id ; break;
			case 1 : antiMatrix = R1 ; break;
			case 2 : antiMatrix = R0 ; break;
			case 3 : antiMatrix = R3 ; break;
			case 4 : antiMatrix = R2 ; break;
			case 5 : antiMatrix = R5 ; break;
			case 6 : antiMatrix = R4 ; break;
			case 7 : antiMatrix = R13; break;
			case 8 : antiMatrix = R12; break;
			case 9 : antiMatrix = R10; break;
			case 10 : antiMatrix = R11; break;
			case 11 : antiMatrix = R8; break;
			case 12 : antiMatrix = R9; break;
			case 13 : antiMatrix = R7; break;
			case 14 : antiMatrix = R6; break;
			case 15 : antiMatrix = R14;break;
			case 16 : antiMatrix = R15;break;
			case 17 : antiMatrix = R16;break;
			case 18 : antiMatrix = R17;break;
			case 19 : antiMatrix = R18;break;
			case 20 : antiMatrix = R19;break;
			case 21 : antiMatrix = R20;break;
			case 22 : antiMatrix = R21;break;
			case 23 : antiMatrix = R22;break;
					
			default :              break;
		}
	
	// Ray marching.
	// Step with "ptTree" along ray from start to stop bounds.
	int numLoop = 0;
	while
		( //ptTree < 2000.f*sqrtf(3.f)&& 
		numLoop < 5000	// TO DO : remove this hard-coded value or let only for DEBUG
		&& !pShader.stopCriterion( samplePosTree ) )
	{
		CUDAPM_KERNEL_START_EVENT( pPixelCoords, 4 );

		//float3 samplePosTree = pRayStartTree + ptTree * pRayDirTree;
		const float coneAperture = pShader.getConeAperture( ptTree );
		
		// Declare an empty node of the data structure.
		// It will be filled during the traversal according to cuurent sample position "samplePosTree"
		GvStructure::GsNode node;

		CUDAPM_KERNEL_START_EVENT( pPixelCoords, 2 );

		// [ 1 ]- Descent the data structure (in general an octree)
		// until max depth is reach or current traversed node has no subnodes,
		// or cone aperture is greater than voxel size.
		float nodeSizeTree;
		float3 sampleOffsetInNodeTree;
		
		bool modifInfoWriten = false;
		rendererDescentOctree< TPriorityOnBrick >
							( pDataStructure, pCache, node, samplePosTree, coneAperture,
                            nodeSizeTree, sampleOffsetInNodeTree, brickSampler, modifInfoWriten ,pShader);
		
		const float rayLengthInNodeTree = GvRendering::getRayLengthInNode( sampleOffsetInNodeTree, nodeSizeTree, rayDirTree );

		CUDAPM_KERNEL_STOP_EVENT( pPixelCoords, 2 );

		// Early loop termination
		if ( TFastUpdateMode && modifInfoWriten )
		{
			break;
		}

		// [ 2 ] - If node is a brick, renderer it.
		//if ( node.isBrick() )
		if ( node.hasBrick() )
		{
			CUDAPM_KERNEL_START_EVENT( pPixelCoords, 3 );

			// PASCAL
			// This is used to stop the ray with a z-depth value smaller than the remaining brick ray length
			//
			// QUESTION : pas forc�ment, si objet qui cache est transparent !??
			// => non, comme d'hab en OpenGL => dessiner d'abord les objets opaques
			const float rayLengthInBrick =rayLengthInNodeTree;// mincc( rayLengthInNodeTree, ptMaxTree - (ptTree-offsetPtree) );	// ==> attention, ( ptMaxTree - ptTree < 0 ) ?? ==> non, � casue du test du WHILE !! c'est OK !!

			// This is where shader program occurs
			float dt = ::rendererBrickSampling< TFastUpdateMode, TPriorityOnBrick >
									( pDataStructure, pShader, pCache, rayStartTree, rayDirTree,
									ptTree,offsetPtree, rayLengthInBrick, brickSampler, modifInfoWriten );

			CUDAPM_KERNEL_STOP_EVENT( pPixelCoords, 3 );

			//ptTree += dt;
			ptTree += rayLengthInBrick;
			ptTree += pShader.getConeAperture( ptTree );

            // brique pleine
		}
		else
		{
			
			ptTree += rayLengthInNodeTree;// + coneAperture;
			ptTree += pShader.getConeAperture( ptTree );

            // brique vide
		}

		samplePosTree = rayStartTree + (ptTree-offsetPtree) * rayDirTree;
		
		
		if (reflection_total<cNbMirrorReflections) 
		{
			float epsilon = 1e-5;
	
			if (samplePosTree.x >= 1.f )
			{
				// Reflexion
				//samplePosTree.x  = 1.f;//- (samplePosTree.x-1.f);
				//rayDirTree.x = (-rayDirTree.x);
				//pShader._renderViewContext.x = 2.f-pShader._renderViewContext.x;
				
				// Modulo 

				samplePosTree.x  = 0.f+epsilon;
				pShader._renderViewContext.x -= 1.f;
				
				float3 centre_face = make_float3(1.f,0.5f,0.5f);
				multMatrix(antiMatrix,&centre_face);

				brickIndex += make_float3((centre_face.x - 0.5f)*2.f , (centre_face.y - 0.5f)*2.f,(centre_face.z - 0.5f)*2.f );
				
				reflection =true;
				
				
				//printf("%f,%f\n",(ptTree-offsetPtree),ptMaxTree);
			}
			if (samplePosTree.x <= 0.f)
			{
				// Reflexion
				//samplePosTree.x  = 0.f;//- (samplePosTree.x-0.f);
				//rayDirTree.x = (-rayDirTree.x);
				//pShader._renderViewContext.x = 0.f-pShader._renderViewContext.x;

				// Modulo 
				samplePosTree.x  = 1.f-epsilon;
				pShader._renderViewContext.x += 1.f;
				float3 centre_face = make_float3(0.f,0.5f,0.5f);
				multMatrix(antiMatrix,&centre_face);

				brickIndex += make_float3((centre_face.x - 0.5f)*2.f , (centre_face.y - 0.5f)*2.f,(centre_face.z - 0.5f)*2.f );
				
				reflection =true;
				
				//printf("%f,%f\n",(ptTree-offsetPtree),ptMaxTree);
			}
			if (samplePosTree.y >= 1.f)
			{
				// Reflexion
				//samplePosTree.y  = 1.f;//- (samplePosTree.y-1.f);
				//rayDirTree.y = (-rayDirTree.y);
				//pShader._renderViewContext.y = 2.f-pShader._renderViewContext.y;

				// Modulo 
				samplePosTree.y  = 0.f+epsilon;
				pShader._renderViewContext.y -= 1.f;
				float3 centre_face = make_float3(0.5f,1.f,0.5f);
				multMatrix(antiMatrix,&centre_face);

				brickIndex += make_float3((centre_face.x - 0.5f)*2.f , (centre_face.y - 0.5f)*2.f,(centre_face.z - 0.5f)*2.f );
				
				reflection =true;
				
				//printf("%f,%f\n",(ptTree-offsetPtree),ptMaxTree);
			}
			if (samplePosTree.y <= 0.f)
			{
				// Reflexion
				//samplePosTree.y  = 0.f;//- (samplePosTree.y-0.f);
				//rayDirTree.y = (-rayDirTree.y);
				//pShader._renderViewContext.y = 0.f-pShader._renderViewContext.y;

				// Modulo 
				samplePosTree.y  = 1.f-epsilon;
				pShader._renderViewContext.y += 1.f;
				float3 centre_face = make_float3(0.5f,0.f,0.5f);
				multMatrix(antiMatrix,&centre_face);

				brickIndex += make_float3((centre_face.x - 0.5f)*2.f , (centre_face.y - 0.5f)*2.f,(centre_face.z - 0.5f)*2.f );
				
				reflection =true;
				
				//printf("%f,%f\n",(ptTree-offsetPtree),ptMaxTree);
			}
			if (samplePosTree.z >= 1.f)
			{
				// Reflexion
				//samplePosTree.z  = 1.f;//- (samplePosTree.z-1.f);
				//rayDirTree.z = (-rayDirTree.z);
				//pShader._renderViewContext.z = 2.f-pShader._renderViewContext.z;

				// Modulo 
				samplePosTree.z  = 0.f+epsilon;
				pShader._renderViewContext.z -= 1.f;
				float3 centre_face = make_float3(0.5f,0.5f,1.f);
				multMatrix(antiMatrix,&centre_face);

				brickIndex += make_float3((centre_face.x - 0.5f)*2.f , (centre_face.y - 0.5f)*2.f,(centre_face.z - 0.5f)*2.f );
				
				reflection =true;
				
				//printf("%f,%f\n",(ptTree-offsetPtree),ptMaxTree);
			}
			if (samplePosTree.z <= 0.f)
			{
				// Reflexion
				//samplePosTree.z  = 0.f;//- (samplePosTree.z-0.f);
				//rayDirTree.z = (-rayDirTree.z);
				//pShader._renderViewContext.z = 0.f-pShader._renderViewContext.z;

				// Modulo 
				samplePosTree.z  = 1.f-epsilon;
				pShader._renderViewContext.z += 1.f;
				float3 centre_face = make_float3(0.5f,0.5f,0.f);
				multMatrix(antiMatrix,&centre_face);

				brickIndex += make_float3((centre_face.x - 0.5f)*2.f , (centre_face.y - 0.5f)*2.f,(centre_face.z - 0.5f)*2.f );
				
				reflection =true;
				
				//printf("%f,%f\n",(ptTree-offsetPtree),ptMaxTree);
			}
			if (reflection ) 
			{
				float3 camera = pShader._renderViewContext;
				multMatrix2(antiMatrix,&rayDirTree);		
				multMatrix(antiMatrix,&samplePosTree);
				multMatrix(antiMatrix,&(camera));

				
				switch ( abs(( (int)((brickIndex.x-0.5f)*2.f + (brickIndex.y-0.5f)*3.f + (brickIndex.z-0.5f)*5.f   )) % 24))
				{
					case 0 : matrix = Id; antiMatrix = Id ; break;
					case 1 : matrix = R0; antiMatrix = R1 ;  break;
					case 2 : matrix = R1; antiMatrix = R0 ; break;
					case 3 : matrix = R2; antiMatrix = R3 ; break;
					case 4 : matrix = R3; antiMatrix = R2 ; break;
					case 5 : matrix = R4; antiMatrix = R5 ; break;
					case 6 : matrix = R5; antiMatrix = R4 ; break;
					case 7 : matrix = R6; antiMatrix = R13 ; break;
					case 8 : matrix = R7; antiMatrix = R12 ; break;
					case 9 : matrix = R8; antiMatrix = R10 ; break;
					case 10 : matrix = R9; antiMatrix = R11 ; break;
					case 11 : matrix = R10; antiMatrix = R8 ; break;
					case 12 : matrix = R11; antiMatrix = R9 ; break;
					case 13 : matrix = R12; antiMatrix = R7 ; break;
					case 14 : matrix = R13; antiMatrix = R6 ; break;
					case 15 : matrix = R14; antiMatrix = R14 ; break;
					case 16 : matrix = R15; antiMatrix = R15 ; break;
					case 17 : matrix = R16; antiMatrix = R16 ; break;
					case 18 : matrix = R17; antiMatrix = R17 ; break;
					case 19 : matrix = R18; antiMatrix = R18 ; break;
					case 20 : matrix = R19; antiMatrix = R19 ; break;
					case 21 : matrix = R20; antiMatrix = R20 ; break;
					case 22 : matrix = R21; antiMatrix = R21 ; break;
					case 23 : matrix = R22; antiMatrix = R22 ; break;
					
					
					default :              break;
				}
				multMatrix2(matrix,&rayDirTree);		
				multMatrix(matrix,&samplePosTree);
				multMatrix(matrix,&(camera));

				pShader._renderViewContext = camera;
				rayStartTree = samplePosTree;
				
				offsetPtree = ptTree;
				pShader._distanceBeforeReflection = offsetPtree;
				//printf("out : %f,%f,%f\n",rayStartTree.x,rayStartTree.y,rayStartTree.z);
				reflection_total ++;
				reflection= false;
			}
		} else { 
			if (samplePosTree.x >= 1.f || samplePosTree.x <= 0.f || samplePosTree.y >= 1.f || samplePosTree.y <= 0.f || samplePosTree.z >= 1.f || samplePosTree.z <= 0.f)
			{
				// Shader post-shade process
				pShader.postShade();
				return;
			}
		}
	
		
		
			// Update internal counter
		numLoop++;

		CUDAPM_KERNEL_STOP_EVENT( pPixelCoords, 4 );
	} // while

	CUDAPM_KERNEL_STOP_EVENT( pPixelCoords, 5 );

	// Shader post-shade process
	pShader.postShade();
	
}

/******************************************************************************
 * CUDA kernel
 * This is the main GigaVoxels KERNEL
 * It is in charge of casting rays and found the color and depth values at pixels.
 *
 * @param pVolumeTree data structure
 * @param pCache cache
 ******************************************************************************/
template<	class TBlockResolution, bool TFastUpdateMode, bool TPriorityOnBrick, 
			class TSampleShaderType, class TVolTreeKernelType, class TCacheType >
__global__
void RenderKernelSimple( TVolTreeKernelType pVolumeTree, TCacheType pCache )
{
	// Per-pixel shader instance
	typename TSampleShaderType::KernelType sampleShader;

	// Shared memory
	//__shared__ float3 rayStartInWorld;
	__shared__ float3 rayStartInTree;

	CUDAPM_KERNEL_DEFINE_EVENT( 0 );
	CUDAPM_KERNEL_DEFINE_EVENT( 1 );

	// Compute thread ID
	uint Pid = threadIdx.x + threadIdx.y * TBlockResolution::x;

	// Retrieve current processed pixel position
	// This function modifies the pixel accessing pattern (i.e. z-curve)
	uint2 pixelCoords;
	/*uint2 blockPos;*/ // NOTE : this "block position" parameter seemed not used anymore
	initPixelCoords< TBlockResolution >( Pid, /*blockPos,*/ pixelCoords );

	CUDAPM_KERNEL_START_EVENT( pixelCoords, 0 );

	// Check if were are inside the frame (window or viewport ?)
	bool outOfFrame = ( pixelCoords.x >= k_renderViewContext.frameSize.x ) || ( pixelCoords.y >= k_renderViewContext.frameSize.y );
	// FUTUR optimization
	//
	//bool outOfFrame = ( ( pixelCoords.x >= /*projectedBBoxSize*/k_renderViewContext._projectedBBox.z ) || ( pixelCoords.y >= /*projectedBBoxSize*/k_renderViewContext._projectedBBox.w ) );
	//bool outOfFrame = ( ( pixelCoords.x > /*projectedBBoxSize*/k_renderViewContext._projectedBBox.z ) || ( pixelCoords.y > /*projectedBBoxSize*/k_renderViewContext._projectedBBox.w ) );
	if ( ! outOfFrame )
	//bool inFrame = ( ( pixelCoords.x < k_renderViewContext._projectedBBox.z ) || ( pixelCoords.y < k_renderViewContext._projectedBBox.w ) );
	//if ( inFrame )
	{
		// Read depth from the input depth buffer.
		// Depth buffer contains the Zwindow (distance to camera plane) which is different from Zeye (distance to camera)
		// Zwindow is between 0.0 and 1.0
		// The depth buffer doesn't contain distance values from the camera.
		// The depth values are the perpendicular distance to the plane of the camera.
		float frameDepth = GvRendering::getInputDepth( pixelCoords );		// TO DO : this read memory could be placed before to avoid cycles waiting...

		// FUTUR optimization
		//
		//// Add offset of the projected BBox bottom left corner
		//pixelCoords.x += /*projectedBBoxBottomLeft*/k_renderViewContext._projectedBBox.x;
		//pixelCoords.y += /*projectedBBoxBottomLeft*/k_renderViewContext._projectedBBox.y;

		//// Calculate eye ray in world space

		//float3 pixelVecWP = k_renderViewContext.viewPlaneDirWP
		//					+ k_renderViewContext.viewPlaneXAxisWP * static_cast< float >( pixelCoords.x )
		//					+ k_renderViewContext.viewPlaneYAxisWP * static_cast< float >( pixelCoords.y );

		//rayStartInWorld = k_renderViewContext.viewCenterWP;
		//float3 rayDirInWorld = normalize( pixelVecWP );

		//// Transform the ray from World to Tree Space
		//rayStartInTree = mul( k_renderViewContext.invModelMatrix, rayStartInWorld );	// ce terme pourrait/devrait �tre calcul� sur le HOST car il est constant...
		//
		//// Beware, vectors and transformed by inverse transpose...
		//// TO DO : do correction
		//float3 rayDirInTree = normalize( mulRot( k_renderViewContext.invModelMatrix, rayDirInWorld ) );

		//---------------------------------------
		// TEST
		// Calculate eye ray in tree space
		float3 rayDirInTree = k_renderViewContext.viewPlaneDirTP
							+ k_renderViewContext.viewPlaneXAxisTP * static_cast< float >( pixelCoords.x )
							+ k_renderViewContext.viewPlaneYAxisTP * static_cast< float >( pixelCoords.y );
		/*float3*/ rayStartInTree = k_renderViewContext.viewCenterTP;
		// + PASCAL
		rayDirInTree = normalize( rayDirInTree );
		//---------------------------------------
			
		float boxInterMin = 0.0f;
		float boxInterMax = 10000.0f;
		int hit = GvRendering::intersectBox( rayStartInTree, rayDirInTree, make_float3( 0.001f ), make_float3( 0.999f ), boxInterMin, boxInterMax );
		bool masked = ! ( hit && ( boxInterMax > 0.0f ) );
		boxInterMin = maxcc( boxInterMin, k_renderViewContext.frustumNear );

		float t = boxInterMin + sampleShader.getConeAperture( boxInterMin );
		float tMax = boxInterMax;

		if ( frameDepth < 1.0f )
		{
			// Retrieve the view-space depth from the depth buffer. Only works if w was 1.0f.
			// See: http://www.opengl.org/discussion_boards/ubbthreads.php?ubb=showflat&Number=304624&page=2
			float clipZ = 2.0f * frameDepth - 1.0f;
			float frameT = k_renderViewContext.frustumD / ( -clipZ - k_renderViewContext.frustumC );
			frameT = -frameT;

			tMax = mincc( frameT, boxInterMax );
			//tMax = boxInterMax;
		}

		/*if ( t == 0.0f || t >= tMax )
		{
			masked = true;
		}*/

		// In this demo, we are always inside the box
		masked = false;

		if ( ! masked )
		{
			// Read color from the input color buffer
            uchar4 frameColor = GvRendering::getInputColor( pixelCoords );

			// Launch N3-tree traversal and rendering
			CUDAPM_KERNEL_START_EVENT( pixelCoords, 1 );
			renderVolTree_Std< TFastUpdateMode, TPriorityOnBrick >( pVolumeTree, sampleShader, pCache, pixelCoords, rayStartInTree, rayDirInTree, tMax, t );
			CUDAPM_KERNEL_STOP_EVENT( pixelCoords, 1 );

			// Retrieve the accumulated color
			float4 accCol = sampleShader.getColor();

			// Convert color from uchar [ 0 ; 255 ] to float [ 0.0 ; 1.0 ]
			float4 scenePixelColorF = make_float4( (float)frameColor.x / 255.0f, (float)frameColor.y / 255.0f, (float)frameColor.z / 255.0f, (float)frameColor.w / 255.0f );
			
			// Update color
			float4 pixelColorF = accCol + scenePixelColorF * ( 1.0f - accCol.w );

			// Clamp color to be within the interval [+0.0, 1.0]
			pixelColorF.x = __saturatef( pixelColorF.x );
			pixelColorF.y = __saturatef( pixelColorF.y );
			pixelColorF.z = __saturatef( pixelColorF.z );
			pixelColorF.w = 1.0f;		// <== why 1.0f and not __saturatef( pixelColorF.w ) ?	// Pour �viter une op�ration OpenGL de ROP ? Car �a a �t� penser pour rendre en dernier au d�part ?
			//pixelColorF.w = __saturatef( pixelColorF.w );
			
			// Convert color from float [ 0.0 ; 1.0 ] to uchar [ 0 ; 255 ]
			frameColor = make_uchar4( (uchar)( pixelColorF.x * 255.0f ), (uchar)( pixelColorF.y * 255.0f ), (uchar)( pixelColorF.z * 255.0f ), (uchar)( pixelColorF.w * 255.0f ) );
			
			// Project the depth and check against the current one
			float pixDepth = 1.0f;

			if ( accCol.w > cOpacityStep )
			{
				float VP = -fabsf( t * rayDirInTree.z );
				//http://www.opengl.org/discussion_boards/ubbthreads.php?ubb=showflat&Number=234519&page=2
				float clipZ = ( VP * k_renderViewContext.frustumC + k_renderViewContext.frustumD ) / -VP;
				
				//pixDepth = clamp( ( clipZ + 1.0f ) / 2.0f, 0.0f, 1.0f );		// TO DO : use __saturatef instead !!
				pixDepth = __saturatef( ( clipZ + 1.0f ) / 2.0f );		// TO DO : use __saturatef instead !!	=====> ( [ x 0.5f ] instead ) ??
			}

			//frameDepth = getFrameDepthIn( pixelCoords );
			frameDepth = min( frameDepth, pixDepth );

			// Write color in color buffer
			GvRendering::setOutputColor( pixelCoords, frameColor );
			
			// Write depth in depth buffer
			GvRendering::setOutputDepth( pixelCoords, frameDepth );
		}
	} // !outOfFrame

	CUDAPM_KERNEL_STOP_EVENT( pixelCoords, 0 );
}

// FIXME: Move this to another place
/******************************************************************************
 * CUDA kernel ...
 *
 * @param syntheticBuffer ...
 * @param totalNumElems ...
 ******************************************************************************/
__global__
void SyntheticInfo_Render( uchar4 *syntheticBuffer, uint totalNumElems )
{
	uint2 pixelCoords;
	pixelCoords.x = threadIdx.x + __uimul(blockIdx.x, blockDim.x);
	pixelCoords.y = threadIdx.y + __uimul(blockIdx.y, blockDim.y);

	uint elemIdx= pixelCoords.x+pixelCoords.y* (blockDim.x*gridDim.x);

	//uint totalNumElems=syntheticBuffer.getResolution().x;

	uchar4 pixelColor;

	if ( elemIdx < totalNumElems )
	{
		uchar4 sval = syntheticBuffer[ elemIdx ];

		if ( sval.w )
		{
			pixelColor = make_uchar4( 255, 0, 0, 255 );
		}
		else if ( sval.x )
		{
			pixelColor = make_uchar4( 0, 255, 0, 255 );
		}
		else
		{
			pixelColor = make_uchar4( 0, 0, 0, 255 );
		}

		GvRendering::setOutputColor( pixelCoords, pixelColor );
	}
	else
	{
		//pixelColor = make_uchar4( 255, 255, 255, 255 );
	}
}

#endif // !_VOLUME_TREE_RENDERER_CUDA_KERNEL_H_
