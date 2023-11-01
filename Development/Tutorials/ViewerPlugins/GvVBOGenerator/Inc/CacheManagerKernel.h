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

//
///** 
// * @version 1.0
// */
//
//#ifndef _CACHE_MANAGER_KERNEL_H_
//#define _CACHE_MANAGER_KERNEL_H_
//
///******************************************************************************
// ******************************* INCLUDE SECTION ******************************
// ******************************************************************************/
//
//// Cuda
//#include <vector_types.h>
//
//// GigaVoxels
//#include <GvCache/GsCacheManagerKernel.h>
//
///******************************************************************************
// ************************* DEFINE AND CONSTANT SECTION ************************
// ******************************************************************************/
//
///******************************************************************************
// ***************************** TYPE DEFINITION ********************************
// ******************************************************************************/
//
///******************************************************************************
// ******************************** CLASS USED **********************************
// ******************************************************************************/
//
///******************************************************************************
// ****************************** CLASS DEFINITION ******************************
// ******************************************************************************/
//
///******************************************************************************
// ***************************** KERNEL DEFINITION ******************************
// ******************************************************************************/
//
///******************************************************************************
// * GvKernel_ReadVboNbPoints kernel
// *
// * This kernel retrieve number of points contained in each used bricks
// *
// * @param pNbPointsList [out] list of points inside each brick
// * @param pNbBricks number of bricks to process
// * @param pBrickAddressList list of brick addresses in cache (used to retrive positions where to fetch data)
// * @param pDataStructure data structure in cache where to fecth data
// ******************************************************************************/
//template< class TDataStructureKernelType >
//__global__
//void GvKernel_ReadVboNbPoints( uint* pNbPointsList, const uint pNbBricks, const uint* pBrickAddressList, TDataStructureKernelType pDataStructure )
//{
//	// Retrieve global data index
//	uint lineSize = __uimul( blockDim.x, gridDim.x );
//	uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );
//
//	// Check bounds
//	if ( elem < pNbBricks )
//	{
//		//	printf( "\nGvKernel_ReadVboNbPoints : ELEM %d : dataPoolPosition = [ %d ]", elem, pBrickAddressList[ elem ] );
//
//		// Retrieve the number of points in the current brick
//		//
//		// Brick position in cache (without border)
//		//float3 dataPoolPosition = make_float3( GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ elem ] ) ) * pDataStructure.brickCacheResINV;
//		//float3 dataPoolPosition = make_float3( 20, 0, 0 ) * pDataStructure.brickCacheResINV;
//		//uint3 unpackedBrickAddress = GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ elem ] );
//		//uint3 unpackedBrickAddress = 10 * GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ elem ] );
//
//		// The adress pBrickAddressList[ elem ] is the adress of a brick in the elemAdressList 1D linearized array cache.
//		// The adress is the adress of the brick beginning with the border (there is no one border ofset)
//		uint3 unpackedBrickAddress = ( TDataStructureKernelType::BrickResolution::get() + 2 * TDataStructureKernelType::brickBorderSize ) * GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ elem ] );
//		//	printf( "\nunpackedBrickAddress = [ %d | %d | %d ]", unpackedBrickAddress.x, unpackedBrickAddress.y, unpackedBrickAddress.z );
//
//		float3 dataPoolPosition = make_float3( unpackedBrickAddress.x, unpackedBrickAddress.y, unpackedBrickAddress.z ) * pDataStructure.brickCacheResINV;
//		//	printf( "\nGvKernel_ReadVboNbPoints : dataPoolPosition = [ %f | %f | %f ]", dataPoolPosition.x, dataPoolPosition.y, dataPoolPosition.z );
//		// Shift from one voxel to be on the brick border
//		//dataPoolPosition -= pDataStructure.brickCacheResINV;
//		//const float4 nbPoints = pDataStructure.template getSampleValueTriLinear< 0 >( pBrickSampler.brickChildPosInPool - pDataStructure.brickCacheResINV,
//		//																			/*offset of 1/2 voxel to reach texel*/0.5f * pDataStructure.brickCacheResINV );
//		//	printf( "\n\tGvKernel_ReadVboNbPoints : dataPoolPosition = [ %f | %f | %f ]", dataPoolPosition.x, dataPoolPosition.y, dataPoolPosition.z );
//		const float4 nbPoints = pDataStructure.template getSampleValueTriLinear< 0 >( dataPoolPosition,
//			/*offset of 1/2 voxel to reach texel*/0.5f * pDataStructure.brickCacheResINV );
//
//		// Write to output global memory
//		pNbPointsList[ elem ] = nbPoints.x;
//
//		//	printf( "\nGvKernel_ReadVboNbPoints : nbPoints = [ %f | %f | %f | %f ]", nbPoints.x, nbPoints.y, nbPoints.z, nbPoints.w );
//	}
//}
//
///******************************************************************************
// * GvKernel_UpdateVBO kernel
// *
// * This kernel update the VBO by dumping all used bricks content (i.e. points)
// *
// * @param pVBO VBO to update
// * @param pNbBricks number of bricks to process
// * @param pBrickAddressList list of brick addresses in cache (used to retrive positions where to fetch data)
// * @param pNbPointsList list of points inside each brick
// * @param pVboIndexOffsetList list of number of points for each used bricks
// * @param pDataStructure data structure in cache where to fecth data
// ******************************************************************************/
//template< class TDataStructureKernelType >
//__global__
//void GvKernel_UpdateVBO( float4* pVBO, const uint pNbBricks, const uint* pBrickAddressList, const uint* pNbPointsList, const uint* pVboIndexOffsetList, TDataStructureKernelType pDataStructure )
//{
//	// Retrieve global data index
//	uint lineSize = __uimul( blockDim.x, gridDim.x );
//	uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );
//
//	// Check bounds
//	if ( elem < pNbBricks )
//	{
//		// Iterate through points
//		const uint nbPoints = pNbPointsList[ elem ];
//		const uint indexOffset = pVboIndexOffsetList[ elem ];
//
//		//	printf( "\nGvKernel_UpdateVBO : brick indx [ %d / %d ] / nb points [ %d ] / indexOffset [ %d ]", elem + 1, pNbBricks, nbPoints, indexOffset );
//
//		for ( int i = 0; i < nbPoints; ++i )
//		{
//			// Retrieve the number of points in the current brick
//			//
//			// Brick position in cache (without border)
//			//float3 dataPoolPosition = make_float3( GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ elem ] ) ) * pDataStructure.brickCacheResINV;
//			//float3 dataPoolPosition = 10.0f * make_float3( GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ elem ] ) ) * pDataStructure.brickCacheResINV;
//
//			uint3 unpackedBrickAddress = ( TDataStructureKernelType::BrickResolution::get() + 2 * TDataStructureKernelType::brickBorderSize ) * GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ elem ] );
//
//			float3 dataPoolPosition = make_float3( unpackedBrickAddress.x, unpackedBrickAddress.y, unpackedBrickAddress.z ) * pDataStructure.brickCacheResINV;
//
//			// Shift from one voxel to be on the brick border
//			//dataPoolPosition -= pDataStructure.brickCacheResINV;
//			//const float4 point = pDataStructure.template getSampleValueTriLinear< 0 >( pBrickSampler.brickChildPosInPool - pDataStructure.brickCacheResINV,
//			//	/*offset of 1/2 voxel to reach texel*/( i + 2 ) * 0.5f * pDataStructure.brickCacheResINV );
//
//			//const float4 nbPoints = pDataStructure.template getSampleValueTriLinear< 0 >( dataPoolPosition,
//			//																		/*offset of 1/2 voxel to reach texel*/( i + 2 ) * 0.5f * pDataStructure.brickCacheResINV );
//
//
//			const float4 point = pDataStructure.template getSampleValueTriLinear< 0 >( dataPoolPosition
//				, 0.5f * pDataStructure.brickCacheResINV +  make_float3( ( i + 2 ) * pDataStructure.brickCacheResINV.x, 0.f, 0.f ) );
//
//			//	printf( "\nGvKernel_UpdateVBO : position [ %d / %d ] = [ %f | %f | %f | %f ]", i + 1, nbPoints, point.x, point.y, point.z, point.w );
//
//			// Write to output global memory
//			pVBO[ indexOffset + i ] = point;
//		}
//	}
//}
//	
//#endif // !_CACHE_MANAGER_KERNEL_H_
