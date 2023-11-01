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

#ifndef _VOLUME_TREE_CACHE_KERNEL_H_
#define _VOLUME_TREE_CACHE_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvStructure/GsDataProductionManagerKernel.h>
#include <GvCache/GsCacheManagerKernel.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct VolumeTreeCacheKernel
 *
 * @brief The VolumeTreeCacheKernel struct provides methods to update buffer
 * of requests on device.
 *
 * Device-side object used to update the buffer of requests emitted by the renderer
 * during the data structure traversal. Requests can be either "node subdivision"
 * or "load brick of voxels".
 */
template< class NodeTileRes, class BrickFullRes, class NodeAddressType, class BrickAddressType, class TPriorityPoliciesManager >
struct VolumeTreeCacheKernel : public GvStructure::GsDataProductionManagerKernel< NodeTileRes, BrickFullRes, NodeAddressType, BrickAddressType, TPriorityPoliciesManager >
{
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Brick cache manager
	 */
	GvCache::GsCacheManagerKernel< BrickFullRes, BrickAddressType > _vboCacheManager;

	/******************************** METHODS *********************************/

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

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

// #include "VolumeTreeCacheKernel.inl"

/******************************************************************************
 * GvKernel_FlagTimeStampsSP kernel
 *
 * This kernel creates the usage mask list of used and non used elements (in current rendering pass) in a single pass
 *
 * @param pCacheManager Cache manager
 * @param pNumElem Number of elememts to process
 * @param pTimeStampsElemAddressList Timestamp buffer list
 * @param pTempMaskList Resulting temporary mask list of non-used elements
 * @param pTempMaskList2 Resulting temporary mask list of used elements
 ******************************************************************************/
//template< class ElementRes, class AddressType >
template< class TCacheManagerKernelType >
__global__
void GvKernel_FlagTimeStampsSP( /*GvCache::GsCacheManagerKernel< ElementRes, AddressType >*/ TCacheManagerKernelType pCacheManager,
								  const uint pNumElem, const uint* pTimeStampsElemAddressList, uint* pTempMaskList, uint* pTempMaskList2 )
{
	// Retrieve global data index
	const uint lineSize = __uimul( blockDim.x, gridDim.x );
	const uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

	// Check bounds
	if ( elem < pNumElem )
	{
		// Retrieve element processed by current thread
		const uint elemAddressEnc = pTimeStampsElemAddressList[ elem ];

		// Unpack its adress
		//const uint3 elemAddress = AddressType::unpackAddress( elemAddressEnc );
		const uint3 elemAddress = GvStructure::VolTreeBrickAddress::unpackAddress( elemAddressEnc );

		////-------------------------------------------
		//const uint brickTime = pCacheManager._timeStampArray.get( elemAddress );
		//if ( brickTime != 0 )
		//{
		//	printf( "\nADDRESS [ %u %u %u ] - TIME [ %u ] ---- elemAddressEnc [ %u ] - elem [ %u ] ", elemAddress.x, elemAddress.y, elemAddress.z, brickTime, elemAddressEnc, elem );
		//}
		////-------------------------------------------

		// Generate an error
		if ( pCacheManager._timeStampArray.get( elemAddress ) == k_currentTime )
		//if ( brickTime == k_currentTime )
		{
			pTempMaskList[ elem ] = 0;
			pTempMaskList2[ elem ] = 1;
		}
		else
		{
			pTempMaskList[ elem ] = 1;
			pTempMaskList2[ elem ] = 0;
		}
	}
}

	/******************************************************************************
	 * GvKernel_ReadVboNbPoints kernel
	 *
	 * This kernel retrieve number of points contained in each used bricks
	 *
	 * @param pNbPointsList [out] list of points inside each brick
	 * @param pNbBricks number of bricks to process
	 * @param pBrickAddressList list of brick addresses in cache (used to retrive positions where to fetch data)
	 * @param pDataStructure data structure in cache where to fecth data
	 ******************************************************************************/
	template< class TDataStructureKernelType >
	__global__
	void GvKernel_ReadVboNbPoints( uint* pNbPointsList, const uint pNbBricks, const uint* pBrickAddressList, TDataStructureKernelType pDataStructure )
	{
		// Retrieve global data index
		uint lineSize = __uimul( blockDim.x, gridDim.x );
		uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

		// Check bounds
		if ( elem < pNbBricks )
		{
			// Retrieve the number of points in the current brick
			//
			// Brick position in cache (without border)
			
			// The adress pBrickAddressList[ elem ] is the adress of a brick in the elemAdressList 1D linearized array cache.
			// The adress is the adress of the brick beginning with the border (there is no one border ofset)
			uint3 unpackedBrickAddress = ( TDataStructureKernelType::BrickResolution::get() + 2 * TDataStructureKernelType::brickBorderSize ) * GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ elem ] );
					
			float3 dataPoolPosition = make_float3( unpackedBrickAddress.x, unpackedBrickAddress.y, unpackedBrickAddress.z ) * pDataStructure.brickCacheResINV;
		
			// Shift from one voxel to be on the brick border
			const float4 nbPoints = pDataStructure.template getSampleValueTriLinear< 0 >( dataPoolPosition,
																						/*offset of 1/2 voxel to reach texel*/0.5f * pDataStructure.brickCacheResINV );
																																	
			// Write to output global memory
			//
			// WARNING : nb is in the fourth component
			pNbPointsList[ elem ] = nbPoints.w;
		}
	}

	/******************************************************************************
	 * GvKernel_UpdateVBO kernel
	 *
	 * This kernel update the VBO by dumping all used bricks content (i.e. points)
	 *
	 * @param pVBO VBO to update
	 * @param pNbBricks number of bricks to process
	 * @param pBrickAddressList list of brick addresses in cache (used to retrive positions where to fetch data)
	 * @param pNbPointsList list of points inside each brick
	 * @param pVboIndexOffsetList list of number of points for each used bricks
	 * @param pDataStructure data structure in cache where to fecth data
	 ******************************************************************************/
	template< class TDataStructureKernelType >
	__global__ 
	void GvKernel_UpdateVBO( float3* pVBO, const uint pNbBricks, const uint* pBrickAddressList, const uint* pNbPointsList, const uint* pVboIndexOffsetList, TDataStructureKernelType pDataStructure )
	{
	
		// Retrieve global data index
		uint lineSize = __uimul( blockDim.x, gridDim.x );
		uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );
		uint total_size = lineSize * gridDim.y;
		uint total_number_of_stars = pNbPointsList[pNbBricks-1] + pVboIndexOffsetList[pNbBricks-1];

		//----------------------------------------
		if ( total_number_of_stars > 999999 )
		{
			return;
		}
		//----------------------------------------
		
		int middle_index = pNbBricks>>1; // Index of the middle of the offset array
		int left_chunk_size = middle_index; // Size of the left chunk 
		int right_chunk_size = pNbBricks - left_chunk_size; // Size of the right chuck
		int old_left_chunk_size;	// Memory
		int index_of_brick=0;		// Where is the brick

		while (elem< total_number_of_stars)
		{
	
			bool out = false;			// Exit the while loop
			
			
			while(out == false) {
				if ( pVboIndexOffsetList[middle_index]>elem){
					if (left_chunk_size <= 1)
					{
						index_of_brick = middle_index-1;
						out = true;
					}
					old_left_chunk_size = left_chunk_size;
					left_chunk_size = left_chunk_size>>1;
					right_chunk_size = old_left_chunk_size - left_chunk_size;
					middle_index -= right_chunk_size;

				} else {
					if (right_chunk_size <= 1)
					{
						index_of_brick = middle_index;
						out = true;
					}
					left_chunk_size = right_chunk_size>>1 ;
					middle_index += left_chunk_size ;
					right_chunk_size = right_chunk_size - left_chunk_size;
					
				}
				
			}

			
			
			// Retrieve the number of points in the current brick
			//
			// Brick position in cache (without border)

			const uint3 brickCacheResolution = TDataStructureKernelType::BrickResolution::get() + 2 * TDataStructureKernelType::brickBorderSize;


			uint local_offset = pVboIndexOffsetList[index_of_brick];
			
			// Retrieve sphere index
			uint3 index3D;
			index3D.x = ( elem + 2 - local_offset) % static_cast< uint >( brickCacheResolution.x );
			index3D.y = ( ( elem + 2 - local_offset) / static_cast< uint >( brickCacheResolution.x ) ) % static_cast< uint >( brickCacheResolution.y );
			index3D.z = ( elem + 2 - local_offset) / static_cast< uint >( brickCacheResolution.x * brickCacheResolution.y );
					

			uint3 unpackedBrickAddress = ( TDataStructureKernelType::BrickResolution::get() + 2 * TDataStructureKernelType::brickBorderSize ) * GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ index_of_brick ] );
		
			float3 dataPoolPosition = make_float3( unpackedBrickAddress.x, unpackedBrickAddress.y, unpackedBrickAddress.z ) * pDataStructure.brickCacheResINV;

			// Sample data structrure to retrieve sphere data (position and radius)
			const float4 point = pDataStructure.template getSampleValueTriLinear< 0 >(
												dataPoolPosition,
												0.5f * pDataStructure.brickCacheResINV
												+ make_float3( index3D.x * pDataStructure.brickCacheResINV.x, index3D.y * pDataStructure.brickCacheResINV.y, index3D.z * pDataStructure.brickCacheResINV.z )
			);


			// Write to output global memory
			pVBO[ elem ] = make_float3( point.x, point.y, point.z );

			elem += total_size;
		}

		
		/*
		// Check bounds
		if ( elem < pNbBricks )
		{
			// Iterate through points
			const uint nbPoints = pNbPointsList[ elem ];
			const uint indexOffset = pVboIndexOffsetList[ elem ];
			for ( int i = 0; i < nbPoints; ++i )
			{
				// Retrieve the number of points in the current brick
				//
				// Brick position in cache (without border)

				const uint3 brickCacheResolution = TDataStructureKernelType::BrickResolution::get() + 2 * TDataStructureKernelType::brickBorderSize;

				// Retrieve sphere index
				uint3 index3D;
				index3D.x = ( i + 2 ) % static_cast< uint >( brickCacheResolution.x );
				index3D.y = ( ( i + 2 ) / static_cast< uint >( brickCacheResolution.x ) ) % static_cast< uint >( brickCacheResolution.y );
				index3D.z = ( i + 2 ) / static_cast< uint >( brickCacheResolution.x * brickCacheResolution.y );
							
				uint3 unpackedBrickAddress = ( TDataStructureKernelType::BrickResolution::get() + 2 * TDataStructureKernelType::brickBorderSize ) * GvStructure::GsNode::unpackBrickAddress( pBrickAddressList[ elem ] );
		
				float3 dataPoolPosition = make_float3( unpackedBrickAddress.x, unpackedBrickAddress.y, unpackedBrickAddress.z ) * pDataStructure.brickCacheResINV;

				//// Shift from one voxel to be on the brick border
				//const float4 point = pDataStructure.template getSampleValueTriLinear< 0 >( dataPoolPosition
				//		, 0.5f * pDataStructure.brickCacheResINV +  make_float3( ( i + 2 ) * pDataStructure.brickCacheResINV.x, 0.f, 0.f ) );

				// Sample data structrure to retrieve sphere data (position and radius)
				const float4 point = pDataStructure.template getSampleValueTriLinear< 0 >(
													dataPoolPosition,
													0.5f * pDataStructure.brickCacheResINV
													+ make_float3( index3D.x * pDataStructure.brickCacheResINV.x, index3D.y * pDataStructure.brickCacheResINV.y, index3D.z * pDataStructure.brickCacheResINV.z )
				);

				// Write to output global memory
				pVBO[ indexOffset + i ] = make_float3( point.x, point.y, point.z );
				
			}
		}
		
		*/
	}

#endif
