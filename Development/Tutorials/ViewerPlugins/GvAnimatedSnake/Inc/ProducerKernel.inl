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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvStructure/GsNode.h>
//#include <GvStructure/GsVolumeTree.h>

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Initialize the producer
 * 
 * @param volumeTreeKernel Reference on a volume tree data structure
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::initialize( DataStructureKernel& pDataStructure )
{
	//_dataStructureKernel = pDataStructure;
}

/******************************************************************************
 * Produce data on device.
 * Implement the produceData method for the channel 0 (nodes)
 *
 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
 *
 * In the function, user has to produce data for a node tile or a brick of voxels :
 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
 * etc...
 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
 * user had previously defined (color, normal, density, etc...)
 *
 * @param pGpuPool The device side pool (nodes or bricks)
 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
 * @param pNewElemAddress The address at which to write the produced data in the pool
 * @param pParentLocInfo The localization info used to locate an element in the pool
 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
 *
 * @return A feedback value that the user can return.
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GsLocalizationInfo& parentLocInfo, Loki::Int2Type< 0 > )
{
	// NOTE :
	// In this method, you are inside a node tile.
	// The goal is to determine, for each node of the node tile, which type of data it holds.
	// Data type can be :
	// - a constant region,
	// - a region with data,
	// - a region where max resolution is reached.
	// So, one thread is responsible of the production of one node of a node tile.
	
	// Retrieve current node tile localization information code and depth
	const GvCore::GsLocalizationInfo::CodeType *parentLocCode = &parentLocInfo.locCode;
	const GvCore::GsLocalizationInfo::DepthType *parentLocDepth = &parentLocInfo.locDepth;

	// Process ID gives the 1D index of a node in the current node tile
	if ( processID < NodeRes::getNumElements() )
	{
		// First, compute the 3D offset of the node in the node tile
		uint3 subOffset = NodeRes::toFloat3( processID );

		// Node production corresponds to subdivide a node tile.
		// So, based on the index of the node, find the node child.
		// As we want to sudbivide the curent node, we retrieve localization information
		// at the next level
		uint3 regionCoords = parentLocCode->addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth->addLevel().get();

		// Create a new node for which you will have to fill its information.
		GvStructure::GsNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		// Call what we call an oracle that will determine the type of the region of the node accordingly
		GsOracleRegionInfo::OracleRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		// Now that the type of the region is found, fill the new node information
		if ( nodeinfo == GsOracleRegionInfo::GPUVP_CONSTANT )
		{
			newnode.setTerminal( true );
		}
		else if ( nodeinfo == GsOracleRegionInfo::GPUVP_DATA )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GsOracleRegionInfo::GPUVP_DATA_MAXRES )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}

		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// newElemAddress.x + processID : is the adress of the new node in the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( newElemAddress.x + processID, newnode.childAddress );
		nodePool.getChannel( Loki::Int2Type< 1 >() ).set( newElemAddress.x + processID, newnode.brickAddress );
	}

	return 0;
}

/******************************************************************************
 * Produce data on device.
 * Implement the produceData method for the channel 1 (bricks)
 *
 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
 *
 * In the function, user has to produce data for a node tile or a brick of voxels :
 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
 * etc...
 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
 * user had previously defined (color, normal, density, etc...)
 *
 * @param pGpuPool The device side pool (nodes or bricks)
 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
 * @param pNewElemAddress The address at which to write the produced data in the pool
 * @param pParentLocInfo The localization info used to locate an element in the pool
 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
 *
 * @return A feedback value that the user can return.
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GsLocalizationInfo& parentLocInfo, Loki::Int2Type< 1 > )
{
	// NOTE :
	// In this method, you are inside a brick of voxels.
	// The goal is to determine, for each voxel of the brick, the value of each of its channels.
	//
	// Data type has previously been defined by the user, as a
	// typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;
	//
	// In this tutorial, we have choosen one channel containing color at channel 0
	
	// Real brick size (with borders)
	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	
	// The original KERNEL execution configuration on the HOST has a 2D block size :
	// dim3 blockSize( 16, 8, 1 );
	//
	// Each block process one brick of voxels.
	//
	// One thread iterate in 3D space given a pattern defined by the 2D block size
	// and the following "for" loops. Loops take into account borders.
	// In fact, each thread of the current 2D block compute elements layer by layer
	// on the z axis.
	//
	// One thread process only a subset of the voxels of the brick.
	//
	// Iterate through z axis step by step as blockDim.z is equal to 1
	uint3 elemOffset;
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x )
			{
				// Compute position index
				uint3 locOffset = elemOffset + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z );

				// Test if the computed position index is inside the brick (with borders)
				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Determine voxel color
					float4 voxelColor;
					// The border of each voxel will be displayed in black to give "cube look-n-feel" (12 edges)
					if (
						locOffset.x == 1 && locOffset.y == 1 ||
						locOffset.x == BrickRes::get().x && locOffset.y == BrickRes::get().x ||
						locOffset.x == 1 && locOffset.z == 1 ||
						locOffset.x == BrickRes::get().x && locOffset.z == BrickRes::get().x ||
						locOffset.y == 1 && locOffset.z == 1 ||
						locOffset.y == BrickRes::get().x  && locOffset.z == BrickRes::get().x ||

						locOffset.x == BrickRes::get().x  && locOffset.y == 1 ||
						locOffset.x == 1 && locOffset.y == BrickRes::get().x  ||
						locOffset.x == BrickRes::get().x  && locOffset.z == 1 ||
						locOffset.x == 1 && locOffset.z == BrickRes::get().x  ||
						locOffset.y == BrickRes::get().x  && locOffset.z == 1 ||
						locOffset.y == 1 && locOffset.z == BrickRes::get().x )
					{
						voxelColor = make_float4( 0.f, 0.f, 0.f, cShapeOpacity );
					} 
					else
					{
						voxelColor = make_float4( cShapeColor.x, cShapeColor.y, cShapeColor.z, cShapeOpacity );
					}
		
					// Alpha pre-multiplication used to avoid the "color bleeding" effect
					/*voxelColor.x *= voxelColor.w;
					voxelColor.y *= voxelColor.w;
					voxelColor.z *= voxelColor.w;*/

					// Compute the new element's address
					uint3 destAddress = newElemAddress + locOffset;

					// Write the voxel's color in the first field
					dataPool.template setValue< 0 >( destAddress, voxelColor );
				}
			}
		}
	}

	return 0;
}

/******************************************************************************
 * Helper function used to determine the type of zones in the data structure.
 *
 * The data structure is made of regions containing data, empty or constant regions.
 * Besides, this function can tell if the maximum resolution is reached in a region.
 *
 * @param regionCoords region coordinates
 * @param regionDepth region depth
 *
 * @return the type of the region
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline GsOracleRegionInfo::OracleRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth )
{
	// Limit the depth.
	// Currently, 32 is the max depth of the GigaVoxels engine.
	if ( regionDepth >= 32 )
	{
		return GsOracleRegionInfo::GPUVP_DATA_MAXRES;
	}

	// Spatial oracle : there is data everywhere
	return GsOracleRegionInfo::GPUVP_DATA;
}
