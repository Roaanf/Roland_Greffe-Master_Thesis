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
 ****************************** INLINE DEFINITION *****************************
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
 * Initialize
 *
 * @param pNodesBuffer node buffer
 * @param pBricksPool bricks buffer
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::init( const GvCore::GsLinearMemoryKernel< uint >& pNodesBuffer, const BricksPoolKernelType& pBricksPool )
{
	_nodesCache = pNodesBuffer;
	_bricksCache = pBricksPool;
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
	const GvCore::GsLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GsLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	if ( processID < NodeRes::getNumElements() )
	{
		uint3 subOffset = NodeRes::toFloat3( processID );

		uint3 regionCoords = parentLocCode.addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth.addLevel().get();

		GvStructure::GsNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		GsOracleRegionInfo::OracleRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth, requestID, processID );

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

		// Write node info into the node pool
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
	// Here we are in a brick
	//
	// Each bloc is in charge of the production of one brick

	// Data types
	typedef typename GvCore::DataChannelType< DataTList, 0 >::Result ColorType;
	typedef typename GvCore::DataChannelType< DataTList, 1 >::Result NormalType;
	ColorType color;
	NormalType normal;

	// Brick offset in the mapped memory array where data has been produced on host
	// - requestID is the index of the brick
	const uint brickAddress = requestID * /*nbVoxels*/BrickFullRes::numElements;

	// Iterate through elements of the brick
	//
	// Each thread compute several elements spaced by blockDim.x (see the FOR loop offset)
	// - thread 0 => 0 | blockDim.x * blockDim.y | 2 x blockDim.x * blockDim.y | 3 x blockDim.x * blockDim.y | etc...
	// - thread 1 => 1 | 1 + blockDim.x * blockDim.y | 1 + 2 x blockDim.x * blockDim.y| 1 + 3 x blockDim.x * blockDim.y | etc...
	// - thread 2 => 2 | 2 + blockDim.x * blockDim.y | 2 + 2 x blockDim.x * blockDim.y| 2 + 3 x blockDim.x * blockDim.y| etc...
	//
	// note : here we have a 2D kernel block size
	uint3 voxelOffset;
	//const uint nbThreads = blockDim.x * blockDim.y;
	for ( uint dataOffset = processID; dataOffset < /*nbVoxels*/BrickFullRes::numElements; dataOffset += /*nbThreads*/BricksKernelBlockSize::numElements )
	{
		// Retrieve data
		// - retrieve voxel color previously generated on CPU
		color = _bricksCache.getChannel( Loki::Int2Type< 0 >() ).get( brickAddress + dataOffset );
		// - retrieve voxel normal previously generated on CPU
		normal = _bricksCache.getChannel( Loki::Int2Type< 1 >() ).get( brickAddress + dataOffset );

		// Write voxel data to data pool
		// - convert 1D element index into 3D offset (to be able to write data in cache)
		voxelOffset.x = dataOffset % BrickFullRes::x;
		voxelOffset.y = ( dataOffset / BrickFullRes::x ) % BrickFullRes::y;
		voxelOffset.z = dataOffset / ( BrickFullRes::x * BrickFullRes::y );
		// - write data
		const uint3 destAddress = newElemAddress + make_uint3( voxelOffset );
		dataPool.setValue< 0 >( destAddress, color );
		dataPool.setValue< 1 >( destAddress, normal );
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
 * @param nodeTileIndex ...
 * @param nodeTileOffset ...
 *
 * @return the type of the region
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline GsOracleRegionInfo::OracleRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth, uint nodeTileIndex, uint nodeTileOffset )
{
	if ( regionDepth >= 32 )
	{
		return GsOracleRegionInfo::GPUVP_DATA_MAXRES;
	}

	// Retrieve node info previously generated on CPU
	uint nodeInfo = _nodesCache.get( nodeTileIndex * NodeRes::getNumElements() + nodeTileOffset );
	if ( nodeInfo )
	{
		return GsOracleRegionInfo::GPUVP_DATA;
	}
	else
	{
		return GsOracleRegionInfo::GPUVP_CONSTANT;
	}
}
