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
 * Inititialize
 *
 * @param maxdepth max depth
 * @param nodescache nodes cache
 * @param datacachepool data cache pool
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::init( uint maxdepth, const GvCore::GsLinearMemoryKernel< uint >& nodescache, const DataCachePoolKernelType& datacachepool )
{
	_maxDepth = maxdepth;
	_cpuNodesCache = nodescache;
	_cpuDataCachePool = datacachepool;
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
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID,
				uint3 newElemAddress, const GvCore::GsLocalizationInfo& parentLocInfo,
				Loki::Int2Type< 0 > )
{
	// NOTE :
	// In this method, you are inside a node tile.
	// A pre-process step on HOST has previously determined, for each node of the node tile, which type of data it holds.
	// Data type can be :
	// - a constant region,
	// - a region with data,
	// - a region where max resolution is reached.
	// So, one thread is responsible of the production of one node of a node tile.

	// Get localization info (code and depth)
	uint3 parentLocCode = parentLocInfo.locCode.get();
	uint parentLocDepth = parentLocInfo.locDepth.get();

	// Check bound
	if ( processID < NodeRes::getNumElements() )
	{
		// Create a new node
		GvStructure::GsNode newnode;

		// Initialize the child address with the HOST nodes cache
		newnode.childAddress = _cpuNodesCache.get( requestID * NodeRes::getNumElements() + processID );

		// Initialize the brick address
		newnode.brickAddress = 0;

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
template< typename TGPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( TGPUPoolKernelType& pDataPool, uint pRequestID, uint pProcessID,
				uint3 pNewElemAddress, const GvCore::GsLocalizationInfo& pParentLocInfo,
				Loki::Int2Type< 1 > )
{
	// parentLocDepth++; //Shift needed, to be corrected
	bool nonNull = ProducerKernel_ChannelLoad< TDataStructureType, TGPUPoolKernelType, GvCore::DataNumChannels< DataTList >::value - 1>::produceDataChannel( *this, pDataPool, pNewElemAddress, pParentLocInfo, pRequestID, pProcessID );

	return 0;
	//if (nonNull)
	//	return 0;
	//else
	//	return 2;
}

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Produce data at the specified channel
 *
 * @param gpuVPLK reference on the volume producer load kernel
 * @param dataPool the data pool in which to write data
 * @param elemAddress The address at which to write the produced data in the pool
 * @param parentLocInfo The localization info used to locate an element in the pool
 * @param pRequestID The current processed element coming from the data requests list (a brick)
 * @param pProcessID Index of one of the elements inside a voxel bricks
******************************************************************************/
template< typename TDataStructureType, typename TGPUPoolKernelType, int channel >
__device__
inline bool ProducerKernel_ChannelLoad< TDataStructureType, TGPUPoolKernelType, channel >
::produceDataChannel( ProducerKernel< TDataStructureType >& gpuVPLK,
					 TGPUPoolKernelType& dataPool, uint3 elemAddress, const GvCore::GsLocalizationInfo& parentLocInfo, uint requestID, uint processID )
{
	uint blockIndex = requestID;

	// Number of voxels
	uint brickNumVoxels = BrickFullRes::numElements;
	uint blockStartAddress = blockIndex * ProducerKernel< TDataStructureType >::BrickVoxelAlignment;

	uint blockNumThreads = blockDim.x * blockDim.y * blockDim.z;

	// Iterate through voxels of the current brick
	uint decal;
	for ( decal = 0; decal < brickNumVoxels; decal += blockNumThreads )
	{
		uint locDecal = decal + processID;

		if ( locDecal < brickNumVoxels )
		{
			typedef typename GvCore::DataChannelType< DataTList, channel >::Result VoxelType;
			VoxelType voxelData;

			uint locDecalOffset = locDecal;
			voxelData = gpuVPLK._cpuDataCachePool.getChannel( Loki::Int2Type< channel >() ).get( blockStartAddress + locDecalOffset );

			uint3 voxelOffset;
			voxelOffset.x = locDecal % BrickFullRes::x;
			voxelOffset.y = ( locDecal / BrickFullRes::x ) % BrickFullRes::y;
			voxelOffset.z = ( locDecal / ( BrickFullRes::x * BrickFullRes::y ) );
			uint3 destAddress = elemAddress + make_uint3( voxelOffset );

			// Write the voxel's data for the specified channel index
			dataPool.setValue< channel >( destAddress, voxelData );
		}
	}

	// Recursive call to produce data until the last channel is reached
	return ProducerKernel_ChannelLoad< TDataStructureType, TGPUPoolKernelType, channel - 1 >::produceDataChannel( gpuVPLK, dataPool, elemAddress, parentLocInfo, requestID, processID );
}

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Produce data at the specified channel
 *
 * @param gpuVPLK reference on the volume producer load kernel
 * @param dataPool the data pool in which to write data
 * @param elemAddress The address at which to write the produced data in the pool
 * @param parentLocInfo The localization info used to locate an element in the pool
 * @param pRequestID The current processed element coming from the data requests list (a brick)
 * @param pProcessID Index of one of the elements inside a voxel bricks
******************************************************************************/
template< typename TDataStructureType, typename TGPUPoolKernelType >
__device__
inline bool ProducerKernel_ChannelLoad< TDataStructureType, TGPUPoolKernelType, -1 >
::produceDataChannel( ProducerKernel< TDataStructureType >& gpuVPLK,
					  TGPUPoolKernelType& dataPool, uint3 elemAddress, const GvCore::GsLocalizationInfo& parentLocInfo, uint requestID, uint processID )
{
	return false;
}
