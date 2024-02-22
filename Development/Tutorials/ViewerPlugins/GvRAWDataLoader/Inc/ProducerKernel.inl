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

/**
 * Transfer function texture
 */
texture< float4, cudaTextureType1D, cudaReadModeElementType > transferFunctionTexture;

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
::init( uint maxdepth, const GvCore::GsLinearMemoryKernel< uint >& nodescache, const DataCachePoolKernelType& datacachepool, const GvCore::GsLinearMemoryKernel< unsigned short >& requestscache)
{
	_maxDepth = maxdepth;
	_hostNodeCache = nodescache;
	_hostDataCache = datacachepool;
	_hostRangeCache = requestscache;
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
::produceData( GPUPoolKernelType& pNodePool, uint pRequestID, uint pProcessID,
				uint3 pNewElemAddress, const GvCore::GsLocalizationInfo& pParentLocInfo,
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
	uint3 parentLocCode = pParentLocInfo.locCode.get();
	uint parentLocDepth = pParentLocInfo.locDepth.get();

	// Check bound
	if ( pProcessID < NodeRes::getNumElements() )
	{
		// Create a new node
		GvStructure::GsNode newnode;

		// Initialize the child address with the HOST nodes cache
		newnode.childAddress = _hostNodeCache.get( pRequestID * NodeRes::getNumElements() + pProcessID );

		// Initialize the brick address
		newnode.brickAddress = 0;

		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// pNewElemAddress.x + pProcessID : is the adress of the new node in the node pool
		pNodePool.getChannel( Loki::Int2Type< 0 >() ).set( pNewElemAddress.x + pProcessID, newnode.childAddress );
		pNodePool.getChannel( Loki::Int2Type< 1 >() ).set( pNewElemAddress.x + pProcessID, newnode.brickAddress );
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
* @param pNewElemAddress The address at which to write the produced data in the pool (IN VOXEL)
* @param pParentLocInfo The localization info used to locate an element in the pool
* @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
* 
* This is done VOXEL per VOXEL the issue is very probably here !!!
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
	// Voxel data type definition
	typedef typename GvCore::DataChannelType< DataTList, 0/*data channel index*/ >::Result VoxelType;

	// Useful variables
	VoxelType voxelData;
	uint3 voxelOffset;
	uint3 destAddress;
	
	// Shared Memory initialization
	__shared__ bool smIsEmptyNode;
	if ( pProcessID == 0 )
	{
		smIsEmptyNode = false;
	}
	// Thread Synchronization
	__syncthreads();

	unsigned short min = _hostRangeCache.get(pRequestID * 2);
	unsigned short max = _hostRangeCache.get(pRequestID * 2 + 1);
	if (((min <= cProducerThresholdLow) && (max <= cProducerThresholdLow)) || ((min >= cProducerThresholdHigh) && (max >= cProducerThresholdHigh))) {
		// The brick is not in the Producer's range
		return 2;
	}
	/*
	if (threadIdx.x == 0) {
		printf("pNewElemAddress : %u / %u / %u \n", pNewElemAddress.x, pNewElemAddress.y, pNewElemAddress.z);
		//printf("destAddress : %u / %u / %u \n", destAddress.x, destAddress.y, destAddress.z);
	}
	*/

	// Iterate through voxels of the current brick
	const size_t blockStartAddress = (size_t)pRequestID/*brickIndex*/ * (size_t)ProducerKernel< TDataStructureType >::BrickVoxelAlignment;
	const size_t nbThreads = (size_t)blockDim.x * (size_t)blockDim.y * (size_t)blockDim.z;
	for ( size_t index = pProcessID; index < BrickFullRes::numElements/*nbVoxels*/; index += nbThreads )
	{
		// Retrieve Host data
		voxelData = _hostDataCache.getChannel( Loki::Int2Type< 0/*data channel index*/ >() ).get( blockStartAddress + index );

		/*
		// Threshold management
		// => modify the return value to flag the node as empty if required
		if ( (voxelData >= cProducerThresholdLow) && (voxelData <= cProducerThresholdHigh) )
		{
			smIsEmptyNode = false;
		}
		*/

		// Compute offset in memory where to write data
		voxelOffset.x = index % BrickFullRes::x;
		voxelOffset.y = ( index / BrickFullRes::x ) % BrickFullRes::y;
		voxelOffset.z = index / ( BrickFullRes::x * BrickFullRes::y );
		destAddress = pNewElemAddress + make_uint3( voxelOffset );

		// Write the voxel's data for the specified channel index
		pDataPool.setValue< 0/*data channel index*/ >( destAddress, voxelData );
	}

	// Thread Synchronization
	__syncthreads();
	if ( smIsEmptyNode )
	{
		return 2;
	}

	return 0;
}
