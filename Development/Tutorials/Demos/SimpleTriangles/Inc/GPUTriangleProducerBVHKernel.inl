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

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Initialize
 *
 * @param h_nodesbufferarray node buffer
 * @param h_vertexbufferpool data buffer
 ******************************************************************************/
template< typename TDataStructureType, uint TDataPageSize >
inline void GPUTriangleProducerBVHKernel< TDataStructureType, TDataPageSize >
::init( VolTreeBVHNodeStorage* h_nodesbufferarray, GvCore::GPUPoolKernel< GvCore::GsLinearMemoryKernel, DataTypeList > h_vertexbufferpool )
{
	_nodesBufferKernel = h_nodesbufferarray;
	_dataBufferKernel = h_vertexbufferpool;
}

/******************************************************************************
 * Produce node tiles
 *
 * @param nodePool ...
 * @param requestID ...
 * @param processID ...
 * @param pNewElemAddress ...
 * @param parentLocInfo ...
 *
 * @return ...
 ******************************************************************************/
template< typename TDataStructureType, uint TDataPageSize >
template< typename GPUPoolKernelType >
__device__
inline uint GPUTriangleProducerBVHKernel< TDataStructureType, TDataPageSize >
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID,
				uint3 pNewElemAddress, const VolTreeBVHNodeUser& node, Loki::Int2Type< 0 > )
{
	return 0;
}

/******************************************************************************
 * Produce bricks of data
 *
 * @param dataPool ...
 * @param requestID ...
 * @param processID ...
 * @param pNewElemAddress ...
 * @param parentLocInfo ...
 *
 * @return ...
 ******************************************************************************/
template< typename TDataStructureType, uint TDataPageSize >
template< typename GPUPoolKernelType >
__device__
inline uint GPUTriangleProducerBVHKernel< TDataStructureType, TDataPageSize >
::produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID,
				uint3 pNewElemAddress, VolTreeBVHNodeUser& pNode, Loki::Int2Type< 1 > )
{
	const uint hostElementAddress = pNode.getDataIdx() * BVH_DATA_PAGE_SIZE;

	// Check bounds
	if ( processID < TDataPageSize )
	{
		// Retrieve data
		const float4 position = _dataBufferKernel.getChannel( Loki::Int2Type< 0 >() ).get( hostElementAddress + processID );
		const uchar4 color = _dataBufferKernel.getChannel( Loki::Int2Type< 1 >() ).get( hostElementAddress + processID );

		// Write data in data pool
		dataPool.getChannel( Loki::Int2Type< 0 >() ).set( pNewElemAddress.x + processID, position );
		dataPool.getChannel( Loki::Int2Type< 1 >() ).set( pNewElemAddress.x + processID, color );
	}

	// Update the node's data field and set the gpu flag
	pNode.setDataIdx( pNewElemAddress.x / BVH_DATA_PAGE_SIZE );
	pNode.setGPULink();

	return 0;
}

/******************************************************************************
 * Seems to be unused anymore...
 ******************************************************************************/
template< typename TDataStructureType, uint TDataPageSize >
template< class GPUTreeBVHType >
__device__
inline uint GPUTriangleProducerBVHKernel< TDataStructureType, TDataPageSize >
::produceNodeTileData( GPUTreeBVHType& gpuTreeBVH, uint requestID, uint processID, VolTreeBVHNodeUser& node, uint newNodeTileAddressNode )
{
	// Shared Memory
	__shared__ VolTreeBVHNodeStorage newNodeStorage[ 2 ];	// Not needed, can try without shared memory later

	// TODO: loop to deal with multiple pages per block
	if ( processID < VolTreeBVHNodeStorage::numWords * 2 )
	{
		uint tileElemNum = processID / VolTreeBVHNodeStorage::numWords;	// TODO:check perfos !
		uint tileElemWord = processID % VolTreeBVHNodeStorage::numWords;

		uint cpuBufferPageAddress = node.getSubNodeIdx() + tileElemNum;

		// Parallel coalesced read
		newNodeStorage[ tileElemNum ].words[ tileElemWord ] = _nodesBufferKernel[ cpuBufferPageAddress ].words[ tileElemWord ];	// Can be optimized for address comp

		// Parallel write
		gpuTreeBVH.parallelWriteBVHNode( tileElemWord, newNodeStorage[ tileElemNum ], newNodeTileAddressNode + tileElemNum );
	}

	node.setSubNodeIdx( newNodeTileAddressNode );
	node.setGPULink();

	return 0;
}
