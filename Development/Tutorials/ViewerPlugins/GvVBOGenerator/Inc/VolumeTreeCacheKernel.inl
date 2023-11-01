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
///******************************************************************************
// ******************************* INCLUDE SECTION ******************************
// ******************************************************************************/
//
///******************************************************************************
// ****************************** INLINE DEFINITION *****************************
// ******************************************************************************/
//
//namespace GvStructure
//{
//
///******************************************************************************
// * Update buffer with a subdivision request for a given node.
// *
// * @param nodeAddressEnc the encoded node address
// ******************************************************************************/
//template< class NodeTileRes, class BrickFullRes, class NodeAddressType, class BrickAddressType >
//__device__
//inline void GsDataProductionManagerKernel< NodeTileRes, BrickFullRes, NodeAddressType, BrickAddressType >
//::subDivRequest( uint nodeAddressEnc )
//{
//	// Retrieve 3D node address
//	uint3 nodeAddress = NodeAddressType::unpackAddress( nodeAddressEnc );
//
//	// Update buffer with a subdivision request for that node
//	_updateBufferArray.set( nodeAddress, ( nodeAddressEnc & 0x3FFFFFFF ) | VTC_REQUEST_SUBDIV );
//}
//
///******************************************************************************
// * Update buffer with a load request for a given node.
// *
// * @param nodeAddressEnc the encoded node address
// ******************************************************************************/
//template< class NodeTileRes, class BrickFullRes, class NodeAddressType, class BrickAddressType >
//__device__
//inline void GsDataProductionManagerKernel< NodeTileRes, BrickFullRes, NodeAddressType, BrickAddressType >
//::loadRequest( uint nodeAddressEnc )
//{
//	// Retrieve 3D node address
//	uint3 nodeAddress = NodeAddressType::unpackAddress( nodeAddressEnc );
//
//	// Update buffer with a load request for that node
//	_updateBufferArray.set( nodeAddress, ( nodeAddressEnc & 0x3FFFFFFF ) | VTC_REQUEST_LOAD );
//}
//
//} // namespace GvStructure
//
///******************************************************************************
// ****************************** KERNEL DEFINITION *****************************
// ******************************************************************************/
//
//namespace GvStructure
//{
//
///******************************************************************************
// * KERNEL ClearVolTreeRoot
// *
// * @param volumeTree ...
// * @param rootAddress ...
// ******************************************************************************/
//template< typename VolTreeKernelType >
//__global__
//void ClearVolTreeRoot( VolTreeKernelType volumeTree, uint rootAddress )
//{
//	//uint lineSize = __uimul( blockDim.x, gridDim.x );
//	uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x );// + __uimul( blockIdx.y, lineSize );
//
//	if ( elem < VolTreeKernelType::NodeResolution::getNumElements() )
//	{
//		GvStructure::GsNode node;
//		node.childAddress = 0;
//		node.brickAddress = 0;
//
//		volumeTree.setNode( node, rootAddress + elem );
//	}
//}
//
//// Updates
///******************************************************************************
// * KERNEL UpdateBrickUsage
// *
// * @param volumeTree ...
// * @param rootAddress ...
// ******************************************************************************/
//template< typename ElementRes, typename GPUCacheType >
//__global__
//void UpdateBrickUsage( uint numElems, uint* lruElemAddressList, GPUCacheType gpuCache )
//{
//	uint elemNum = blockIdx.x;
//
//	if ( elemNum < numElems )
//	{
//		uint elemIndexEnc = lruElemAddressList[ elemNum ];
//		uint3 elemIndex = GvStructure::VolTreeBrickAddress::unpackAddress( elemIndexEnc );
//		uint3 elemAddress = elemIndex * ElementRes::get();
//
//		// FIXME: fixed border size !
//		uint3 brickAddress = elemAddress + make_uint3( 1 );
//		gpuCache._brickCacheManager.setElementUsage( brickAddress );
//	}
//}
//
///******************************************************************************
//* KERNEL GvKernel_PreProcessRequests
//*
//* This kernel is used as first pass a stream compaction algorithm
//* in order to create the masks of valid requests
//* (i.e. the ones that have been requested during the N3-Tree traversal).
//*
//* @param pRequests Array of requests (i.e. subdivide nodes or load/produce bricks)
//* @param pIsValidMask Resulting array of isValid masks
//* @param pNbElements Number of elememts to process
//******************************************************************************/
//__global__
//void GvKernel_PreProcessRequests( const uint* pRequests, unsigned int* pIsValidMasks, const uint pNbElements )
//{
//	// Retrieve global data index
//	uint lineSize = __uimul( blockDim.x, gridDim.x );
//	uint index = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );
//
//	// Check bounds
//	if ( index < pNbElements )
//	{
//		// Set the associated isValid mask, knowing that
//		// the input requests buffer is reset to 0 at each frame
//		// (i.e. a value different of zero means that a request has been emitted).
//		if ( pRequests[ index ] == 0 )
//		{
//			pIsValidMasks[ index ] = 0;
//		}
//		else
//		{
//			pIsValidMasks[ index ] = 1;
//		}
//	}
//}
//
//} // namespace GvStructure
