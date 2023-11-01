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
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Get data from data pool given a channel and an address
 *
 * @param ... ...
 *
 * @return ...
 ******************************************************************************/
template< class TDataTypeList >
template< int TChannelIndex >
__device__
inline typename GvCore::DataChannelType< TDataTypeList, TChannelIndex >::Result BvhTreeKernel< TDataTypeList >
::getVertexData( uint pAddress )
{
	return _dataPool.getChannel( Loki::Int2Type< TChannelIndex >() ).get( pAddress );
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::fetchBVHNode( VolTreeBVHNodeUser& resnode, uint pAddress )
{
	VolTreeBVHNode tempNodeUnion;

	/*for ( uint i = 0; i < VolTreeBVHNodeStorageUINT::numWords; i++ )
	{
		tempNodeUnion.storageUINTNode.words[ i ] = tex1Dfetch( volumeTreeBVHTexLinear, pAddress * VolTreeBVHNodeStorageUINT::numWords + i );
	}*/
	tempNodeUnion.storageUINTNode = _volumeTreeBVHArray.get( pAddress );

	resnode	= tempNodeUnion.userNode;
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::parallelFetchBVHNode( uint Pid, VolTreeBVHNodeUser& resnode, uint pAddress )
{
	// Shared Memory declaration
	__shared__ VolTreeBVHNode tempNodeUnion;

	if ( Pid < VolTreeBVHNodeStorageUINT::numWords )
	{
		//
		//tempNodeUnion.storageUINTNode.words[Pid] =k_volumeTreeBVHArray.get(pAddress).words[Pid];
#if 1
		uint* arrayAddress = (uint*)_volumeTreeBVHArray.getPointer( 0 );
		tempNodeUnion.storageUINTNode.words[ Pid ] = arrayAddress[ pAddress * VolTreeBVHNodeStorageUINT::numWords + Pid ];
#else
		tempNodeUnion.storageUINTNode.words[Pid] =tex1Dfetch(volumeTreeBVHTexLinear, pAddress*VolTreeBVHNodeStorageUINT::numWords+Pid);
#endif
	}

	// Thread synchronization
	__syncthreads();

	///resnode	=tempNodeUnion.userNode;

	if ( Pid == 0 )
	{
		resnode	= tempNodeUnion.userNode;
	}

	// Thread synchronization
	__syncthreads();
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::parallelFetchBVHNodeTile( uint Pid, VolTreeBVHNodeUser* resnodetile, uint pAddress )
{
	// Shared memory
	//__shared__ VolTreeBVHNode tempNodeUnion;

	if ( Pid < VolTreeBVHNodeStorageUINT::numWords * 2 )
	{
		uint* arrayAddress = (uint*)_volumeTreeBVHArray.getPointer( 0 );
		uint* resnodetileUI = (uint*)resnodetile;

		resnodetileUI[ Pid ] =  arrayAddress[ pAddress * VolTreeBVHNodeStorageUINT::numWords + Pid ];
	}

	// Thread synchronization
	__syncthreads();
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::writeBVHNode( const VolTreeBVHNodeUser& node, uint pAddress )
{
	VolTreeBVHNode tempNodeUnion;
	tempNodeUnion.userNode = node;

	// Update 
	_volumeTreeBVHArray.set( pAddress, tempNodeUnion.storageUINTNode );
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::parallelWriteBVHNode( uint Pid, const VolTreeBVHNodeStorage& node, uint pAddress )
{
	//Warning, no checking on Pid

	VolTreeBVHNodeStorage* storageArrayPtr = (VolTreeBVHNodeStorage*)_volumeTreeBVHArray.getPointer();

	storageArrayPtr[ pAddress ].words[ Pid ] = node.words[ Pid ];
}
