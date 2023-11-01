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
 * Constructor
 *
 * @param pNodePoolSize Cache size used to store nodes
 * @param pVertexPoolSize Cache size used to store bricks
 ******************************************************************************/
template< class DataTList >
BvhTree< DataTList >::BvhTree( uint nodePoolSize, uint vertexPoolSize )
:	GsIDataStructure()
{
	// Node pool initialization
	_nodePool = new GvCore::GsLinearMemory< VolTreeBVHNodeUser >( dim3( nodePoolSize, 1, 1 ) );
	_nodePool->fill( 0 );

	// Data pool initialization
	_dataPool = new GvCore::GPUPoolHost< GvCore::GsLinearMemory, DataTList >( make_uint3( vertexPoolSize, 1, 1 ) );
	//_dataPool->fill(0);

	// Associated device-side object initialization
	_kernelObject._dataPool = _dataPool->getKernelObject();
	GvCore::GsLinearMemory< VolTreeBVHNodeStorageUINT >* nodesArrayGPUStoragePtr = ( GvCore::GsLinearMemory< VolTreeBVHNodeStorageUINT >* )_nodePool;
	_kernelObject._volumeTreeBVHArray = nodesArrayGPUStoragePtr->getDeviceArray();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< class DataTList >
BvhTree<DataTList>::~BvhTree()
{
	delete _nodePool;
	delete _dataPool;
	//delete vertexPosArrayGPU;
}

/******************************************************************************
 * CUDA initialization
 ******************************************************************************/
template< class DataTList >
void BvhTree< DataTList >::cuda_Init()
{
	volumeTreeBVHTexLinear.normalized = false;					  // access with normalized texture coordinates
	volumeTreeBVHTexLinear.filterMode = cudaFilterModePoint;		// nearest interpolation
	volumeTreeBVHTexLinear.addressMode[ 0 ] = cudaAddressModeClamp;   // wrap texture coordinates
	volumeTreeBVHTexLinear.addressMode[ 1 ] = cudaAddressModeClamp;
	volumeTreeBVHTexLinear.addressMode[ 2 ] = cudaAddressModeClamp;

	GS_CUDA_SAFE_CALL( cudaBindTexture( NULL, volumeTreeBVHTexLinear, _nodePool->getPointer() ) );

	GV_CHECK_CUDA_ERROR( "BvhTree::cuda_Init end" );
}

/******************************************************************************
 * Initialize the cache
 *
 * @param pBvhTrianglesManager Helper class that store the node and data pools from a mesh
 ******************************************************************************/
template< class DataTList >
void BvhTree< DataTList >::initCache( BVHTrianglesManager< DataTList, BVH_DATA_PAGE_SIZE >* bvhTrianglesManager )
{
	// TODO : Stream Me \o/

	// Initialize the node buffer
	memcpyArray( _nodePool, (VolTreeBVHNodeUser*)( bvhTrianglesManager->getNodesBuffer()->getPointer() ), bvhTrianglesManager->getNodesBuffer()->getResolution().x );
	
	//memcpyArray( _dataPool->getChannel( Loki::Int2Type< 0 >() ), bvhTrianglesManager->getDataBuffer()->getChannel( Loki::Int2Type< 0 >() )->getPointer( 0 ) );
}

/******************************************************************************
 * Get the associated device-side object
 ******************************************************************************/
template< class DataTList >
inline BvhTree< DataTList >::BvhTreeKernelType BvhTree< DataTList >::getKernelObject()
{
	return _kernelObject;
}

/******************************************************************************
 * Clear
 ******************************************************************************/
template< class DataTList >
void BvhTree<DataTList>::clear()
{
	/*((GsLinearMemory<uint>*)volTreeChildArrayGPU )->fill(0);
	((GsLinearMemory<uint>*)volTreeDataArrayGPU )->fill(0);*/
}

/******************************************************************************
 * This method is called to serialize an object
 *
 * @param pStream the stream where to write
 ******************************************************************************/
template< class DataTList >
inline void BvhTree< DataTList >
::write( std::ostream& pStream ) const
{
	// TO DO
	// ...
}

/******************************************************************************
 * This method is called deserialize an object
 *
 * @param pStream the stream from which to read
 ******************************************************************************/
template< class DataTList >
inline void BvhTree< DataTList >
::read( std::istream& pStream )
{
	// TO DO
	// ...
}
