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
 * @param bvhTree BVH tree
 * @param gpuprod producer
 * @param voltreepoolres ...
 * @param nodetileres nodetile resolution
 * @param brickpoolres brick pool resolution
 * @param brickRes brick resolution
 ******************************************************************************/
template< typename BvhTreeType >
BvhTreeCache< BvhTreeType >
::BvhTreeCache( BvhTreeType* bvhTree, uint3 voltreepoolres, uint3 nodetileres, uint3 brickpoolres, uint3 brickRes )
:	GvStructure::GsIDataProductionManager()
{
	_bvhTree		= bvhTree;

	_nodePoolRes	= make_uint3( voltreepoolres.x * voltreepoolres.y * voltreepoolres.z, 1, 1);
	_brickPoolRes	= brickpoolres;

	// Node cache initialization
	nodesCacheManager = new NodesCacheManager( _nodePoolRes, nodetileres );
	//nodesCacheManager->setProvider( gpuprod );

	// Brick cache initialization
	bricksCacheManager = new BricksCacheManager( _brickPoolRes, brickRes );
	//bricksCacheManager->setProvider( gpuprod );

	// Request buffer initialization
	d_UpdateBufferArray			= new GvCore::GsLinearMemory< uint >( _nodePoolRes );
	d_UpdateBufferCompactList	= new thrust::device_vector< uint >( _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z );

	totalNumBricksLoaded = 0;
}

/******************************************************************************
 * Pre-render pass
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::preRenderPass()
{
	CUDAPM_START_EVENT( gpucache_preRenderPass );

	// Clear subdiv pool
	d_UpdateBufferArray->fill( 0 );

	updateSymbols();

	CUDAPM_STOP_EVENT( gpucache_preRenderPass );
}

/******************************************************************************
 * Post-render pass
 ******************************************************************************/
template< typename BvhTreeType >
uint BvhTreeCache< BvhTreeType >
::handleRequests()
{
	//updateSymbols();

	CUDAPM_START_EVENT( cache_updateTimestamps );
	updateTimeStamps();
	CUDAPM_STOP_EVENT( cache_updateTimestamps );

	updateSymbols();

	// Collect and compact update informations for both octree and bricks
	CUDAPM_START_EVENT( dataProduction_manageRequests );
	uint numUpdateElems = manageUpdates();
	CUDAPM_STOP_EVENT( dataProduction_manageRequests );

	// Manage the node subdivision requests 
	CUDAPM_START_EVENT( producer_nodes );
	uint numSubDiv = manageSubDivisions( numUpdateElems );
	CUDAPM_STOP_EVENT( producer_nodes );
	//std::cout << "numSubDiv: "<< numSubDiv << "\n";

	// Manage the brick load/produce requests
	CUDAPM_START_EVENT( producer_bricks );
	uint numBrickLoad = 0;
	if ( numSubDiv < numUpdateElems )
	{
		numBrickLoad = manageDataLoadGPUProd( numUpdateElems );
	}
	CUDAPM_STOP_EVENT( producer_bricks );

	//std::cout << "Cache num elems updated: " << numSubDiv + numBrickLoad <<"\n";

	return numSubDiv + numBrickLoad;
}

/******************************************************************************
 * Update all needed symbols in constant memory
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::updateSymbols()
{
	CUDAPM_START_EVENT( gpucache_updateSymbols );

	// Update node cache manager's symbols in constant memory
	nodesCacheManager->updateSymbols();

	// Update brick manager's symbols in constant memory
	bricksCacheManager->updateSymbols();

	// Copy node tile's time stamp buffer
	// Linux : taking address of temporary is error...
	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_NTTimeStampArray, ( &nodesCacheManager->getdTimeStampArray()->getDeviceArray() ), sizeof( nodesCacheManager->getdTimeStampArray()->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );
	GvCore::GsLinearMemoryKernel< uint > nodeCacheArrayDevice = nodesCacheManager->getdTimeStampArray()->getDeviceArray();
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_NTTimeStampArray, &nodeCacheArrayDevice, sizeof( nodesCacheManager->getdTimeStampArray()->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );

	// Copy brick's time stamp buffer
	// Linux : taking address of temporary is error...
	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_BTimeStampArray,	( &bricksCacheManager->getdTimeStampArray()->getDeviceArray() ), sizeof( bricksCacheManager->getdTimeStampArray()->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );
	GvCore::GsLinearMemoryKernel< uint > brickCacheArrayDevice = bricksCacheManager->getdTimeStampArray()->getDeviceArray();
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_BTimeStampArray, &brickCacheArrayDevice, sizeof( bricksCacheManager->getdTimeStampArray()->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );

	// Unified
	//
	// Copy request buffer
	// Linux : taking address of temporary is error...
	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_UpdateBufferArray,	( &(d_UpdateBufferArray->getDeviceArray()) ), sizeof( d_UpdateBufferArray->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );
	GvCore::GsLinearMemoryKernel< uint > requestBufferArray = d_UpdateBufferArray->getDeviceArray();
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_UpdateBufferArray,	&requestBufferArray, sizeof( d_UpdateBufferArray->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );

	CUDAPM_STOP_EVENT( gpucache_updateSymbols );
}

/******************************************************************************
 * Update time stamps
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::updateTimeStamps()
{
	CUDAPM_START_EVENT( cache_updateTimestamps_dataStructure );
	nodesCacheManager->updateSymbols();
	numNodeTilesNotInUse = nodesCacheManager->updateTimeStamps();
	CUDAPM_STOP_EVENT( cache_updateTimestamps_dataStructure );

	CUDAPM_START_EVENT( cache_updateTimestamps_bricks );
	bricksCacheManager->updateSymbols();
	numBricksNotInUse = bricksCacheManager->updateTimeStamps();
	CUDAPM_STOP_EVENT( cache_updateTimestamps_bricks );
}

/******************************************************************************
 * Manage updates
 *
 * @return ...
 ******************************************************************************/
template< typename BvhTreeType >
uint BvhTreeCache< BvhTreeType >
::manageUpdates()
{
	uint totalNumElems = _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z;

	uint numElems = 0;

	CUDAPM_START_EVENT( dataProduction_manageRequests_elemsReduction );

	// Copy current requests and return their total number
	numElems = thrust::copy_if(
		/*input first element*/thrust::device_ptr< uint >( d_UpdateBufferArray->getPointer( 0 ) ),
		/*input last element*/thrust::device_ptr< uint >( d_UpdateBufferArray->getPointer( 0 ) ) + totalNumElems,
		/*output result*/d_UpdateBufferCompactList->begin(), /*predicate*/GvCore::not_equal_to_zero< uint >() ) - d_UpdateBufferCompactList->begin();

	CUDAPM_STOP_EVENT( dataProduction_manageRequests_elemsReduction );

	return numElems;
}

/******************************************************************************
 * Manage the node subdivision requests
 *
 * @param pNumUpdateElems number of elements to process
 *
 * @return ...
 ******************************************************************************/
template< typename BvhTreeType >
uint BvhTreeCache< BvhTreeType >
::manageSubDivisions( uint pNumUpdateElems )
{
	// Buffer of requests
	uint* updateCompactList = thrust::raw_pointer_cast( &(*d_UpdateBufferCompactList)[ 0 ] );
	// numValidNodes = ( nodesCacheManager->totalNumLoads ) * NodeTileRes::getNumElements();

	// Ask node cache maanger to handle "node subdivision" requests
	assert( _producers.size() > 0 );
	assert( _producers[ 0 ] != NULL );
	ProducerType* producer = _producers[ 0 ];
	return nodesCacheManager->genericWrite( updateCompactList, pNumUpdateElems,
							/*mask of node subdivision request*/0x40000000U, 5000, 0xffffffffU/*numValidNodes*/, _bvhTree->_nodePool, producer );
}

/******************************************************************************
 * Manage the brick load/produce requests
 *
 * @param pNumUpdateElems number of elements to process
 *
 * @return ...
 ******************************************************************************/
// Suppose that the subdivision set the node type
template< typename BvhTreeType >
uint BvhTreeCache< BvhTreeType >
::manageDataLoadGPUProd( uint pNumUpdateElems )
{
	// Buffer of requests
	uint* updateCompactList = thrust::raw_pointer_cast( &(*d_UpdateBufferCompactList)[ 0 ] );
	//uint numValidNodes = ( nodesCacheManager->totalNumLoads ) * NodeTileRes::getNumElements();

	// Ask brick cache maanger to handle "brick load/produce" requests
	assert( _producers.size() > 0 );
	assert( _producers[ 0 ] != NULL );
	ProducerType* producer = _producers[ 0 ];
	return bricksCacheManager->genericWrite( updateCompactList, pNumUpdateElems,
		/*mask of brick load/production request*/0x80000000U, 5000, 0xffffffffU/*numValidNodes*/, _bvhTree->_dataPool, producer );
}

/******************************************************************************
 * Add a producer
 *
 * @param pProducer the producer to add
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::addProducer( ProducerType* pProducer )
{
	assert( pProducer != NULL );
	
	// TO DO
	// ...
	_producers.push_back( pProducer );
}

/******************************************************************************
 * Remove a producer
 *
 * @param pProducer the producer to remove
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::removeProducer( ProducerType* pProducer )
{
	assert( pProducer != NULL );

	// TO DO
	// ...
	assert( false );
}

/******************************************************************************
 * This method is called to serialize an object
 *
 * @param pStream the stream where to write
 ******************************************************************************/
template< typename BvhTreeType >
inline void BvhTreeCache< BvhTreeType >
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
template< typename BvhTreeType >
inline void BvhTreeCache< BvhTreeType >
::read( std::istream& pStream )
{
	// TO DO
	// ...
}
