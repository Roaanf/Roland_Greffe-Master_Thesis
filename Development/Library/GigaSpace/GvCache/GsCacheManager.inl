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
#include "GvCore/GsError.h"
// #include "GvCache/GsCacheManagementKernel.h"

// System
#include <cassert>

// STL
#include <algorithm>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

namespace GvCache
{

	/**
	 * In the GigaSpace engine, Cache management requires to :
	 * - protect "null" reference (element address)
	 * - root nodes in the data structure (i.e. octree, etc...)
	 * So, each array managed by the Cache needs to take care of these particular elements.
	 *
	 * Note : still a bug when too much loading - TODO: check this
	 */
	template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
	const uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >::_cNbLockedElements = 1/*null reference*/ + 1/*root node*/;

} // namespace GvCache

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCache
{

/******************************************************************************
 * Constructor
 *
 * @param pCachesize size of the cache
 * @param pPageTableArray the array of elements that the cache has to managed
 * @param pGraphicsInteroperability a flag used to map buffers to OpenGL graphics library
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::GsCacheManager( const uint3& pCachesize, PageTableArrayType* pPageTableArray, uint pGraphicsInteroperability )
:	GvCore::GsISerializable()
,	_cacheSize( pCachesize )
,	_policy( eDefaultPolicy )
,	_exceededCapacity( false )
{
	// Compute elements cache size
	_elemsCacheSize = _cacheSize / ElementRes::get();

	// Page table initialization
	_d_pageTableArray = pPageTableArray;
	_pageTable = new PageTableType();

	// Initialize the timestamp buffer
	_d_TimeStampArray = new GvCore::GsLinearMemory< uint >( _elemsCacheSize, pGraphicsInteroperability );
	_d_TimeStampArray->fill( 0 );

	this->_numElements = _elemsCacheSize.x * _elemsCacheSize.y * _elemsCacheSize.z - _cNbLockedElements;

	_numElemsNotUsed = _numElements;

	// List of elements in the cache
	_d_elemAddressList = new thrust::device_vector< uint >( this->_numElements );
	_d_elemAddressListTmp = new thrust::device_vector< uint >( this->_numElements );

	// Buffer of usage (masks of non-used and used elements in the current frame)
	_d_TempMaskList = GsCacheManagerResources::getTempUsageMask1( static_cast< size_t >( this->_numElements ) );
	_d_TempMaskList2 = GsCacheManagerResources::getTempUsageMask2( static_cast< size_t >( this->_numElements ) );

	thrust::fill( _d_TempMaskList->begin(), _d_TempMaskList->end(), 0 );
	thrust::fill( _d_TempMaskList2->begin(), _d_TempMaskList2->end(), 0 );

	uint3 pageTableRes = _d_pageTableArray->getResolution();
	uint pageTableResLinear = pageTableRes.x * pageTableRes.y * pageTableRes.z;

	// Buffer of requests (masks of associated request in the current frame)
	//_d_TempUpdateMaskList = GsCacheManagerResources::getTempUsageMask1( pageTableRes.x * pageTableRes.y  *pageTableRes.z );
	_d_TempUpdateMaskList = new thrust::device_vector< uint >( pageTableResLinear );
	_d_UpdateCompactList = new thrust::device_vector< uint >( pageTableResLinear );

#if CUDAPERFMON_CACHE_INFO==1
	_d_CacheStateBufferArray = new GsLinearMemory< uchar4 >( make_uint3( this->_numElements, 1, 1 ) );
#endif

	// Init
	//
	// Allocate size to speed up insert without need for push_back
	thrust::host_vector< uint > tmpelemaddress( _elemsCacheSize.x * _elemsCacheSize.y * _elemsCacheSize.z );
	uint3 pos;
	uint index = 0;
	for ( pos.z = 0; pos.z < _elemsCacheSize.z; pos.z++ )
	for ( pos.y = 0; pos.y < _elemsCacheSize.y; pos.y++ )
	for ( pos.x = 0; pos.x < _elemsCacheSize.x; pos.x++ )
	{
		tmpelemaddress[ index ] = AddressType::packAddress( pos );
		index++;
	}
	// Dont use element zero !
	thrust::copy( /*first iterator*/tmpelemaddress.begin() + _cNbLockedElements, /*last iterator*/tmpelemaddress.end(), /*output iterator*/_d_elemAddressList->begin() );

	uint cudppNumElem = std::max( pageTableRes.x * pageTableRes.y * pageTableRes.z, this->_numElements );
	_scanplan = GsCacheManagerResources::getScanPlan( cudppNumElem );

	GS_CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_numElementsPtr, sizeof( size_t ) ) );

	// The associated GPU side object receive a reference on the timestamp buffer
	_d_cacheManagerKernel._timeStampArray = _d_TimeStampArray->getDeviceArray();

#if GPUCACHE_BENCH_CPULRU==1
	_cpuTimeStampArray = new Array3D< uint >(  _d_TimeStampArray->getResolution() );
	_cpuTimeStampsElemAddressList = new thrust::host_vector< uint >( this->_numElements );
	_cpuTimeStampsElemAddressList2 = new thrust::host_vector< uint >( this->_numElements );

	thrust::copy( tmpelemaddress.begin() + numLockedElements, tmpelemaddress.end(), _cpuTimeStampsElemAddressList->begin() );
	thrust::copy( tmpelemaddress.begin() + numLockedElements, tmpelemaddress.end(), _cpuTimeStampsElemAddressList2->begin() );
#endif

	// LOG info
	std::cout << "\nCache Manager [ id " << Id::value << " ]" << std::endl;
	std::cout << "- cache size : " << _cacheSize << std::endl;
	std::cout << "- elements cache size : " << _elemsCacheSize << std::endl;
	std::cout << "- nb elements : " << this->_numElements << std::endl;
	std::cout << "- page table's linear resolution : " << pageTableResLinear << std::endl;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::~GsCacheManager()
{
	GS_CUDA_SAFE_CALL( cudaFree( _d_numElementsPtr ) );

	// TO DO
	// Attention, il arrive que ce "plan" soir partager entre les deux caches de brick et de noeuds
	// et lors de la destruction du 2�me cache manager, cela produit une erreur CUDPP.
	// ...
	// TO DO
	// Move this in another place
	/*CUDPPResult result = cudppDestroyPlan( _scanplan );
	if ( CUDPP_SUCCESS != result )
	{
		printf( "Error destroying CUDPPPlan\n" );
		exit( -1 );
	}*/


	// TO DO
	// MEMORY LEAK
	// Il y a des "new" dans le constructeur, il faut ajouter "delete" dans le destructeur...
	// ...

	// Page table
	delete _pageTable;
	_pageTable = NULL;

	delete _d_TimeStampArray;
	_d_TimeStampArray = NULL;

	delete _d_elemAddressList;
	_d_elemAddressList = NULL;
	delete _d_elemAddressListTmp;
	_d_elemAddressListTmp = NULL;

	delete _d_TempUpdateMaskList;
	_d_TempUpdateMaskList = NULL;
	delete _d_UpdateCompactList;
	_d_UpdateCompactList = NULL;
}

/******************************************************************************
 * Get the associated device side object
 *
 * @return the associated device side object
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
GsCacheManagerKernel< ElementRes, AddressType > GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::getKernelObject()
{
	return	_d_cacheManagerKernel;
}

/******************************************************************************
 * Set the cache policy
 *
 * @param pPolicy the cache policy
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
void GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::setPolicy( GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >::ECachePolicy pPolicy )
{
	this->_policy = pPolicy;
}

/******************************************************************************
 * Get the cache policy
 *
 * @return the cache policy
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >::ECachePolicy GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::getPolicy() const
{
	return this->_policy;
}

/******************************************************************************
 * Get the number of elements managed by the cache.
 *
 * @return
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::getNumElements() const
{
	return this->_numElements;
}

/******************************************************************************
 * Get the number of elements managed by the cache.
 *
 * @return
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::getNbUnusedElements() const
{
	return this->_numElemsNotUsed;
}

/******************************************************************************
 * Get the timestamp list of the cache.
 * There is as many timestamps as elements in the cache.
 *
 * @return ...
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
GvCore::GsLinearMemory< uint >* GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::getTimeStampList() const
{
	return _d_TimeStampArray;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
thrust::device_vector< uint >* GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::getElementList() const
{
	return _d_elemAddressList;
}

/******************************************************************************
 * Update symbols
 * (variables in constant memory)
 ******************************************************************************/
//template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
//void GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >::updateSymbols(){
//
//	CUDAPM_START_EVENT(gpucachemgr_updateSymbols);
//
//	/*GS_CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_VTC_TimeStampArray,
//		(&_d_TimeStampArray->getDeviceArray()),
//		sizeof(_d_TimeStampArray->getDeviceArray()), 0, cudaMemcpyHostToDevice));*/
//
//	CUDAPM_STOP_EVENT(gpucachemgr_updateSymbols);
//}

/******************************************************************************
 * Clear the cache
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
void GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::clearCache()
{
	// FIXME: do we really need to do it with a cuda kernel ?
	// Anyway, need to take account for the locked elements.
	//
	// Allocate size to speed up insert without need for push_back
	thrust::host_vector< uint > tmpelemaddress( _elemsCacheSize.x * _elemsCacheSize.y * _elemsCacheSize.z );
	uint3 pos;
	uint index = 0;
	for ( pos.z = 0; pos.z < _elemsCacheSize.z; pos.z++ )
	for ( pos.y = 0; pos.y < _elemsCacheSize.y; pos.y++ )
	for ( pos.x = 0; pos.x < _elemsCacheSize.x; pos.x++ )
	{
		tmpelemaddress[ index ] = AddressType::packAddress( pos );
		index++;
	}
	// Don't use element zero !
	thrust::copy( tmpelemaddress.begin() + _cNbLockedElements, tmpelemaddress.end(), /*output*/_d_elemAddressList->begin() );

	// Init
	//CUDAPM_START_EVENT( gpucachemgr_clear_cpyAddr )

	//// Uses kernel filling
	//uint* timeStampsElemAddressListPtr = thrust::raw_pointer_cast( &(*_d_elemAddressList)[ 0 ] );

	//const dim3 blockSize( 128, 1, 1 );
	//uint numBlocks = iDivUp( this->_numElements, blockSize.x );
	//dim3 gridSize = dim3( std::min( numBlocks, 65535U ), iDivUp( numBlocks, 65535U ), 1 );

	//InitElemAddressList< AddressType ><<< gridSize, blockSize, 0 >>>( timeStampsElemAddressListPtr, this->_numElements, _elemsCacheSize );
	//GV_CHECK_CUDA_ERROR( "InitElemAddressList" );

	//CUDAPM_STOP_EVENT( gpucachemgr_clear_cpyAddr )

	// Clear buffers holding mask of used and non-used elements
	CUDAPM_START_EVENT( gpucachemgr_clear_fillML )
	thrust::fill( _d_TempMaskList->begin(), _d_TempMaskList->end(), 0 );
	thrust::fill( _d_TempMaskList2->begin(), _d_TempMaskList2->end(), 0 );
	CUDAPM_STOP_EVENT( gpucachemgr_clear_fillML )

	// Clear the time-stamp buffer
	CUDAPM_START_EVENT( gpucachemgr_clear_fillTimeStamp )
	_d_TimeStampArray->fill( 0 );
	CUDAPM_STOP_EVENT( gpucachemgr_clear_fillTimeStamp )

	// Reset flag
	_exceededCapacity = false;
}

#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_GVCACHEMANAGER
/******************************************************************************
 * Update the list of available elements according to their timestamps.
 * Unused and recycled elements will be placed first.
 *
 * @param manageUpdatesOnly ...
 *
 * @return the number of available elements
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::updateTimeStamps( bool manageUpdatesOnly )
{
	if ( true || this->_lastNumLoads > 0 )
	{
 		// uint numElemsNotUsed = 0;
		_numElemsNotUsed = 0;

		uint cacheNumElems = getNumElements();

		// std:cout << "manageUpdatesOnly "<< (int)manageUpdatesOnly << "\n";

		uint activeNumElems = cacheNumElems;
		// TODO: re-enable manageUpdatesOnly !
		/*if ( manageUpdatesOnly )
		{
			activeNumElems = this->_lastNumLoads;
		}*/
		uint inactiveNumElems = cacheNumElems - activeNumElems;

		const uint nbElemToSort = activeNumElems;
		if ( nbElemToSort > 0 )
		{
			uint sortingStartPos = 0;

#if GPUCACHE_BENCH_CPULRU==0

			// ---- [ 1 ] ---- 1st step
			//
			// Find "used" and "unused" elements during current frame in one single pass

			CUDAPM_START_EVENT( cache_updateTimestamps_createMasks );

			// Create masks in a single pass
			dim3 blockSize( 64, 1, 1 );
			//dim3 blockSize( 128, 1, 1 );
			uint numBlocks = iDivUp( nbElemToSort, blockSize.x );
			dim3 gridSize = dim3( std::min( numBlocks, 65535U ) , iDivUp( numBlocks, 65535U ), 1 );

			// This kernel creates the usage mask list of used and non-used elements (in current rendering pass) in a single pass
			//
			// Note : Generate an error with CUDA 3.2
			GsKernel_CacheManager_retrieveElementUsageMasks< ElementRes, AddressType >
				<<< gridSize, blockSize, 0 >>>( /*in*/_d_cacheManagerKernel,
						/*in*/nbElemToSort,
						/*in*/thrust::raw_pointer_cast( &(*_d_elemAddressList)[ sortingStartPos ] ),
						/*out*/thrust::raw_pointer_cast( &(*_d_TempMaskList)[ 0 ] )		/*resulting mask list of non-used elements*/,
						/*out*/thrust::raw_pointer_cast( &(*_d_TempMaskList2)[ 0 ] )	/*resulting mask list of used elements*/ );
			GV_CHECK_CUDA_ERROR( "GsKernel_CacheManager_retrieveElementUsageMasks" );

			//const uint cacheId = Id::value;
			//if ( cacheId == 1 )
			//{
			//	//std::cout << "GsKernel_CacheManager_retrieveElementUsageMasks" << std::endl;
			//
			//	GvKernel_DataCacheManagerFlagTimeStamps<<< gridSize, blockSize >>>( nbElemToSort,
			//											thrust::raw_pointer_cast( &(*_d_elemAddressList)[ sortingStartPos ] ),
			//											_d_TimeStampArray->getPointer(),
			//											thrust::raw_pointer_cast( &(*_d_TempMaskList)[ 0 ] ),
			//											thrust::raw_pointer_cast( &(*_d_TempMaskList2)[ 0 ] ),
			//											_d_TimeStampArray->getResolution().x, _d_TimeStampArray->getResolution().x * _d_TimeStampArray->getResolution().y );
			//	GV_CHECK_CUDA_ERROR( "GvKernel_DataCacheManagerFlagTimeStamps" );
			//}
			//else if ( cacheId == 0 )
			//{
			//	//std::cout << "GvKernel_NodeCacheManagerFlagTimeStamps" << std::endl;
			//
			//	GvKernel_NodeCacheManagerFlagTimeStamps<<< gridSize, blockSize >>>( nbElemToSort,
			//											thrust::raw_pointer_cast( &(*_d_elemAddressList)[ sortingStartPos ] ),
			//											_d_TimeStampArray->getPointer(),
			//											thrust::raw_pointer_cast( &(*_d_TempMaskList)[ 0 ] ),
			//											thrust::raw_pointer_cast( &(*_d_TempMaskList2)[ 0 ] ) );
			//	GV_CHECK_CUDA_ERROR( "GvKernel_NodeCacheManagerFlagTimeStamps" );
			//}
			//else
			//{
			//	assert( false );
			//	std::cout << "Error" << std::endl;
			//}

			CUDAPM_STOP_EVENT( cache_updateTimestamps_createMasks );

			// ---- [ 2 ] ---- 2nd step
			//
			// ...

			thrust::device_vector< uint >::const_iterator elemAddressListFirst = _d_elemAddressList->begin();
			thrust::device_vector< uint >::const_iterator elemAddressListLast = _d_elemAddressList->begin() + nbElemToSort;
			thrust::device_vector< uint >::iterator elemAddressListTmpFirst = _d_elemAddressListTmp->begin();

			// Stream compaction to collect non-used elements at the beginning
			CUDAPM_START_EVENT( cache_updateTimestamps_threadReduc1 );

			// Given an array d_in and an array of 1/0 flags in deviceValid, returns a compacted array in d_out of corresponding only the "valid" values from d_in.
			//
			// Takes as input an array of elements in GPU memory (d_in) and an equal-sized unsigned int array in GPU memory (deviceValid) that indicate which of those input elements are valid.
			// The output is a packed array, in GPU memory, of only those elements marked as valid.
			//
			// Internally, uses cudppScan.
			//
			// Algorithme : WRITE in temporary buffer _d_elemAddressListTmp, the compacted list of NON-USED elements from _d_elemAddressList
			// - number of un-used is returned in _d_numElementsPtr
			// - elements are written at the beginning of the array
			cudppCompact( /*handle to CUDPPCompactPlan*/_scanplan,
				/* OUT : compacted output */thrust::raw_pointer_cast( &(*_d_elemAddressListTmp)[ inactiveNumElems ] ),
				/* OUT :  number of elements valid flags in the d_isValid input array */_d_numElementsPtr,
				/* input to compact */thrust::raw_pointer_cast( &(*_d_elemAddressList)[ sortingStartPos ] ),
				/* which elements in input are valid */thrust::raw_pointer_cast( &(*_d_TempMaskList)[ 0 ] ),	/*non-used elements*/
				/* nb of elements in input */nbElemToSort );

			// ---- [ 3 ] ---- 3rd step
			//
			// ...

			// Get number of non-used elements
			size_t numElemsNotUsedST;
			GS_CUDA_SAFE_CALL( cudaMemcpy( &numElemsNotUsedST, _d_numElementsPtr, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
			_numElemsNotUsed = (uint)numElemsNotUsedST + inactiveNumElems;

			//--------------------------------------------------------
			// TO DO
			//
			// - optimization : if _numElemsNotUsed is equal to max elements in cache, exit o avoid launching all following kernels
			//--------------------------------------------------------

			CUDAPM_STOP_EVENT( cache_updateTimestamps_threadReduc1 );

			// ---- [ 4 ] ---- 4th step
			//
			// ...

			// Stream compaction to collect used elements at the end
			CUDAPM_START_EVENT( cache_updateTimestamps_threadReduc2 );

			// Given an array d_in and an array of 1/0 flags in deviceValid, returns a compacted array in d_out of corresponding only the "valid" values from d_in.
			//
			// Takes as input an array of elements in GPU memory (d_in) and an equal-sized unsigned int array in GPU memory (deviceValid) that indicate which of those input elements are valid.
			// The output is a packed array, in GPU memory, of only those elements marked as valid.
			//
			// Internally, uses cudppScan.
			//
			// Algorithme : WRITE in temporary buffer _d_elemAddressListTmp, the compacted list of USED elements from _d_elemAddressList
			// - elements are written at the end of the array, i.e. after the previous NON-USED ones
			cudppCompact( /*handle to CUDPPCompactPlan*/_scanplan,
				/* OUT : compacted output */thrust::raw_pointer_cast( &(*_d_elemAddressListTmp)[ _numElemsNotUsed ] ),
				/* OUT :  number of elements valid flags in the d_isValid input array */_d_numElementsPtr,
				/* input to compact */thrust::raw_pointer_cast( &(*_d_elemAddressList)[ sortingStartPos ] ),
				/* which elements in input are valid */thrust::raw_pointer_cast( &(*_d_TempMaskList2)[ 0 ] ),
				/* nb of elements in input */nbElemToSort );

			CUDAPM_STOP_EVENT( cache_updateTimestamps_threadReduc2 );

#else
			memcpyArray( _cpuTimeStampArray, _d_TimeStampArray );

			uint curDstPos = 0;

			CUDAPM_START_EVENT( gpucachemgr_updateTimeStampsCPU );
			// Copy unused
			for ( uint i = 0; i < cacheNumElems; ++i )
			{
				uint elemAddressEnc = (*_cpuTimeStampsElemAddressList)[ i ];
				uint3 elemAddress;
				elemAddress = AddressType::unpackAddress( elemAddressEnc );

				if ( _cpuTimeStampArray->get( elemAddress ) != GsCacheManager_currentTime )
				{
					(*_cpuTimeStampsElemAddressList2)[ curDstPos ] = elemAddressEnc;
					curDstPos++;
				}
			}
			_numElemsNotUsed = curDstPos;

			// Copy used
			for ( uint i = 0; i < cacheNumElems; ++i )
			{
				uint elemAddressEnc = (*_cpuTimeStampsElemAddressList)[ i ];
				uint3 elemAddress;
				elemAddress = AddressType::unpackAddress( elemAddressEnc );

				if ( _cpuTimeStampArray->get( elemAddress ) == GsCacheManager_currentTime )
				{
					(*_cpuTimeStampsElemAddressList2)[ curDstPos ] = elemAddressEnc;
					curDstPos++;
				}
			}
			CUDAPM_STOP_EVENT( gpucachemgr_updateTimeStampsCPU );

			thrust::copy( _cpuTimeStampsElemAddressList2->begin(), _cpuTimeStampsElemAddressList2->end(), _d_elemAddressListTmp->begin() );
#endif
		}
		else
		{
			_numElemsNotUsed = cacheNumElems;
		}

		// ---- [ 5 ] ---- 5th step
		//
		// ...

		// Swap buffers
		//
		// Temporary buffer "_d_elemAddressListTmp" where non-used elements are the beginning and used elements at the end,
		// is swapped with _d_elemAddressList
		thrust::device_vector< uint >* tmpl = _d_elemAddressList;
		_d_elemAddressList = _d_elemAddressListTmp;
		_d_elemAddressListTmp = tmpl;

#if CUDAPERFMON_CACHE_INFO==1
		{
			uint* usedPageList=thrust::raw_pointer_cast( &(*_d_elemAddressList)[ _numElemsNotUsed ] );

			uint numPageUsed = getNumElements() - _numElemsNotUsed;

			if ( numPageUsed > 0 )
			{
				dim3 blockSize( 128, 1, 1 );
				uint numBlocks = iDivUp( numPageUsed, blockSize.x );
				dim3 gridSize = dim3( std::min( numBlocks, 32768U ) , iDivUp( numBlocks,32768U ), 1 );

				SyntheticInfo_Update_PageUsed< ElementRes, AddressType >
					<<< gridSize, blockSize, 0 >>>(	_d_CacheStateBufferArray->getPointer(), numPageUsed, usedPageList, _elemsCacheSize );
				GV_CHECK_CUDA_ERROR( "SyntheticInfo_Update_PageUsed" );

				// Update counter
				_numPagesUsed = numPageUsed;
			}
		}
#endif
		this->_lastNumLoads = 0;
	}

	return _numElemsNotUsed;
}
#else // GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_GVCACHEMANAGER
/******************************************************************************
 * Update the list of available elements according to their timestamps.
 * Unused and recycled elements will be placed first.
 *
 * @param manageUpdatesOnly ...
 *
 * @return the number of available elements
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::updateTimeStamps( bool manageUpdatesOnly )
{
	const uint cacheId = Id::value;

	if ( true || this->_lastNumLoads > 0 )
	{
 		_numElemsNotUsed = 0;

		uint cacheNumElems = getNumElements();

		// std:cout << "manageUpdatesOnly "<< (int)manageUpdatesOnly << "\n";

		uint activeNumElems = cacheNumElems;
		// TODO: re-enable manageUpdatesOnly !
		/*if ( manageUpdatesOnly )
		{
			activeNumElems = this->_lastNumLoads;
		}*/
		uint inactiveNumElems = cacheNumElems - activeNumElems;

		const uint nbElemToSort = activeNumElems;
		if ( nbElemToSort > 0 )
		{
			uint sortingStartPos = 0;

			// ---- [ 1 ] ---- 1st step
			//
			// Find "used" and "unused" elements during current frame in one single pass

			CUDAPM_START_EVENT( cache_updateTimestamps_createMasks );

			// Create masks in a single pass
			dim3 blockSize( 64, 1, 1 );
			//dim3 blockSize( 128, 1, 1 );
			uint numBlocks = iDivUp( nbElemToSort, blockSize.x );
			dim3 gridSize = dim3( std::min( numBlocks, 65535U ) , iDivUp( numBlocks, 65535U ), 1 );

			// This kernel creates the usage mask list of used and non-used elements (in current rendering pass) in a single pass
			//
			// Note : Generate an error with CUDA 3.2
			GsKernel_CacheManager_retrieveElementUsageMasks< ElementRes, AddressType >
				<<< gridSize, blockSize, 0 >>>( /*in*/_d_cacheManagerKernel,
						/*in*/nbElemToSort,
						/*in*/thrust::raw_pointer_cast( &(*_d_elemAddressList)[ sortingStartPos ] ),
						/*out*/thrust::raw_pointer_cast( &(*_d_TempMaskList)[ 0 ] )		/*resulting mask list of non-used elements*/,
						/*out*/thrust::raw_pointer_cast( &(*_d_TempMaskList2)[ 0 ] )	/*resulting mask list of used elements*/ );
			GV_CHECK_CUDA_ERROR( "GsKernel_CacheManager_retrieveElementUsageMasks" );

			CUDAPM_STOP_EVENT( cache_updateTimestamps_createMasks );

			// ---- [ 2 ] ---- 2nd step
			//
			// ...

			thrust::device_vector< uint >::const_iterator elemAddressListFirst = _d_elemAddressList->begin();
			thrust::device_vector< uint >::const_iterator elemAddressListLast = _d_elemAddressList->begin() + nbElemToSort;
			thrust::device_vector< uint >::iterator elemAddressListTmpFirst = _d_elemAddressListTmp->begin();

			// Stream compaction to collect non-used elements at the beginning
			CUDAPM_START_EVENT( cache_updateTimestamps_threadReduc1 );

			// Given an array d_in and an array of 1/0 flags in deviceValid, returns a compacted array in d_out of corresponding only the "valid" values from d_in.
			//
			// Takes as input an array of elements in GPU memory (d_in) and an equal-sized unsigned int array in GPU memory (deviceValid) that indicate which of those input elements are valid.
			// The output is a packed array, in GPU memory, of only those elements marked as valid.
			//
			// Internally, uses cudppScan.
			//
			// Algorithmn : WRITE in temporary buffer _d_elemAddressListTmp, the compacted list of NON-USED elements from _d_elemAddressList
			// - number of un-used is returned in _d_numElementsPtr
			// - elements are written at the beginning of the array
			cudppCompact( /*handle to CUDPPCompactPlan*/_scanplan,
				/* OUT : compacted output */thrust::raw_pointer_cast( &(*_d_elemAddressListTmp)[ inactiveNumElems ] ),
	#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_WITH_COMBINED_COPY
				/* OUT :  number of elements valid flags in the d_isValid input array */_d_numElementsPtr,
	#else
				/* OUT :  number of elements valid flags in the d_isValid input array */_d_nbValidRequests + cacheId + 1,
	#endif
				/* input to compact */thrust::raw_pointer_cast( &(*_d_elemAddressList)[ sortingStartPos ] ),
				/* which elements in input are valid */thrust::raw_pointer_cast( &(*_d_TempMaskList)[ 0 ] ),	/*non-used elements*/
				/* nb of elements in input */nbElemToSort );

			CUDAPM_STOP_EVENT( cache_updateTimestamps_threadReduc1 );
		}
	}

	return _numElemsNotUsed;
}
/******************************************************************************
 * Update the list of available elements according to their timestamps.
 * Unused and recycled elements will be placed first.
 *
 * @param manageUpdatesOnly ...
 *
 * @return the number of available elements
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::updateTimeStampsCopy( bool manageUpdatesOnly )
{
	const uint cacheId = Id::value;

	if ( true || this->_lastNumLoads > 0 )
	{
 		_numElemsNotUsed = 0;

		uint cacheNumElems = getNumElements();

		uint activeNumElems = cacheNumElems;

		//uint inactiveNumElems = cacheNumElems - activeNumElems;

		const uint nbElemToSort = activeNumElems;
		if ( nbElemToSort > 0 )
		{
			// ---- [ 3 ] ---- 3rd step
			//
			// ...

			// Get number of non-used elements
	#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_WITH_COMBINED_COPY
			GS_CUDA_SAFE_CALL( cudaMemcpyAsync( &_numElemsNotUsedST, _d_numElementsPtr, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	#else
			GS_CUDA_SAFE_CALL( cudaMemcpyAsync( &_numElemsNotUsedST, _d_nbValidRequests + cacheId + 1, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	#endif
		}
		else
		{
			_numElemsNotUsed = cacheNumElems;
		}
	}

	return _numElemsNotUsed;
}
/******************************************************************************
 * Update the list of available elements according to their timestamps.
 * Unused and recycled elements will be placed first.
 *
 * @param manageUpdatesOnly ...
 *
 * @return the number of available elements
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::updateTimeStampsFinal( bool manageUpdatesOnly )
{
	const uint cacheId = Id::value;

	if ( true || this->_lastNumLoads > 0 )
	{
 		_numElemsNotUsed = 0;

		uint cacheNumElems = getNumElements();

		uint activeNumElems = cacheNumElems;

		uint inactiveNumElems = cacheNumElems - activeNumElems;

		const uint nbElemToSort = activeNumElems;
		if ( nbElemToSort > 0 )
		{
			uint sortingStartPos = 0;

			// ---- [ 2 ] ---- 2nd step
			//
			// ...

			thrust::device_vector< uint >::const_iterator elemAddressListFirst = _d_elemAddressList->begin();
			thrust::device_vector< uint >::const_iterator elemAddressListLast = _d_elemAddressList->begin() + nbElemToSort;
			thrust::device_vector< uint >::iterator elemAddressListTmpFirst = _d_elemAddressListTmp->begin();

			// Get number of non-used elements
			//
			// BEWARE : here, _numElemsNotUsedST MUST have been populated by cudaMemcpyAsync() => so, synchronization is mandatory
			_numElemsNotUsed = (uint)_numElemsNotUsedST + inactiveNumElems;

			//--------------------------------------------------------
			// TO DO
			//
			// - optimization : if _numElemsNotUsed is equal to max elements in cache, exit o avoid launching all following kernels
			//--------------------------------------------------------

			// ---- [ 4 ] ---- 4th step
			//
			// ...

			// Stream compaction to collect used elements at the end
			CUDAPM_START_EVENT( cache_updateTimestamps_threadReduc2 );

			// Given an array d_in and an array of 1/0 flags in deviceValid, returns a compacted array in d_out of corresponding only the "valid" values from d_in.
			//
			// Takes as input an array of elements in GPU memory (d_in) and an equal-sized unsigned int array in GPU memory (deviceValid) that indicate which of those input elements are valid.
			// The output is a packed array, in GPU memory, of only those elements marked as valid.
			//
			// Internally, uses cudppScan.
			//
			// Algorithme : WRITE in temporary buffer _d_elemAddressListTmp, the compacted list of USED elements from _d_elemAddressList
			// - elements are written at the end of the array, i.e. after the previous NON-USED ones
			cudppCompact( /*handle to CUDPPCompactPlan*/_scanplan,
				/* OUT : compacted output */thrust::raw_pointer_cast( &(*_d_elemAddressListTmp)[ _numElemsNotUsed ] ),
	#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_WITH_COMBINED_COPY
				/* OUT :  number of elements valid flags in the d_isValid input array */_d_numElementsPtr,
	#else
				/* OUT :  number of elements valid flags in the d_isValid input array */_d_nbValidRequests + cacheId + 1,
	#endif
				/* input to compact */thrust::raw_pointer_cast( &(*_d_elemAddressList)[ sortingStartPos ] ),
				/* which elements in input are valid */thrust::raw_pointer_cast( &(*_d_TempMaskList2)[ 0 ] ),
				/* nb of elements in input */nbElemToSort );

			CUDAPM_STOP_EVENT( cache_updateTimestamps_threadReduc2 );
		}
		else
		{
			_numElemsNotUsed = cacheNumElems;
		}

		// ---- [ 5 ] ---- 5th step
		//
		// ...

		// Swap buffers
		//
		// Temporary buffer "_d_elemAddressListTmp" where non-used elements are the beginning and used elements at the end,
		// is swapped with _d_elemAddressList
		thrust::device_vector< uint >* tmpl = _d_elemAddressList;
		_d_elemAddressList = _d_elemAddressListTmp;
		_d_elemAddressListTmp = tmpl;

		this->_lastNumLoads = 0;
	}

	return _numElemsNotUsed;
}
#endif

#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_PRODUCER
/******************************************************************************
 * Main method to launch the production of nodes or bricks
 *
 * @param updateList global buffer of requests of used elements only (node subdivision an brick load/produce)
 * @param numUpdateElems ...
 * @param updateMask Type of request to handle (node subdivision or brick load/produce)
 * @param maxNumElems Max number of elements to process
 * @param numValidNodes ...
 * @param gpuPool pool used to write new produced data inside (node pool or data pool)
 *
 * @return the number of produced elements
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
template< typename GPUPoolType, typename TProducerType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::genericWrite( uint* updateList, uint numUpdateElems, uint updateMask,
			   uint maxNumElems, uint numValidNodes, const GPUPoolType& gpuPool, TProducerType* pProducer )
{
	assert( pProducer != NULL );
	// TO DO : check pProducer at run-time

	uint numElems = 0;

	if ( numUpdateElems > 0 )
	{
		const uint cacheId = Id::value;

		// ---- [ 1 ] ---- 1st step
		//
		// Fill the buffer of mask of requests

		CUDAPM_START_EVENT_CHANNEL( 0, cacheId, gpucache_nodes_manageUpdates );

		// The "_d_UpdateCompactList" buffer is filled with only elements addresses that have a type of request equal to "updateMask"
		// - it returns the number of elements concerned
		numElems = createUpdateList( updateList, numUpdateElems, updateMask );

		CUDAPM_STOP_EVENT_CHANNEL( 0, cacheId, gpucache_nodes_manageUpdates );

		// Prevent loading more than the cache size
		numElems = std::min( numElems, getNumElements() );

		// Warning to prevent user that there are no more avalaible slots
		if ( numElems > _numElemsNotUsed )
		{
			// Update flag
			_exceededCapacity = true;

			// LOG info
			// TO DO : remove that cause it slows application (it is called at each frame)
			std::cout << "CacheManager< " << cacheId << " >: Warning: " << _numElemsNotUsed << " non-used slots available, but ask for : " << numElems << std::endl;
		}
		else
		{
			// Reset flag
			_exceededCapacity = false;
		}

		// Handle cache policy
		if ( this->_policy & ePreventReplacingUsedElementsPolicy )
		{
			// Prevent replacing elements in use
			numElems = std::min( numElems, _numElemsNotUsed );
		}
		if ( this->_policy & eSmoothLoadingPolicy )
		{
			// Smooth loading
			numElems = std::min( numElems, maxNumElems );
		}

		// Check if we have requests to handle
		if ( numElems > 0 )
		{
		//	std::cout << "CacheManager<" << cacheId << ">: " << numElems << " requests" << std::endl;

			// ---- [ 2 ] ---- 2nd step
			//
			// Invalidation phase

			// Update internal counter
			_totalNumLoads += numElems;
			_lastNumLoads = numElems;

		/*	std::cout << "\t_totalNumLoads : " << _totalNumLoads << std::endl;
			std::cout << "\t_lastNumLoads : " << _lastNumLoads << std::endl;
			std::cout << "\t_numElemsNotUsed : " << _numElemsNotUsed << std::endl;*/

			CUDAPM_START_EVENT_CHANNEL( 1, cacheId, gpucache_bricks_bricksInvalidation );

			invalidateElements( numElems, numValidNodes );		// WARNING !!!! numElems a �t� modifi� auparavant !!!! ===> ERREUR !!!!!!!!!!!!!!

			CUDAPM_STOP_EVENT_CHANNEL( 1, cacheId, gpucache_bricks_bricksInvalidation );

			// ---- [ 3 ] ---- 3rd step
			//
			// Write new elements into the cache

			CUDAPM_START_EVENT_CHANNEL( 0, cacheId, gpucache_nodes_subdivKernel );
			CUDAPM_START_EVENT_CHANNEL( 1, cacheId, gpucache_bricks_gpuFetchBricks );
			CUDAPM_EVENT_NUMELEMS_CHANNEL( 1, cacheId, gpucache_bricks_gpuFetchBricks, numElems );

			// Buffer of nodes addresses with their requests
			thrust::device_vector< uint >* nodesAddressCompactList = _d_UpdateCompactList;
			// Sorted buffer of elements (according to timestamps : "un-used ones" first then "used ones" after)
			thrust::device_vector< uint >* elemsAddressCompactList = _d_elemAddressList;

#if CUDAPERFMON_CACHE_INFO==1
			{
				dim3 blockSize( 64, 1, 1 );
				uint numBlocks = iDivUp( numElems, blockSize.x );
				dim3 gridSize = dim3( std::min( numBlocks, 32768U ) , iDivUp( numBlocks,32768U ), 1 );

				SyntheticInfo_Update_DataWrite< ElementRes, AddressType ><<< gridSize, blockSize, 0 >>>(
					_d_CacheStateBufferArray->getPointer(), numElems,
					thrust::raw_pointer_cast( &(*elemsAddressCompactList)[ 0 ] ),
					_elemsCacheSize );

				GV_CHECK_CUDA_ERROR( "SyntheticInfo_Update_DataWrite" );
			}

			_numPagesWrited = numElems;
#endif
			// Ask provider to produce data
			//
			// This method is called by the cache manager when you have to produce data for a given pool
			//
			// numElems : number of elements to produce
			// nodesAddressCompactList : list containing the addresses of the "numElems" nodes concerned
			// elemsAddressCompactList : list containing "numElems" addresses where store the result
			// gpuPool : pool for which we need to produce elements (node pol or data pool)
			// _pageTable : page table associated to the pool (used to retrieve node's localization info [code and depth])
			//_provider->template produceData< ElementRes >( numElems, nodesAddressCompactList, elemsAddressCompactList, gpuPool, _pageTable );
			pProducer->produceData( numElems, nodesAddressCompactList, elemsAddressCompactList, Id() );

			CUDAPM_STOP_EVENT_CHANNEL( 0, cacheId, gpucache_nodes_subdivKernel );
			CUDAPM_STOP_EVENT_CHANNEL( 1, cacheId, gpucache_bricks_gpuFetchBricks );
		}
	}

	return numElems;
}
#else
/******************************************************************************
 * Main method to launch the production of nodes or bricks
 *
 * @param updateList global buffer of requests of used elements only (node subdivision or brick load/produce)
 * @param numUpdateElems ...
 * @param updateMask Type of request to handle (node subdivision or brick load/produce)
 * @param maxNumElems Max number of elements to process
 * @param numValidNodes ...
 * @param gpuPool pool used to write new produced data inside (node pool or data pool)
 *
 * @return the number of produced elements
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
template< typename GPUPoolType, typename TProducerType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::genericWrite( uint* updateList, uint numUpdateElems, uint updateMask,
			   uint maxNumElems, uint numValidNodes, const GPUPoolType& gpuPool, TProducerType* pProducer )
{
	assert( pProducer != NULL );
	// TO DO : check pProducer at run-time

	uint numElems = 0;

	if ( numUpdateElems > 0 )
	{
		// ---- [ 1 ] ---- 1st step
		//
		// Fill the buffer of mask of requests

		CUDAPM_START_EVENT_CHANNEL( 0, Id::value, gpucache_nodes_manageUpdates );

		// The "_d_UpdateCompactList" buffer is filled with only elements addresses that have a type of request equal to "updateMask"
		// - it returns the number of elements concerned
		numElems = createUpdateList( updateList, numUpdateElems, updateMask );

		CUDAPM_STOP_EVENT_CHANNEL( 0, Id::value, gpucache_nodes_manageUpdates );
	}

	return numElems;
}
/******************************************************************************
 * Main method to launch the production of nodes or bricks
 *
 * @param updateList global buffer of requests of used elements only (node subdivision or brick load/produce)
 * @param numUpdateElems ...
 * @param updateMask Type of request to handle (node subdivision or brick load/produce)
 * @param maxNumElems Max number of elements to process
 * @param numValidNodes ...
 * @param gpuPool pool used to write new produced data inside (node pool or data pool)
 *
 * @return the number of produced elements
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
template< typename GPUPoolType, typename TProducerType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::genericWriteAsync( uint* updateList, uint numUpdateElems, uint updateMask,
			   uint maxNumElems, uint numValidNodes, const GPUPoolType& gpuPool, TProducerType* pProducer, uint pNb )
{
	assert( pProducer != NULL );
	// TO DO : check pProducer at run-time

	uint numElems = 0;

	if ( numUpdateElems > 0 )
	{
		const uint cacheId = Id::value;

		// ---- [ 1 ] ---- 1st step
		//
		// Fill the buffer of mask of requests

		CUDAPM_START_EVENT_CHANNEL( 0, cacheId, gpucache_nodes_manageUpdates );

		// The "_d_UpdateCompactList" buffer is filled with only elements addresses that have a type of request equal to "updateMask"
		// - it returns the number of elements concerned
		//numElems = createUpdateList( updateList, numUpdateElems, updateMask );
		numElems = pNb;

		CUDAPM_STOP_EVENT_CHANNEL( 0, cacheId, gpucache_nodes_manageUpdates );

		// Prevent loading more than the cache size
		numElems = std::min( numElems, getNumElements() );

		// Warning to prevent user that there are no more available slots
		if ( numElems > _numElemsNotUsed )
		{
			// Update flag
			_exceededCapacity = true;

			// LOG info
			// TO DO : remove that cause it slows application (it is called at each frame)
			std::cout << "CacheManager< " << cacheId << " >: Warning: " << _numElemsNotUsed << " non-used slots available, but ask for : " << numElems << std::endl;
		}
		else
		{
			// Reset flag
			_exceededCapacity = false;
		}

		// Handle cache policy
		if ( this->_policy & ePreventReplacingUsedElementsPolicy )
		{
			// Prevent replacing elements in use
			numElems = std::min( numElems, _numElemsNotUsed );
		}
		if ( this->_policy & eSmoothLoadingPolicy )
		{
			// Smooth loading
			numElems = std::min( numElems, maxNumElems );
		}

		// Check if we have requests to handle
		if ( numElems > 0 )
		{
		//	std::cout << "CacheManager<" << cacheId << ">: " << numElems << " requests" << std::endl;

			// ---- [ 2 ] ---- 2nd step
			//
			// Invalidation phase

			// Update internal counter
			_totalNumLoads += numElems;
			_lastNumLoads = numElems;

		/*	std::cout << "\t_totalNumLoads : " << _totalNumLoads << std::endl;
			std::cout << "\t_lastNumLoads : " << _lastNumLoads << std::endl;
			std::cout << "\t_numElemsNotUsed : " << _numElemsNotUsed << std::endl;*/

			CUDAPM_START_EVENT_CHANNEL( 1, cacheId, gpucache_bricks_bricksInvalidation );

			invalidateElements( numElems, numValidNodes );		// WARNING !!!! numElems a �t� modifi� auparavant !!!! ===> ERREUR !!!!!!!!!!!!!!

			CUDAPM_STOP_EVENT_CHANNEL( 1, cacheId, gpucache_bricks_bricksInvalidation );

			// ---- [ 3 ] ---- 3rd step
			//
			// Write new elements into the cache

			CUDAPM_START_EVENT_CHANNEL( 0, cacheId, gpucache_nodes_subdivKernel );
			CUDAPM_START_EVENT_CHANNEL( 1, cacheId, gpucache_bricks_gpuFetchBricks );
			CUDAPM_EVENT_NUMELEMS_CHANNEL( 1, cacheId, gpucache_bricks_gpuFetchBricks, numElems );

			// Buffer of nodes addresses with their requests
			thrust::device_vector< uint >* nodesAddressCompactList = _d_UpdateCompactList;
			// Sorted buffer of elements (according to timestamps : "un-used ones" first then "used ones" after)
			thrust::device_vector< uint >* elemsAddressCompactList = _d_elemAddressList;

#if CUDAPERFMON_CACHE_INFO==1
			{
				dim3 blockSize( 64, 1, 1 );
				uint numBlocks = iDivUp( numElems, blockSize.x );
				dim3 gridSize = dim3( std::min( numBlocks, 32768U ), iDivUp( numBlocks,32768U ), 1 );

				SyntheticInfo_Update_DataWrite< ElementRes, AddressType ><<< gridSize, blockSize, 0 >>>(
					_d_CacheStateBufferArray->getPointer(),
					numElems,
					thrust::raw_pointer_cast( &(*elemsAddressCompactList)[ 0 ] ),
					_elemsCacheSize );

				GV_CHECK_CUDA_ERROR( "SyntheticInfo_Update_DataWrite" );
			}

			_numPagesWrited = numElems;
#endif
			// Ask provider to produce data
			//
			// This method is called by the cache manager when you have to produce data for a given pool
			//
			// numElems: number of elements to produce
			// nodesAddressCompactList: list containing the addresses of the "numElems" nodes concerned
			// elemsAddressCompactList: list containing "numElems" addresses where store the result
			//_provider->template produceData< ElementRes >( numElems, nodesAddressCompactList, elemsAddressCompactList, gpuPool, _pageTable );
			pProducer->produceData( numElems, nodesAddressCompactList, elemsAddressCompactList, Id() );

			CUDAPM_STOP_EVENT_CHANNEL( 0, cacheId, gpucache_nodes_subdivKernel );
			CUDAPM_STOP_EVENT_CHANNEL( 1, cacheId, gpucache_bricks_gpuFetchBricks );
		}
	}

	return numElems;
}
#endif

#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_PRODUCER
/******************************************************************************
 * Create the "update" list of a given type.
 *
 * "Update list" is the list of elements and their associated requests of a given type (node subdivision or brick load/produce)
 *
 * @param inputList Buffer of node addresses and their associated requests. First two bits of their 32 bit addresses stores the type of request
 * @param inputNumElem Number of elements to process
 * @param pTestFlag type of request (node subdivision or brick load/produce)
 *
 * @return the number of requests of given type
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::createUpdateList( uint* inputList, uint inputNumElem, uint pTestFlag )
{
	// ---- [ 1 ] ---- 1st step
	//
	// Fill the buffer of masks of valid elements whose attribute is equal to "pTestFlag"
	// Result is placed in "_d_TempUpdateMaskList"

	CUDAPM_START_EVENT( gpucachemgr_createUpdateList_createMask );

	// Set kernel execution configuration
	dim3 blockSize( 64, 1, 1 );
	uint numBlocks = iDivUp( inputNumElem, blockSize.x );
	dim3 gridSize = dim3( std::min( numBlocks, 65535U ) , iDivUp( numBlocks, 65535U ), 1 );

	// Given a flag indicating a type of request (subdivision or load), and the updated global buffer of requests,
	// it fills a resulting mask buffer.
	// In fact, this mask indicates, for each element of the usage list, if it was used in the current rendering pass
	CacheManagerCreateUpdateMask<<< gridSize, blockSize, 0 >>>( inputNumElem, inputList, /*output*/(uint*)thrust::raw_pointer_cast( &(*_d_TempUpdateMaskList)[ 0 ] ), pTestFlag );
	GV_CHECK_CUDA_ERROR( "UpdateCreateSubdivMask" );

	CUDAPM_STOP_EVENT( gpucachemgr_createUpdateList_createMask );

	// ---- [ 2 ] ---- 2nd step
	//
	// Concatenate only previous valid elements from input data in "inputList" into the buffer of requests
	// Result is placed in "_d_UpdateCompactList"

	CUDAPM_START_EVENT( gpucachemgr_createUpdateList_elemsReduction );

	// Stream compaction
	cudppCompact( _scanplan,
		/*output*/thrust::raw_pointer_cast( &(*_d_UpdateCompactList)[ 0 ] ), /*nbValidElements*/_d_numElementsPtr,
		/*input*/inputList, /*isValid*/(uint*)thrust::raw_pointer_cast( &(*_d_TempUpdateMaskList)[ 0 ] ),
		/*nbElements*/inputNumElem );
	GV_CHECK_CUDA_ERROR( "cudppCompact" );

	// Get number of elements
	size_t numElems;
	GS_CUDA_SAFE_CALL( cudaMemcpy( &numElems, _d_numElementsPtr, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

	CUDAPM_STOP_EVENT( gpucachemgr_createUpdateList_elemsReduction );

	return static_cast< uint >( numElems );
}
#else // GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_GVCACHEMANAGER
/******************************************************************************
 * Create the "update" list of a given type.
 *
 * "Update list" is the list of elements and their associated requests of a given type (node subdivision or brick load/produce)
 *
 * @param inputList Buffer of node addresses and their associated requests. First two bits of their 32 bit addresses stores the type of request
 * @param inputNumElem Number of elements to process
 * @param pTestFlag type of request (node subdivision or brick load/produce)
 *
 * @return the number of requests of given type
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::createUpdateList( uint* inputList, uint inputNumElem, uint pTestFlag )
{
	const uint cacheId = Id::value;

	// ---- [ 1 ] ---- 1st step
	//
	// Fill the buffer of masks of valid elements whose attribute is equal to "pTestFlag"
	// Result is placed in "_d_TempUpdateMaskList"

	CUDAPM_START_EVENT( gpucachemgr_createUpdateList_createMask );

	// Set kernel execution configuration
	dim3 blockSize( 64, 1, 1 );
	uint numBlocks = iDivUp( inputNumElem, blockSize.x );
	dim3 gridSize = dim3( std::min( numBlocks, 65535U ) , iDivUp( numBlocks, 65535U ), 1 );

	// Given a flag indicating a type of request (subdivision or load), and the updated global buffer of requests,
	// it fills a resulting mask buffer.
	// In fact, this mask indicates, for each element of the usage list, if it was used in the current rendering pass
	CacheManagerCreateUpdateMask<<< gridSize, blockSize, 0 >>>( inputNumElem, inputList, /*output*/(uint*)thrust::raw_pointer_cast( &(*_d_TempUpdateMaskList)[ 0 ] ), pTestFlag );
	GV_CHECK_CUDA_ERROR( "UpdateCreateSubdivMask" );

	CUDAPM_STOP_EVENT( gpucachemgr_createUpdateList_createMask );

	// ---- [ 2 ] ---- 2nd step
	//
	// Concatenate only previous valid elements from input data in "inputList" into the buffer of requests
	// Result is placed in "_d_UpdateCompactList"

	CUDAPM_START_EVENT( gpucachemgr_createUpdateList_elemsReduction );

	// Stream compaction
	cudppCompact( _scanplan,
	#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_WITH_COMBINED_COPY
		/*output*/thrust::raw_pointer_cast( &(*_d_UpdateCompactList)[ 0 ] ), /*nbValidElements*/_d_numElementsPtr,
	#else
		/*output*/thrust::raw_pointer_cast( &(*_d_UpdateCompactList)[ 0 ] ), /*nbValidElements*/_d_nbValidRequests + cacheId + 1,
	#endif
		/*input*/inputList, /*isValid*/(uint*)thrust::raw_pointer_cast( &(*_d_TempUpdateMaskList)[ 0 ] ),
		/*nbElements*/inputNumElem );
	GV_CHECK_CUDA_ERROR( "cudppCompact" );

	// Get number of elements
	size_t numElems = 0;
	//#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_WITH_COMBINED_COPY
	//GS_CUDA_SAFE_CALL( cudaMemcpy( &numElems, _d_numElementsPtr, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	//#else
	//GS_CUDA_SAFE_CALL( cudaMemcpy( &numElems, _d_nbValidRequests + cacheId + 1, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	//#endif

	CUDAPM_STOP_EVENT( gpucachemgr_createUpdateList_elemsReduction );

	return static_cast< uint >( numElems );
}
/******************************************************************************
 * Create the "update" list of a given type.
 *
 * "Update list" is the list of elements and their associated requests of a given type (node subdivision or brick load/produce)
 *
 * @param inputList Buffer of node addresses and their associated requests. First two bits of their 32 bit addresses stores the type of request
 * @param inputNumElem Number of elements to process
 * @param pTestFlag type of request (node subdivision or brick load/produce)
 *
 * @return the number of requests of given type
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
uint GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::createUpdateListCopyAsync( uint* inputList, uint inputNumElem, uint pTestFlag )
{
	const uint cacheId = Id::value;

	// ---- [ 2 ] ---- 2nd step
	//
	// Concatenate only previous valid elements from input data in "inputList" into the buffer of requests
	// Result is placed in "_d_UpdateCompactList"

//	CUDAPM_START_EVENT( gpucachemgr_createUpdateList_elemsReduction );

	// Get number of elements
	size_t numElems;
	#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_WITH_COMBINED_COPY
	GS_CUDA_SAFE_CALL( cudaMemcpy( &numElems, _d_numElementsPtr, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	#else
	GS_CUDA_SAFE_CALL( cudaMemcpy( &numElems, _d_nbValidRequests + cacheId + 1, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	#endif

//	CUDAPM_STOP_EVENT( gpucachemgr_createUpdateList_elemsReduction );

	return static_cast< uint >( numElems );
}
#endif //GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_PRODUCER

/******************************************************************************
 * Invalidate elements
 *
 * Timestamps are reset to 1 and node addresses to 0 (but not the 2 first flags)
 *
 * @param numElems number of elements to invalidate (this is done for the numElems first elements, I.e. the unused ones)
 * @param numValidPageTableSlots ...
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
void GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::invalidateElements( uint numElems, int numValidPageTableSlots )
{
	// ---- [ 1 ] ---- 1st step

	// Invalidation procedure
	{
		// Set kernel execution configuration
		dim3 blockSize( 64, 1, 1 );
		uint numBlocks = iDivUp( numElems, blockSize.x );
		dim3 gridSize = dim3( std::min( numBlocks, 65535U ), iDivUp( numBlocks, 65535U ), 1 );

		// Reset the time stamp info of given elements to 1
		uint* d_sortedElemAddressList = thrust::raw_pointer_cast( &(*_d_elemAddressList)[ 0 ] );
		CacheManagerFlagInvalidations< ElementRes, AddressType >
			<<< gridSize, blockSize, 0 >>>( _d_cacheManagerKernel/*output : time stamps*/, numElems, /*input*/d_sortedElemAddressList );

		GV_CHECK_CUDA_ERROR( "CacheManagerFlagInvalidations" );
	}

	// ---- [ 2 ] ---- 2nd step

	{
		uint3 pageTableRes = _d_pageTableArray->getResolution();
		uint numPageTableElements = pageTableRes.x * pageTableRes.y * pageTableRes.z;
		if ( numValidPageTableSlots >= 0 )
		{
			numPageTableElements = min( numPageTableElements, numValidPageTableSlots );
		}

		// Set kernel execution configuration
		dim3 blockSize( 64, 1, 1 );
		uint numBlocks = iDivUp( numPageTableElements, blockSize.x );
		dim3 gridSize = dim3( std::min( numBlocks, 65535U ), iDivUp( numBlocks, 65535U ), 1 );

		// Reset all node addresses in the cache to NULL (i.e 0).
		// Only the first 30 bits of address are set to 0, not the 2 first flags.
		CacheManagerInvalidatePointers< ElementRes, AddressType >
			<<< gridSize, blockSize, 0 >>>( /*in*/_d_cacheManagerKernel, /*in*/numPageTableElements, /*modified*/_d_pageTableArray->getDeviceArray() );

		GV_CHECK_CUDA_ERROR( "VolTreeInvalidateNodePointers" );
	}
}

/******************************************************************************
 * Get the flag telling whether or not cache has exceeded its capacity
 *
 * @return flag telling whether or not cache has exceeded its capacity
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
inline bool GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::hasExceededCapacity() const
{
	return _exceededCapacity;
}

/******************************************************************************
 * This method is called to serialize an object
 *
 * @param pStream the stream where to write
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
inline void GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::write( std::ostream& pStream ) const
{
	//_pageTable = new PageTableType();

	// - timestamp buffer
	GvCore::Array3D< uint >* timeStamps = new GvCore::Array3D< uint >( _d_TimeStampArray->getResolution(), GvCore::Array3D< uint >::StandardHeapMemory );
	memcpyArray( timeStamps, _d_TimeStampArray );
	pStream.write( reinterpret_cast< const char* >( timeStamps->getPointer() ), sizeof( uint ) * timeStamps->getNumElements() );
	delete timeStamps;

	// List of elements in the cache
	thrust::host_vector< uint >* elemAddresses = new thrust::host_vector< uint >( _d_elemAddressList->size() );
	thrust::copy( _d_elemAddressList->begin(), _d_elemAddressList->end(), elemAddresses->begin() );
	pStream.write( reinterpret_cast< const char* >( elemAddresses->data() ), sizeof( uint ) * elemAddresses->size() );
	delete elemAddresses;

	// Buffer of requests (masks of associated request in the current frame)
	thrust::host_vector< uint >* requestBuffer = new thrust::host_vector< uint >( _d_UpdateCompactList->size() );
	thrust::copy( _d_UpdateCompactList->begin(), _d_UpdateCompactList->end(), requestBuffer->begin() );
	pStream.write( reinterpret_cast< const char* >( requestBuffer->data() ), sizeof( uint ) * requestBuffer->size() );
	delete requestBuffer;
}

/******************************************************************************
 * This method is called to deserialize an object
 *
 * @param pStream the stream from which to read
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType >
inline void GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
::read( std::istream& pStream )
{
}

} // namespace GvCache
