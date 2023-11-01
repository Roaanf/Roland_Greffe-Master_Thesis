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

#ifndef _BVHTREECACHEMANAGER_H_
#define _BVHTREECACHEMANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// CUDA
#include <vector_types.h>

// Cuda SDK
#include <helper_math.h>

#include "RendererBVHTrianglesCommon.h"

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// cudpp
#include <cudpp.h>

//#include "BvhTreeCache.hcu"

// GigaVoxels
#include <GvPerfMon/GsPerformanceMonitor.h>

#include "BvhTreeCacheManager.hcu"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Protect null reference
static const uint cNumLockedElements = 1;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class GPUCacheManager
 *
 * @brief The GPUCacheManager class provides ...
 *
 * @param ElementRes ...
 * @param ProviderType ...
 */
template< unsigned int TId, typename ElementRes >
class GPUCacheManager
{
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * The cache identifier
	 */
	typedef Loki::Int2Type< TId > Id;

	/******************************* ATTRIBUTES *******************************/

#if USE_SYNTHETIC_INFO
	/**
	 * ...
	 */
	GvCore::GsLinearMemory< uchar4 >* d_SyntheticCacheStateBufferArray;
#endif

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GPUCacheManager( uint3 cachesize, uint3 elemsize );
	
	/**
	 * Destructor
	 */
	virtual ~GPUCacheManager();

	/**
	 * ...
	 *
	 * @return ...
	 */
	uint getNumElements()
	{
		return _elemsCacheSize.x * _elemsCacheSize.y * _elemsCacheSize.z;
	}

	/**
	 * ...
	 *
	 * @return ...
	 */
	GvCore::GsLinearMemory< uint >* getdTimeStampArray()
	{
		return d_TimeStampArray;
	}

	/**
	 * ...
	 *
	 * @return ...
	 */
	GvCore::GsLinearMemory< uint >* getTimeStampsElemAddressList()
	{
		return d_elemAddressList;
	}

	/**
	 * Update symbols
	 */
	void updateSymbols();

	/**
	  * Handle requests
	 *
	 * @param updateList input list of node addresses that have been updated with "subdivision" or "load/produce" requests
	 * @param numUpdateElems maximum number of elements to process
	 * @param updateMask a unique given type of requests to take into account
	 * @param maxNumElems ...
	 * @param numValidNodes ...
	 * @param gpuPool associated pool (nodes or bricks)
	 *
	 * @return ... 
	 */
	template< typename GPUPoolType, typename TProducerType >
	uint genericWrite( uint* updateList, uint numUpdateElems, uint updateMask,
		uint maxNumElems, uint numValidNodes, const GPUPoolType& gpuPool, TProducerType* pProducer );

	/**
	 * ...
	 */
	// Return number of elements not used
	uint updateTimeStamps();

	/**
	 * Clear cache
	 */
	void clearCache();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Create the list of nodes that will be concerned by the data production management
	 *
	 * @param inputList input list of node addresses that have been updated with "subdivision" or "load/produce" requests
	 * @param inputNumElem maximum number of elements to process
	 * @param testFlag a unique given type of requests to take into account
	 *
	 * @return the number of requests that the manager will have to handle
	 */
	uint createUpdateList( uint* inputList, uint inputNumElem, uint testFlag );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	uint3 _cacheSize;

	/**
	 * ...
	 */
	uint3 _elemsCacheSize;

	/**
	 * Array of time stamps
	 *
	 * For each elements of the cache, it contains an associated time.
	 * During rendering, used elements are flagged with the current frame number.
	 */
	GvCore::GsLinearMemory< uint >* d_TimeStampArray;

	/**
	 * List of masks of "unused elements" at current time (.i.e. frame)
	 */
	thrust::device_vector< uint >* d_TempMaskList;

	/**
	 * List of masks of "used elements" at current time (.i.e. frame)
	 */
	thrust::device_vector< uint >* d_TempMaskList2; //for cudpp approach

	/**
	 * The final list of elements (nodes or bricks) where data production management
	 * will be able to store its newly produced elements.
	 */
	GvCore::GsLinearMemory< uint >* d_elemAddressList;
	
	/**
	 * Temporary list used to create the list of elements where to write new elements
	 * of the data production management.
	 * It is associated to [ d_elemAddressList ].
	 */
	GvCore::GsLinearMemory< uint >* d_elemAddressListTmp;

	/**
	 * Temporary list used to store the masks of nodes whose request corresponds to a given type.
	 * It is associated to [ d_UpdateCompactList ].
	 */
	thrust::device_vector< uint >* d_TempUpdateMaskList;

	/**
	 * The final list of nodes whose request corresponds to a given type
	 *
	 * These will be the nodes concerned by the data production management
	 */
	GvCore::GsLinearMemory< uint >* d_UpdateCompactList;

	/**
	 * CUDPP
	 */
	size_t* d_numElementsPtr;
	CUDPPHandle scanplan;
	
	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GPUCacheManager( const GPUCacheManager& );

	/**
	 * Copy operator forbidden.
	 */
	GPUCacheManager& operator=( const GPUCacheManager& );

};

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< unsigned int TId, typename ElementRes >
GPUCacheManager< TId, ElementRes >
::~GPUCacheManager()
{
	//GS_CUDA_SAFE_CALL( cudaFree(d_numElementsPtr) );
	//CUDPPResult result = cudppDestroyPlan (scanplan);
	//if (CUDPP_SUCCESS != result) {
	//	printf("Error destroying CUDPPPlan\n");
	//	exit(-1);
	//}
}

/******************************************************************************
 * Update symbols
 ******************************************************************************/
template< unsigned int TId, typename ElementRes >
void GPUCacheManager< TId, ElementRes >
::updateSymbols()
{
	// Update time stamps buffer
	// Linux : taking address of temporary is error...
	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_TimeStampArray, (&d_TimeStampArray->getDeviceArray()), sizeof( d_TimeStampArray->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );
	GvCore::GsLinearMemoryKernel< uint > timeStampArrayDevice = d_TimeStampArray->getDeviceArray();
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_TimeStampArray, &timeStampArrayDevice, sizeof( d_TimeStampArray->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Generate the final list of elements (nodes or bricks) where data production management
 * will be able to store its newly produced elements.
 *
 * Resulting list will be placed in [ d_elemAddressList ]
 ******************************************************************************/
//Return number of elements not used
template< unsigned int TId, typename ElementRes >
uint GPUCacheManager< TId, ElementRes >
::updateTimeStamps()
{
	uint numElemsNotUsed = 0;

	if ( true/* || this->lastNumLoads>0*/ )
	{
 		//uint numElemsNotUsed = 0;
		//uint numElemsNotUsed = 0;

		uint cacheNumElems = getNumElements();

		//std:cout << "manageUpdatesOnly " << (int)manageUpdatesOnly << "\n";

		uint activeNumElems = cacheNumElems;
		//TODO: re-enable manageUpdatesOnly !
		/*if ( manageUpdatesOnly )
			activeNumElems = this->lastNumLoads;*/
		uint inactiveNumElems = cacheNumElems - activeNumElems;

		uint numElemToSort = activeNumElems;
		
		if ( numElemToSort > 0 )
		{
			uint sortingStartPos = 0;

#if GPUCACHE_BENCH_CPULRU==0

			CUDAPM_START_EVENT( cache_updateTimestamps_createMasks );

			// Create masks in a single pass
			dim3 blockSize( 64, 1, 1 );
			uint numBlocks = iDivUp( numElemToSort, blockSize.x );
			dim3 gridSize = dim3( std::min( numBlocks, 65535U ) , iDivUp( numBlocks, 65535U ), 1 );
			
			//---------------------------------------------------
			///// TEST Pascal
			//---------------------------------------------------
			numElemToSort -= 1;
			//---------------------------------------------------

			// Generate an error with CUDA 3.2
			CacheManagerFlagTimeStampsSP/*<ElementRes, AddressType>*/
				<<<gridSize, blockSize, 0>>>(/*d_cacheManagerKernel, */numElemToSort,
					d_elemAddressList->getPointer( sortingStartPos ),
					thrust::raw_pointer_cast(&(*d_TempMaskList)[0]),
					thrust::raw_pointer_cast(&(*d_TempMaskList2)[0]));

			GV_CHECK_CUDA_ERROR("CacheManagerFlagTimeStampsSP");
			CUDAPM_STOP_EVENT(cache_updateTimestamps_createMasks);

		/*	thrust::device_vector<uint>::const_iterator elemAddressListFirst = d_elemAddressList->begin();
			thrust::device_vector<uint>::const_iterator elemAddressListLast = d_elemAddressList->begin() + numElemToSort;
			thrust::device_vector<uint>::iterator elemAddressListTmpFirst = d_elemAddressListTmp->begin();*/

			// Stream compaction to collect non-used elements at the beginning
			CUDAPM_START_EVENT(cache_updateTimestamps_threadReduc1);

			cudppCompact( scanplan,
				d_elemAddressListTmp->getPointer( inactiveNumElems ), d_numElementsPtr,
				d_elemAddressList->getConstPointer( sortingStartPos ),
				thrust::raw_pointer_cast(&(*d_TempMaskList)[0]),
				numElemToSort );

			size_t numElemsNotUsedST;
			// Get number of elements
			GS_CUDA_SAFE_CALL( cudaMemcpy( &numElemsNotUsedST, d_numElementsPtr, sizeof(size_t), cudaMemcpyDeviceToHost) );
			numElemsNotUsed=(uint)numElemsNotUsedST + inactiveNumElems;

			CUDAPM_STOP_EVENT(cache_updateTimestamps_threadReduc1);

			// Stream compaction to collect used elements at the end
			CUDAPM_START_EVENT(cache_updateTimestamps_threadReduc2);

			cudppCompact (scanplan,
				d_elemAddressListTmp->getPointer( numElemsNotUsed ), d_numElementsPtr,
				d_elemAddressList->getConstPointer( sortingStartPos ),
				thrust::raw_pointer_cast(&(*d_TempMaskList2)[0]),
				numElemToSort );

			CUDAPM_STOP_EVENT(cache_updateTimestamps_threadReduc2);
#else

			memcpyArray(cpuTimeStampArray, d_TimeStampArray);
			
			uint curDstPos=0;

			CUDAPM_START_EVENT(gpucachemgr_updateTimeStampsCPU);
			//Copy unused
			for(uint i=0; i<cacheNumElems; ++i){
				uint elemAddressEnc=(*cpuTimeStampsElemAddressList)[i];
				uint3 elemAddress;
				elemAddress=AddressType::unpackAddress(elemAddressEnc);

				if(cpuTimeStampArray->get(elemAddress)!=GPUCacheManager_currentTime){
					(*cpuTimeStampsElemAddressList2)[curDstPos]=elemAddressEnc;
					curDstPos++;
				}
			}
			numElemsNotUsed=curDstPos;

			//copy used
			for(uint i=0; i<cacheNumElems; ++i){
				uint elemAddressEnc=(*cpuTimeStampsElemAddressList)[i];
				uint3 elemAddress;
				elemAddress=AddressType::unpackAddress(elemAddressEnc);

				if(cpuTimeStampArray->get(elemAddress)==GPUCacheManager_currentTime){
					(*cpuTimeStampsElemAddressList2)[curDstPos]=elemAddressEnc;
					curDstPos++;
				}
			}
			CUDAPM_STOP_EVENT(gpucachemgr_updateTimeStampsCPU);

			thrust::copy(cpuTimeStampsElemAddressList2->begin(), cpuTimeStampsElemAddressList2->end(), d_elemAddressListTmp->begin());
#endif
		}
		else
		{ //if( numElemToSort>0 )
			numElemsNotUsed = cacheNumElems;
		}

		////swap buffers////
		GvCore::GsLinearMemory< uint >* tmpl = d_elemAddressList;
		d_elemAddressList = d_elemAddressListTmp;
		d_elemAddressListTmp = tmpl;

#if CUDAPERFMON_CACHE_INFO==1
		{
			uint* usedPageList = d_elemAddressList->getPointer( numElemsNotUsed );

			uint numPageUsed = getNumElements() - numElemsNotUsed;

			if(numPageUsed>0){
				dim3 blockSize(128, 1, 1);
				uint numBlocks=iDivUp(numPageUsed, blockSize.x);
				dim3 gridSize=dim3( std::min( numBlocks, 32768U) , iDivUp(numBlocks,32768U), 1);

				SyntheticInfo_Update_PageUsed< ElementRes, AddressType >
					<<<gridSize, blockSize, 0>>>(
					d_CacheStateBufferArray->getPointer(), numPageUsed, usedPageList, elemsCacheSize);

				GV_CHECK_CUDA_ERROR("SyntheticInfo_Update_PageUsed");

				// update counter
				numPagesUsed=numPageUsed;
			}
		}
#endif
		//this->lastNumLoads=0;
	}

	return numElemsNotUsed;
}

/******************************************************************************
 * Clear cache
 ******************************************************************************/
template< unsigned int TId, typename ElementRes >
void GPUCacheManager< TId, ElementRes >
::clearCache()
{
	////Init
	//thrust::host_vector<uint> tmpelemaddress;

	//uint3 pos;
	//for(pos.x=0; pos.x<elemsCacheSize.x; pos.x++){

	//	tmpelemaddress.push_back(pos.x);
	//}

	//thrust::copy(tmpelemaddress.begin()+1, tmpelemaddress.end(), d_TimeStampsElemAddressList->begin());

	//thrust::fill(d_TempMaskList->begin(), d_TempMaskList->end(), (uint) 0);
	//thrust::fill(d_TempMaskList2->begin(), d_TempMaskList2->end(), (uint) 0);

	////thrust::fill(d_TimeStampsElemAddressList2->begin(), d_TimeStampsElemAddressList2->end(), (uint) 0);
	//d_TimeStampArray->fill(0);
}

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "BvhTreeCacheManager.inl"

#endif // !_BVHTREECACHEMANAGER_H_
