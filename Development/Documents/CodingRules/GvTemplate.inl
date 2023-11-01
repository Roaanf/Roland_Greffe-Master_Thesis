/*
 * Copyright (C) 2011 Fabrice Neyret <Fabrice.Neyret@imag.fr>
 * Copyright (C) 2011 Cyril Crassin <Cyril.Crassin@icare3d.org>
 * Copyright (C) 2011 Morgan Armand <morgan.armand@gmail.com>
 *
 * This file is part of Gigavoxels.
 *
 * Gigavoxels is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gigavoxels is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gigavoxels.  If not, see <http://www.gnu.org/licenses/>.
 */

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCache
{

//Protect null reference + root nodes in the octree -> but still a bug when too much loading
//Todo: check this bug !
static const uint cNumLockedElements = 1 + 1;

/******************************************************************************
 * Constructor
 *
 * @param cachesize
 * @param pageTableArray
 * @param graphicsInteroperability
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >
::GvCacheManager(uint3 cachesize, PageTableArrayType *pageTableArray, uint graphicsInteroperability)
:	cacheSize( cachesize )
{
	// Compute elements cache size
	elemsCacheSize = cacheSize / ElementRes::get();

	d_pageTableArray = pageTableArray;
	pageTable = new PageTableType();

	// Initialize the timestamp buffer
	d_TimeStampArray = new GvCore::Array3DGPULinear< uint >( elemsCacheSize, graphicsInteroperability );
	d_TimeStampArray->fill( 0 );

	this->numElements = elemsCacheSize.x * elemsCacheSize.y * elemsCacheSize.z - cNumLockedElements;

	numElemsNotUsed=numElements;

	d_elemAddressList		= new thrust::device_vector<uint>(this->numElements);
	d_elemAddressListTmp	= new thrust::device_vector<uint>(this->numElements);

	d_TempMaskList			= GvCacheManagerResources::getTempUsageMask1(this->numElements);
	d_TempMaskList2			= GvCacheManagerResources::getTempUsageMask2(this->numElements);

	thrust::fill(d_TempMaskList->begin(), d_TempMaskList->end(), 0);
	thrust::fill(d_TempMaskList2->begin(), d_TempMaskList2->end(), 0);

	uint3 pageTableRes = d_pageTableArray->getResolution();
	uint pageTableResLinear=pageTableRes.x*pageTableRes.y*pageTableRes.z;

	//d_TempUpdateMaskList = GvCacheManagerResources::getTempUsageMask1(pageTableRes.x*pageTableRes.y*pageTableRes.z); 
	d_TempUpdateMaskList	= new thrust::device_vector<uint>(pageTableResLinear);
	d_UpdateCompactList		= new thrust::device_vector<uint>(pageTableResLinear);

#if CUDAPERFMON_CACHE_INFO==1
	d_CacheStateBufferArray = new Array3DGPULinear<uchar4>(make_uint3(this->numElements, 1, 1)); 
#endif

	//Init
	thrust::host_vector<uint> tmpelemaddress;

	uint3 pos;
	for(pos.z=0; pos.z<elemsCacheSize.z; pos.z++)
	for(pos.y=0; pos.y<elemsCacheSize.y; pos.y++)
	for(pos.x=0; pos.x<elemsCacheSize.x; pos.x++)
	{
		tmpelemaddress.push_back(AddressType::packAddress(pos));
	}

	//Dont use element zero !
	thrust::copy( tmpelemaddress.begin() + cNumLockedElements, tmpelemaddress.end(), d_elemAddressList->begin() );

#if USE_CUDPP_LIBRARY
	uint cudppNumElem=std::max(pageTableRes.x*pageTableRes.y*pageTableRes.z, this->numElements);
	scanplan=GvCacheManagerResources::getScanplan(cudppNumElem);

	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_numElementsPtr, sizeof(size_t)));
#endif

	// The associated GPU side object receive a reference on the timestamp buffer
	d_cacheManagerKernel._timeStampArray = d_TimeStampArray->getDeviceArray();

#if GPUCACHE_BENCH_CPULRU==1
	cpuTimeStampArray=new Array3D<uint>(  d_TimeStampArray->getResolution()) ;
	cpuTimeStampsElemAddressList=new thrust::host_vector<uint> (this->numElements);
	cpuTimeStampsElemAddressList2=new thrust::host_vector<uint> (this->numElements);

	thrust::copy(tmpelemaddress.begin()+numLockedElements, tmpelemaddress.end(), cpuTimeStampsElemAddressList->begin());
	thrust::copy(tmpelemaddress.begin()+numLockedElements, tmpelemaddress.end(), cpuTimeStampsElemAddressList2->begin());
#endif

	std::cout << "######## CacheManager " << GPUProviderType::ProviderId::value << " ########" << std::endl;
	std::cout << "cacheSize: " << cacheSize << std::endl;
	std::cout << "elemsCacheSize: " << elemsCacheSize << std::endl;
	std::cout << "NumElements: " << this->numElements << std::endl;
	std::cout << "pageTableResLinear: " << pageTableResLinear << std::endl;
	std::cout << "################################" << std::endl;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >::~GvCacheManager()
{
#if USE_CUDPP_LIBRARY
	CUDA_SAFE_CALL( cudaFree(d_numElementsPtr) );
	CUDPPResult result = cudppDestroyPlan (scanplan);
	if (CUDPP_SUCCESS != result) {
		printf("Error destroying CUDPPPlan\n");
		exit(-1);
	}
#endif
}

/******************************************************************************
 * Get the number of elements managed by the cache.
 *
 * @return
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
uint GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >
::getNumElements() const
{
	return this->numElements;
}

/******************************************************************************
 * Get the timestamp list of the cache.
 * There is as many timestamps as elements in the cache.
 *
 * @return
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
GvCore::Array3DGPULinear< uint > *GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >
::getTimeStampList() const
{
	return d_TimeStampArray;
}

/******************************************************************************
 * ...
 * ...
 *
 * @return
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
thrust::device_vector<uint> *GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >
::getElementList() const
{
	return d_elemAddressList;
}

//template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
//void GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >::updateSymbols(){
//
//	CUDAPM_START_EVENT(gpucachemgr_updateSymbols);
//
//	/*CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_VTC_TimeStampArray,
//		(&d_TimeStampArray->getDeviceArray()),
//		sizeof(d_TimeStampArray->getDeviceArray()), 0, cudaMemcpyHostToDevice));*/
//	
//	CUDAPM_STOP_EVENT(gpucachemgr_updateSymbols);
//}

/******************************************************************************
 * Update the list of available elements according to their timestamps.
 * Unused and recycled elements will be placed first.
 *
 * @param manageUpdatesOnly
 *
 * @return the number of available elements
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
uint GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >
::updateTimeStamps(bool manageUpdatesOnly)
{
	if(true || this->lastNumLoads>0){
 
		//uint numElemsNotUsed=0;
		numElemsNotUsed = 0;

		uint cacheNumElems=getNumElements();

		//std:cout<<"manageUpdatesOnly "<<(int)manageUpdatesOnly<<"\n";

		uint activeNumElems=cacheNumElems;
		//TODO: re-enable manageUpdatesOnly !
		/*if(manageUpdatesOnly)
			activeNumElems=this->lastNumLoads;*/
		uint inactiveNumElems=cacheNumElems-activeNumElems;

		uint numElemToSort=activeNumElems;


		if( numElemToSort>0 ){

			uint sortingStartPos=0;

#if GPUCACHE_BENCH_CPULRU==0

			CUDAPM_START_EVENT(gpucache_updateTimeStamps_createMask);

			// Create masks in a single pass
			dim3 blockSize(64, 1, 1);
			uint numBlocks=iDivUp(numElemToSort, blockSize.x);
			dim3 gridSize=dim3( std::min( numBlocks, 65535U) , iDivUp(numBlocks,65535U), 1);

			// Generate an error with CUDA 3.2
			CacheManagerFlagTimeStampsSP<ElementRes, AddressType>
				<<<gridSize, blockSize, 0>>>(d_cacheManagerKernel, numElemToSort,
					thrust::raw_pointer_cast(&(*d_elemAddressList)[sortingStartPos]),
					thrust::raw_pointer_cast(&(*d_TempMaskList)[0]),
					thrust::raw_pointer_cast(&(*d_TempMaskList2)[0]));

			CUT_CHECK_ERROR("CacheManagerFlagTimeStampsSP");
			CUDAPM_STOP_EVENT(gpucache_updateTimeStamps_createMask);

			thrust::device_vector<uint>::const_iterator elemAddressListFirst = d_elemAddressList->begin();
			thrust::device_vector<uint>::const_iterator elemAddressListLast = d_elemAddressList->begin() + numElemToSort;
			thrust::device_vector<uint>::iterator elemAddressListTmpFirst = d_elemAddressListTmp->begin();

			// Stream compaction to collect non-used elements at the beginning
			CUDAPM_START_EVENT(gpucache_updateTimeStamps_threadReduc1);
# if USE_CUDPP_LIBRARY
			cudppCompact (scanplan,
				thrust::raw_pointer_cast(&(*d_elemAddressListTmp)[inactiveNumElems]), d_numElementsPtr,
				thrust::raw_pointer_cast(&(*d_elemAddressList)[sortingStartPos]),
				thrust::raw_pointer_cast(&(*d_TempMaskList)[0]),
				numElemToSort);

			size_t numElemsNotUsedST;
			// Get number of elements
			CUDA_SAFE_CALL( cudaMemcpy( &numElemsNotUsedST, d_numElementsPtr, sizeof(size_t), cudaMemcpyDeviceToHost) );
			numElemsNotUsed=(uint)numElemsNotUsedST + inactiveNumElems;
# else // USE_CUDPP_LIBRARY
			size_t numElemsNotUsedST = thrust::copy_if(
				elemAddressListFirst,
				elemAddressListLast,
				d_TempMaskList->begin(),
				elemAddressListTmpFirst + inactiveNumElems,
				GvCore::not_equal_to_zero<uint>()) - (elemAddressListTmpFirst + inactiveNumElems);

			numElemsNotUsed=(uint)numElemsNotUsedST + inactiveNumElems;
# endif // USE_CUDPP_LIBRARY
			CUDAPM_STOP_EVENT(gpucache_updateTimeStamps_threadReduc1);

			// Stream compaction to collect used elements at the end
			CUDAPM_START_EVENT(gpucache_updateTimeStamps_threadReduc2);
# if USE_CUDPP_LIBRARY
			cudppCompact (scanplan,
				thrust::raw_pointer_cast(&(*d_elemAddressListTmp)[numElemsNotUsed]), d_numElementsPtr,
				thrust::raw_pointer_cast(&(*d_elemAddressList)[sortingStartPos]),
				thrust::raw_pointer_cast(&(*d_TempMaskList2)[0]),
				numElemToSort);
# else // USE_CUDPP_LIBRARY
			thrust::copy_if(
				elemAddressListFirst,
				elemAddressListLast,
				d_TempMaskList2->begin(),
				elemAddressListTmpFirst + numElemsNotUsed,
				GvCore::not_equal_to_zero<uint>());
# endif // USE_CUDPP_LIBRARY
			CUDAPM_STOP_EVENT(gpucache_updateTimeStamps_threadReduc2);
#else

			memcpyArray(cpuTimeStampArray, d_TimeStampArray);
			
			uint curDstPos=0;

			CUDAPM_START_EVENT(gpucachemgr_updateTimeStampsCPU);
			//Copy unused
			for(uint i=0; i<cacheNumElems; ++i){
				uint elemAddressEnc=(*cpuTimeStampsElemAddressList)[i];
				uint3 elemAddress;
				elemAddress=AddressType::unpackAddress(elemAddressEnc);

				if(cpuTimeStampArray->get(elemAddress)!=GvCacheManager_currentTime){
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

				if(cpuTimeStampArray->get(elemAddress)==GvCacheManager_currentTime){
					(*cpuTimeStampsElemAddressList2)[curDstPos]=elemAddressEnc;
					curDstPos++;
				}
			}
			CUDAPM_STOP_EVENT(gpucachemgr_updateTimeStampsCPU);

			thrust::copy(cpuTimeStampsElemAddressList2->begin(), cpuTimeStampsElemAddressList2->end(), d_elemAddressListTmp->begin());
#endif
		}else{ //if( numElemToSort>0 )
			numElemsNotUsed=cacheNumElems;
		}

		////swap buffers////
		thrust::device_vector<uint> *tmpl = d_elemAddressList;
		d_elemAddressList = d_elemAddressListTmp;
		d_elemAddressListTmp = tmpl;

#if CUDAPERFMON_CACHE_INFO==1
		{
			uint *usedPageList=thrust::raw_pointer_cast( &(*d_elemAddressList)[numElemsNotUsed] );

			uint numPageUsed=getNumElements()-numElemsNotUsed;

			if(numPageUsed>0){
				dim3 blockSize(128, 1, 1);
				uint numBlocks=iDivUp(numPageUsed, blockSize.x);
				dim3 gridSize=dim3( std::min( numBlocks, 32768U) , iDivUp(numBlocks,32768U), 1);

				SyntheticInfo_Update_PageUsed< ElementRes, AddressType >
					<<<gridSize, blockSize, 0>>>(
					d_CacheStateBufferArray->getPointer(), numPageUsed, usedPageList, elemsCacheSize);

				CUT_CHECK_ERROR("SyntheticInfo_Update_PageUsed");

				// update counter
				numPagesUsed=numPageUsed;
			}
		}
#endif
		this->lastNumLoads=0;
	}

	return numElemsNotUsed;
}

/******************************************************************************
 * 
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
void GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >::clearCache(){

	// FIXME: do we really need to do it with a cuda kernel? Anyway, need to take account
	// for the locked elements.
	thrust::host_vector<uint> tmpelemaddress;

	uint3 pos;
	for(pos.z=0; pos.z<elemsCacheSize.z; pos.z++)
	for(pos.y=0; pos.y<elemsCacheSize.y; pos.y++)
	for(pos.x=0; pos.x<elemsCacheSize.x; pos.x++)
	{
		tmpelemaddress.push_back(AddressType::packAddress(pos));
	}

	//Dont use element zero !
	thrust::copy( tmpelemaddress.begin() + cNumLockedElements, tmpelemaddress.end(), d_elemAddressList->begin() );

	//Init
	//CUDAPM_START_EVENT(gpucachemgr_clear_cpyAddr)

	////Uses kernel filling
	//uint *timeStampsElemAddressListPtr=thrust::raw_pointer_cast( &(*d_elemAddressList)[0] );

	//const dim3 blockSize(128, 1, 1);
	//uint numBlocks=iDivUp(this->numElements, blockSize.x);
	//dim3 gridSize=dim3( std::min( numBlocks, 65535U) , iDivUp(numBlocks,65535U), 1);

	//InitElemAddressList<AddressType>
	//	<<<gridSize, blockSize, 0>>>( timeStampsElemAddressListPtr, this->numElements, elemsCacheSize );

	//CUT_CHECK_ERROR("InitElemAddressList");

	//CUDAPM_STOP_EVENT(gpucachemgr_clear_cpyAddr)

	CUDAPM_START_EVENT(gpucachemgr_clear_fillML)
	thrust::fill(d_TempMaskList->begin(), d_TempMaskList->end(), 0);
	thrust::fill(d_TempMaskList2->begin(), d_TempMaskList2->end(), 0);
	CUDAPM_STOP_EVENT(gpucachemgr_clear_fillML)

	CUDAPM_START_EVENT(gpucachemgr_clear_fillTimeStamp)
	d_TimeStampArray->fill(0);
	CUDAPM_STOP_EVENT(gpucachemgr_clear_fillTimeStamp)
}

/******************************************************************************
 * 
 *
 * @param numElems
 * @param numValidPageTableSlots
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
void GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >
::invalidateElements(uint numElems, int numValidPageTableSlots)
{
	////invalidation procedure////
	{

		dim3 blockSize(64, 1, 1);
		uint numBlocks=iDivUp(numElems, blockSize.x);
		dim3 gridSize=dim3( std::min( numBlocks, 65535U) , iDivUp(numBlocks,65535U), 1);

		uint *d_sortedElemAddressList=thrust::raw_pointer_cast(&(*d_elemAddressList)[0]);
		CacheManagerFlagInvalidations<ElementRes, AddressType>
			<<<gridSize, blockSize, 0>>>(d_cacheManagerKernel, numElems, d_sortedElemAddressList);

		CUT_CHECK_ERROR("CacheManagerFlagInvalidations");
	}
	
	{

		uint3 pageTableRes=d_pageTableArray->getResolution();
		uint numPageTableElements=pageTableRes.x*pageTableRes.y*pageTableRes.z;
		if(numValidPageTableSlots>=0)
			numPageTableElements=min(numPageTableElements, numValidPageTableSlots);

		dim3 blockSize(64, 1, 1);
		uint numBlocks=iDivUp(numPageTableElements, blockSize.x);
		dim3 gridSize=dim3( std::min( numBlocks, 65535U) , iDivUp(numBlocks,65535U), 1);
		CacheManagerInvalidatePointers< ElementRes, AddressType >
			<<<gridSize, blockSize, 0>>>(d_cacheManagerKernel, numPageTableElements, d_pageTableArray->getDeviceArray());

		CUT_CHECK_ERROR("VolTreeInvalidateNodePointers");
	}
	
}

/******************************************************************************
 * 
 *
 * @param inputList
 * @param inputNumElem
 * @param testFlag
 *
 * @return 
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
uint GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >
::createUpdateList(uint *inputList, uint inputNumElem, uint testFlag)
{
	CUDAPM_START_EVENT(gpucachemgr_createUpdateList_createMask);

	dim3 blockSize(64, 1, 1);
	uint numBlocks=iDivUp(inputNumElem, blockSize.x);
	dim3 gridSize=dim3( std::min( numBlocks, 65535U) , iDivUp(numBlocks,65535U), 1);
	CacheManagerCreateUpdateMask<<<gridSize, blockSize, 0>>>(inputNumElem, inputList, (uint*)thrust::raw_pointer_cast(&(*d_TempUpdateMaskList)[0]), testFlag );
	CUT_CHECK_ERROR("UpdateCreateSubdivMask");

	CUDAPM_STOP_EVENT(gpucachemgr_createUpdateList_createMask);

	CUDAPM_START_EVENT(gpucachemgr_createUpdateList_elemsReduction);
#if USE_CUDPP_LIBRARY
	cudppCompact (scanplan,
		thrust::raw_pointer_cast(&(*d_UpdateCompactList)[0]), d_numElementsPtr,
		inputList, (uint*)thrust::raw_pointer_cast(&(*d_TempUpdateMaskList)[0]),
		inputNumElem );
	CUT_CHECK_ERROR("cudppCompact");

	uint numElems;
	//get number of elements
	CUDA_SAFE_CALL( cudaMemcpy( &numElems, d_numElementsPtr, sizeof(uint), cudaMemcpyDeviceToHost) );
#else // USE_CUDPP_LIBRARY
	thrust::device_ptr<uint> firstPtr = thrust::device_ptr<uint>(inputList);
	thrust::device_ptr<uint> lastPtr = thrust::device_ptr<uint>(inputList + inputNumElem);

	uint numElems = thrust::copy_if(firstPtr, lastPtr, d_TempUpdateMaskList->begin(),
		d_UpdateCompactList->begin(), GvCore::not_equal_to_zero<uint>()) - d_UpdateCompactList->begin();

	CUT_CHECK_ERROR("cudppCompact");
#endif // USE_CUDPP_LIBRARY
	CUDAPM_STOP_EVENT(gpucachemgr_createUpdateList_elemsReduction);

	return numElems;
}

/******************************************************************************
 * 
 *
 * @return 
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
GvCacheManagerKernel< ElementRes, AddressType > GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >::getKernelObject() {
	return	d_cacheManagerKernel;
}

/******************************************************************************
 * Set the producer
 *
 * @param provider Reference on a producer
 ******************************************************************************/
template< typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType >
void GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >
::setProvider( GPUProviderType* pProvider )
{
	mProvider = pProvider;
}

/******************************************************************************
 * 
 *
 * @param updateList
 * @param numUpdateElems
 * @param updateMask
 * @param maxNumElems
 * @param numValidNodes
 * @param gpuPool
 *
 * @return 
 ******************************************************************************/
template <typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType>
template <typename GPUPoolType>
uint GvCacheManager< ElementRes, AddressType, PageTableArrayType, PageTableType, GPUProviderType >
::genericWrite(uint *updateList, uint numUpdateElems, uint updateMask,
			   uint maxNumElems, uint numValidNodes, const GPUPoolType &gpuPool)
{
	uint numElems = 0;
	uint providerId = GPUProviderType::ProviderId::value;

	if (numUpdateElems > 0)
	{
		CUDAPM_START_EVENT_CHANNEL(0, providerId, gpucache_nodes_manageUpdates);
		numElems = createUpdateList(updateList, numUpdateElems, updateMask);
		CUDAPM_STOP_EVENT_CHANNEL(0, providerId, gpucache_nodes_manageUpdates);

		numElems = std::min(numElems, getNumElements());	// Prevent loading more than the cache size

		//-----------------------------------
		// QUESTION : à quoi sert le test ?
		// - ça arrive sur la "simple sphere" quand on augmente trop le depth
		//-----------------------------------
		if (numElems > numElemsNotUsed)
		{
			std::cout << "CacheManager<" << providerId << ">: Warning: "
				<< numElemsNotUsed << " slots available!" << std::endl;
		}

		///numElems = std::min(numElems, numElemsNotUsed);	// Prevent replacing elements in use
		///numElems = std::min(numElems, maxNumElems);		// Smooth loading

		if (numElems > 0)
		{
			//std::cout << "CacheManager<" << providerId << ">: " << numElems << " requests" << std::endl;

			// Invalidation phase
			totalNumLoads += numElems;
			lastNumLoads = numElems;

			CUDAPM_START_EVENT_CHANNEL(1, providerId, gpucache_bricks_bricksInvalidation);
			invalidateElements(numElems, numValidNodes);
			CUDAPM_STOP_EVENT_CHANNEL(1, providerId, gpucache_bricks_bricksInvalidation);

			CUDAPM_START_EVENT_CHANNEL(0, providerId, gpucache_nodes_subdivKernel);
			CUDAPM_START_EVENT_CHANNEL(1, providerId, gpucache_bricks_gpuFetchBricks);
			CUDAPM_EVENT_NUMELEMS_CHANNEL(1, providerId, gpucache_bricks_gpuFetchBricks, numElems);

			// Write new elements into the cache
			thrust::device_vector<uint> *nodesAddressCompactList = d_UpdateCompactList;
			thrust::device_vector<uint> *elemsAddressCompactList = d_elemAddressList;

#if CUDAPERFMON_CACHE_INFO==1
			{
				dim3 blockSize(64, 1, 1);
				uint numBlocks=iDivUp(numElems, blockSize.x);
				dim3 gridSize=dim3( std::min( numBlocks, 32768U) , iDivUp(numBlocks,32768U), 1);

				SyntheticInfo_Update_DataWrite< ElementRes, AddressType ><<<gridSize, blockSize, 0>>>(
					d_CacheStateBufferArray->getPointer(), numElems,
					thrust::raw_pointer_cast(&(*elemsAddressCompactList)[0]),
					elemsCacheSize);

				CUT_CHECK_ERROR("SyntheticInfo_Update_DataWrite");
			}

			numPagesWrited = numElems;
#endif
			mProvider->template produceData<ElementRes>(numElems, nodesAddressCompactList,
				elemsAddressCompactList, gpuPool, pageTable);

			CUDAPM_STOP_EVENT_CHANNEL(0, providerId, gpucache_nodes_subdivKernel);
			CUDAPM_STOP_EVENT_CHANNEL(1, providerId, gpucache_bricks_gpuFetchBricks);
		}
	}

	return numElems;
}

} // namespace GvCache
