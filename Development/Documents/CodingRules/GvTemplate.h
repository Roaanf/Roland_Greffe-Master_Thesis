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

#ifndef GVTEMPLATE_H
#define GVTEMPLATE_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvNamespace
{

	/** 
	 * @class GvTemplate
	 *
	 * @brief The GvTemplate class provides...
	 *
	 * This class is used to ...
	 */
	template< typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename GPUProviderType >
	class GvCacheManager
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

	public:

		/****************************** INNER TYPES *******************************/

		/**
		 * Type definition for the GPU side associated object
		 */
		typedef GvCacheManagerKernel< ElementRes, AddressType > KernelType;

		/******************************* ATTRIBUTES *******************************/

		/**
		 * Page table
		 */
		PageTableType* pageTable;

		/**
		 *
		 */
		uint totalNumLoads;
		uint lastNumLoads;
		uint numElemsNotUsed;

		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 *
		 * @param cachesize
		 * @param pageTableArray
		 * @param graphicsInteroperability
		 */
		GvCacheManager( uint3 cachesize, PageTableArrayType* pageTableArray, uint graphicsInteroperability = 0 );

		/**
		 * Destructor
		 */
		~GvCacheManager();

		/**
		 * Get the number of elements managed by the cache.
		 *
		 * @return the number of elements managed by the cache
		 */
		uint getNumElements() const;

		/**
		 * Get the timestamp list of the cache.
		 * There is as many timestamps as elements in the cache.
		 */
		GvCore::Array3DGPULinear< uint >* getTimeStampList() const;

		/**
		 * Get the sorted list of cache elements, least recently used first.
		 * There is as many timestamps as elements in the cache.
		 */
		thrust::device_vector< uint >* getElementList() const;

		//void updateSymbols();

		//! Update the list of available elements according to their timestamps.
		//! Unused and recycled elements will be placed first.
		//! \return the number of available elements
		uint updateTimeStamps( bool manageUpdatesOnly );

		/**
		 *
		 */
		void clearCache();

		////////////
		KernelType getKernelObject();

		/**
		  * Set the producer
		 *
		 * @param provider Reference on a producer
		 */
		void setProvider( GPUProviderType* pProvider );

		/**
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
		 */
		template< typename GPUPoolType >
		uint genericWrite( uint* updateList, uint numUpdateElems, uint updateMask,
			uint maxNumElems, uint numValidNodes, const GPUPoolType& gpuPool );

	#if CUDAPERFMON_CACHE_INFO==1
		Array3DGPULinear<uchar4>	*d_CacheStateBufferArray;
		uint numPagesUsed;
		uint numPagesWrited;
	#endif

		/**************************************************************************
		 **************************** PROTECTED SECTION ***************************
		 **************************************************************************/

	protected:

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**************************************************************************
		 ***************************** PRIVATE SECTION ****************************
		 **************************************************************************/

	private:

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**
		 *
		 */
		uint createUpdateList( uint* inputList, uint inputNumElem, uint testFlag );
		void invalidateElements( uint numElems, int numValidPageTableSlots = -1 );

	private:

		/**
		 *
		 */
		uint3 cacheSize;
		uint3 elemsCacheSize;

		/**
		 *
		 */
		uint numElements;

		/**
		 * Timestamp buffer.
		 * It attaches a 32-bit integer timestamp to each element (node tile or brick) of the pool.
		 * Timestamp corresponds to the time of the current rendering pass.
		 */
		GvCore::Array3DGPULinear< uint >* d_TimeStampArray;

		//! This list contains all elements addresses, sorted correctly so the unused one
		//! are at the beginning.
		thrust::device_vector< uint >			*d_elemAddressList;
		thrust::device_vector< uint >			*d_elemAddressListTmp;

		/**
		 *
		 */
		thrust::device_vector< uint >			*d_UpdateCompactList;

		/**
		 *
		 */
		thrust::device_vector< uint >			*d_TempMaskList;
		thrust::device_vector< uint >			*d_TempMaskList2; //for cudpp approach

		/**
		 *
		 */
		thrust::device_vector< uint >			*d_TempUpdateMaskList;

		/**
		 *
		 */
		PageTableArrayType						*d_pageTableArray;

		/**
		 * The associated GPU side object
		 */
		KernelType d_cacheManagerKernel;

		/**
		 * Reference on a provider
		 */
		GPUProviderType* mProvider;

		//CUDPP
		size_t *d_numElementsPtr;
		CUDPPHandle scanplan;

		//Test CPU managment
	#if GPUCACHE_BENCH_CPULRU==1
		Array3D<uint> *cpuTimeStampArray;

		thrust::host_vector< uint >  *cpuTimeStampsElemAddressList;
		thrust::host_vector< uint >  *cpuTimeStampsElemAddressList2;
	#endif

	};

} // namespace GvCache

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvCacheManager.inl"

#endif
