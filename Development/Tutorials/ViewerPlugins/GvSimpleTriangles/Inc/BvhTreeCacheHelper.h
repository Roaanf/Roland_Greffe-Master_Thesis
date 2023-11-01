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

#ifndef _BVH_TREE_CACHE_HELPER_H_
#define _BVH_TREE_CACHE_HELPER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include "BvhTreeCacheHelper.hcu"

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

/** 
 * @class GPUCacheHelper
 *
 * @brief The GPUCacheHelper class provides...
 *
 * ...
 */
class BvhTreeCacheHelper
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * This method is a helper for writing into the cache.
	 *
	 * @param pNumElements The number of elements we need to produce and write.
	 * @param pNodesAddressList The numElements nodes concerned by the production.
	 * @param pElemAddressList The numElements addresses of the new elements.
	 * @param pGpuPool The pool where we will write the produced elements.
	 * @param pGpuProvider The provider called for the production.
	 * @param pPageTable 
	 * @param pBlockSize The user defined blockSize used to launch the kernel.
	 */
	template< typename ElementRes, typename BvhTreeType, typename PoolType, typename ProviderType >
	void genericWriteIntoCache( uint pNumElements, uint* pNodesAddressList, uint* pElemAddressList,
								const BvhTreeType& bvhTree, const PoolType& pGpuPool, const ProviderType& pGpuProvider,
								const dim3& pBlockSize )
	{
		// Define kernel grid size
		dim3 gridSize( std::min( pNumElements, 65535U ), iDivUp( pNumElements, 65535U ), 1 );

		// Launch kernel to produce data on device (node subdivision or data load/production)
		GenericWriteIntoCache< ElementRes >
			<<< gridSize, pBlockSize >>>
			( pNumElements, pNodesAddressList, pElemAddressList, bvhTree->getKernelObject(), pGpuPool->getKernelPool(), pGpuProvider );

		GV_CHECK_CUDA_ERROR( "GenericWriteIntoCache" );
	}

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

};

#endif // !_BVH_TREE_CACHE_HELPER_H_
