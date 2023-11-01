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

//#ifndef _VOLUME_TREE_CACHE_INL_
//#define _VOLUME_TREE_CACHE_INL_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda SDK
#include <helper_math.h>

// GigaVoxels
#include "GvPerfMon/GsPerformanceMonitor.h"
#include "GvCore/GsFunctionalExt.h"
#include "GvCore/GsError.h"
//#include <GvUtils/GvProxyGeometryHandler.h>

#include "GsCompute/GsDataParallelPrimitives.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

	/******************************************************************************
	 * Constructor
	 *
	 * @param pDataStructure a pointer to the data structure.
	 * @param gpuprod a pointer to the user's producer.
	 * @param nodepoolres the 3d size of the node pool.
	 * @param brickpoolres the 3d size of the brick pool.
	 * @param graphicsInteroperability ...
	 ******************************************************************************/
	template< typename TDataStructureType, typename TPriorityPoliciesManager >
	VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >
	::VolumeTreeCache( TDataStructureType* pDataStructure, uint3 nodepoolres, uint3 brickpoolres/*, GvUtils::GvProxyGeometryHandler* _vbo*/, uint graphicsInteroperability )
	:	GvStructure::GsDataProductionManager< TDataStructureType, TPriorityPoliciesManager >( pDataStructure, nodepoolres, brickpoolres, graphicsInteroperability )
	,	_vboCacheManager( NULL )
	//,	_vbo( NULL )
	{
		// Cache managers creation : nodes and bricks
		_vboCacheManager = new VBOCacheManagerType( this->_brickPoolRes, this->_dataStructure->_dataArray/*, _vbo*/, graphicsInteroperability );

		// TEST VBO ----
		//_vbo = _vboCacheManager->_vbo;
		//--------------
		
		// The creation of the localization arrays should be moved here!
		_vboCacheManager->_pageTable->locCodeArray = this->_dataStructure->_localizationCodeArray;
		_vboCacheManager->_pageTable->locDepthArray = this->_dataStructure->_localizationDepthArray;
		_vboCacheManager->_pageTable->getKernel().childArray = this->_dataStructure->_childArray->getDeviceArray();
		_vboCacheManager->_pageTable->getKernel().dataArray = this->_dataStructure->_dataArray->getDeviceArray();
		_vboCacheManager->_pageTable->getKernel().locCodeArray = this->_dataStructure->_localizationCodeArray->getDeviceArray();
		_vboCacheManager->_pageTable->getKernel().locDepthArray = this->_dataStructure->_localizationDepthArray->getDeviceArray();
		_vboCacheManager->_totalNumLoads = 0;
		_vboCacheManager->_lastNumLoads = 0;
		_vboCacheManager->_dataStructure = this->_dataStructure;

		_vboVolumeTreeCacheKernel._updateBufferArray = this->_dataProductionManagerKernel._updateBufferArray;
		_vboVolumeTreeCacheKernel._nodeCacheManager = this->_dataProductionManagerKernel._nodeCacheManager;
		_vboVolumeTreeCacheKernel._brickCacheManager = this->_dataProductionManagerKernel._brickCacheManager;
		_vboVolumeTreeCacheKernel._vboCacheManager = this->_vboCacheManager->getKernelObject();
	}

	/******************************************************************************
	 * Destructor
	 ******************************************************************************/
	template< typename TDataStructureType, typename TPriorityPoliciesManager >
	VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >
	::~VolumeTreeCache()
	{
		// Delete cache manager (nodes and bricks)
		delete _vboCacheManager;
	}

	/******************************************************************************
	 * This method is called before the rendering process. We just clear the request buffer.
	 ******************************************************************************/
	template< typename TDataStructureType, typename TPriorityPoliciesManager >
	void VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >
	::preRenderPass()
	{
		CUDAPM_START_EVENT( gpucache_preRenderPass );

		// Clear subdiv pool
		this->_updateBufferArray->fill( 0 );

		// Number of requests cache has handled
		this->_nbNodeSubdivisionRequests = 0;
		this->_nbBrickLoadRequests = 0;

	#if CUDAPERFMON_CACHE_INFO==1
		_nodesCacheManager->_d_CacheStateBufferArray->fill( 0 );
		_nodesCacheManager->_numPagesUsed = 0;
		_nodesCacheManager->_numPagesWrited = 0;

		_bricksCacheManager->_d_CacheStateBufferArray->fill( 0 );
		_bricksCacheManager->_numPagesUsed = 0;
		_bricksCacheManager->_numPagesWrited = 0;
	#endif

		//---------------------------------------
		// TO DO : clear vbo used flags ?
		//_vboCacheManager->
		//---------------------------------------

		CUDAPM_STOP_EVENT( gpucache_preRenderPass );
	}

	/******************************************************************************
	 * This method is called after the rendering process. She's responsible for processing requests.
	 *
	 * @return the number of requests processed.
	 ******************************************************************************/
	template< typename TDataStructureType, typename TPriorityPoliciesManager >
	uint VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >
	::handleRequests()
	{
		// Generate the requests buffer
		//
		// Collect and compact update informations for both nodes and bricks
		CUDAPM_START_EVENT( dataProduction_manageRequests );
		uint nbRequests = this->manageUpdates();
		CUDAPM_STOP_EVENT( dataProduction_manageRequests );

		// Stop post-render pass if no request
	//	if ( nbRequests > 0 )
	//	{
			// Update time stamps
			CUDAPM_START_EVENT( cache_updateTimestamps );
			this->updateTimeStamps();
			CUDAPM_STOP_EVENT( cache_updateTimestamps );

			//-------------------------------------------------------------------------------------------------------------
			//
			// BEGIN : VBO Generation
			//
			this->_numBricksNotInUse = _vboCacheManager->updateVBO( this->_intraFramePass );
			//
			// END : VBO Generation
			//
			//-------------------------------------------------------------------------------------------------------------

			// Handle requests :

			// [ 1 ] - Handle the "subdivide nodes" requests
			CUDAPM_START_EVENT( producer_nodes );
			//uint numSubDiv = manageSubDivisions( nbRequests );
			this->_nbNodeSubdivisionRequests = this->manageNodeProduction( nbRequests );
			CUDAPM_STOP_EVENT( producer_nodes );

			//  [ 2 ] - Handle the "load/produce bricks" requests
			CUDAPM_START_EVENT( producer_bricks );
			//if ( numSubDiv < nbRequests )
			if ( this->_nbNodeSubdivisionRequests < nbRequests )
			{
				this->_nbBrickLoadRequests = this->manageDataProduction( nbRequests );
			}
			CUDAPM_STOP_EVENT( producer_bricks );
		//}

		return nbRequests;
	}

	/******************************************************************************
	 * This method destroy the current N-tree and clear the caches.
	 ******************************************************************************/
	template< typename TDataStructureType, typename TPriorityPoliciesManager >
	void VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >
	::clearCache()
	{
		// Launch Kernel
		dim3 blockSize( 32, 1, 1 );
		dim3 gridSize( 1, 1, 1 );
		GvStructure::GsKernel_ClearVolTreeRoot<<< gridSize, blockSize >>>( this->_dataStructure->volumeTreeKernel, NodeTileRes::getNumElements() );

		GV_CHECK_CUDA_ERROR( "ClearVolTreeRoot" );

		// Reset nodes cache manager
		this->_nodesCacheManager->clearCache();
		this->_nodesCacheManager->_totalNumLoads = 2;
		this->_nodesCacheManager->_lastNumLoads = 1;

		// Reset bricks cache manager
		this->_bricksCacheManager->clearCache();
		this->_bricksCacheManager->_totalNumLoads = 0;
		this->_bricksCacheManager->_lastNumLoads = 0;

		// VBO
		_vboCacheManager->clearCache();
		_vboCacheManager->_totalNumLoads = 0;
		_vboCacheManager->_lastNumLoads = 0;
	}

	/******************************************************************************
	 * Get the associated device-side object
	 *
	 * @return The device-side object
	 ******************************************************************************/
	template< typename TDataStructureType, typename TPriorityPoliciesManager >
	inline VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >
	::VBOVolumeTreeCacheKernelType VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >::getVBOKernelObject() const
	{
		return _vboVolumeTreeCacheKernel;
	}

	/******************************************************************************
	 * Get the bricks cache manager
	 *
	 * @return the bricks cache manager
	 ******************************************************************************/
	template< typename TDataStructureType, typename TPriorityPoliciesManager >
	inline const VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >::VBOCacheManagerType*
	VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >::getVBOCacheManager() const
	{
		return _vboCacheManager;
	}

	/******************************************************************************
	 * Get the bricks cache manager
	 *
	 * @return the bricks cache manager
	 ******************************************************************************/
	template< typename TDataStructureType, typename TPriorityPoliciesManager >
	inline VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >::VBOCacheManagerType*
	VolumeTreeCache< TDataStructureType, TPriorityPoliciesManager >::editVBOCacheManager()
	{
		return _vboCacheManager;
	}
