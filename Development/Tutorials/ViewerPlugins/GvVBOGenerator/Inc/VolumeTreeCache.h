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

#ifndef _VOLUME_TREE_CACHE_H_
#define _VOLUME_TREE_CACHE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <iostream>

// Cuda
#include <vector_types.h>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// GigaVoxels
#include <GvCore/GsArray.h>
#include <GvCore/GsLinearMemory.h>
#include <GvCore/GsRendererTypes.h>
#include <GvCore/GsPool.h>
#include <GvCore/GsVector.h>
#include <GvCore/GsPageTable.h>
#include <GvCore/GsIProvider.h>
#include <GvRendering/GsRendererHelpersKernel.h>
#include <GvCore/GsOracleRegionInfo.h>
#include <GvCore/GsLocalizationInfo.h>
#include <GvCache/GsCacheManager.h>
#include <GvPerfMon/GsPerformanceMonitor.h>
#include <GvStructure/GsVolumeTreeAddressType.h>
#include <GvStructure/GsDataProductionManager.h>
//#include <GvStructure/GsVolumeTreeCacheKernel.h>

// cudpp
#include <cudpp.h>

// Project
#include "CacheManager.h"
#include "VolumeTreeCacheKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

//// VBO
//namespace GvUtils
//{
//	class GvProxyGeometryHandler;
//}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/
	
	/** 
	 * @class VolumeTreeCache
	 *
	 * @brief The VolumeTreeCache class provides the concept of cache.
	 *
	 * This class implements the cache mechanism for the VolumeTree data structure.
	 * As device memory is limited, it is used to store the least recently used element in memory.
	 * It is responsible to handle the data requests list generated during the rendering process.
	 * (ray-tracing - N-tree traversal).
	 * Request are then sent to producer to load or produced data on the host or on the device.
	 *
	 * @param TDataStructureType The volume tree data structure (nodes and bricks)
	 * @param ProducerType The associated user producer (host or device)
	 * @param NodeTileRes The user defined node tile resolution
	 * @param BrickFullRes The user defined brick resolution
	 */
	template< typename TDataStructureType, typename TPriorityPoliciesManager >
	class VolumeTreeCache : public GvStructure::GsDataProductionManager< TDataStructureType, TPriorityPoliciesManager >
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

	public:

		/****************************** INNER TYPES *******************************/

		/**
		 * Type definition of the parent class
		 */
		typedef GvStructure::GsDataProductionManager< TDataStructureType > ParentClass;

		/**
		 * Type definition of the node tile resolution
		 */
		typedef typename TDataStructureType::NodeTileResolution NodeTileRes;

		/**
		 * Type definition for the bricks cache manager
		 */
		typedef CacheManager
		<
			1, typename ParentClass::BrickFullRes, GvStructure::VolTreeBrickAddress, GvCore::GsLinearMemory< uint >, typename ParentClass::BrickPageTableType, TDataStructureType
		>
		VBOCacheManagerType;

		/**
		 * Type definition for the associated device-side object
		 */
		typedef VolumeTreeCacheKernel
		<
			typename ParentClass::NodeTileResLinear, typename ParentClass::BrickFullRes, GvStructure::VolTreeNodeAddress, GvStructure::VolTreeBrickAddress, TPriorityPoliciesManager
		>
		VBOVolumeTreeCacheKernelType;

		/******************************* ATTRIBUTES *******************************/

		/**
		 * VBO
		 */
		//GvUtils::GvProxyGeometryHandler* _vbo;

		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 *
		 * @param pDataStructure a pointer to the data structure.
		 * @param gpuprod a pointer to the user's producer.
		 * @param nodepoolres the 3d size of the node pool.
		 * @param brickpoolres the 3d size of the brick pool.
		 * @param graphicsInteroperability Graphics interoperabiliy flag
		 */
		VolumeTreeCache( TDataStructureType* pDataStructure, uint3 nodepoolres, uint3 brickpoolres/*, GvUtils::GvProxyGeometryHandler* _vbo*/, uint graphicsInteroperability = 0 );

		/**
		 * Destructor
		 */
		virtual ~VolumeTreeCache();

		/**
		 * This method is called before the rendering process. We just clear the request buffer.
		 */
		virtual void preRenderPass();

		/**
		 * This method is called after the rendering process. She's responsible for processing requests.
		 *
		 * @return the number of requests processed.
		 *
		 * @todo Check wheter or not the inversion call of updateTimeStamps() with manageUpdates() has side effects
		 */
		virtual uint handleRequests();

		/**
		 * This method destroy the current N-tree and clear the caches.
		 */
		virtual void clearCache();

		/**
		 * Get the associated device-side object
		 *
		 * @return The device-side object
		 */
		inline VBOVolumeTreeCacheKernelType getVBOKernelObject() const;

		/**
		 * Get the VBO cache manager
		 *
		 * @return the VBO cache manager
		 */
		inline const VBOCacheManagerType* getVBOCacheManager() const;

		/**
		 * Get the VBO cache manager
		 *
		 * @return the VBO cache manager
		 */
		inline VBOCacheManagerType* editVBOCacheManager();

		/**************************************************************************
		 **************************** PROTECTED SECTION ***************************
		 **************************************************************************/

	protected:

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**************************************************************************
		 ***************************** PRIVATE SECTION ****************************
		 **************************************************************************/

	private:

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/**
		 * The associated device-side object
		 */
		VBOVolumeTreeCacheKernelType _vboVolumeTreeCacheKernel;

		/**
		 * VBO cache manager
		 */
		VBOCacheManagerType* _vboCacheManager;

		/******************************** METHODS *********************************/

	};


/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "VolumeTreeCache.inl"

#endif
