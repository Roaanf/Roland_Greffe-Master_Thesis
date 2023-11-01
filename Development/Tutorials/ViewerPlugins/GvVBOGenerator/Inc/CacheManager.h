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

#ifndef _CACHE_MANAGER_H_
#define _CACHE_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvPerfMon/GsPerformanceMonitor.h>
#include <GvCache/GsCacheManager.h>

// CUDA
#include <vector_types.h>

// CUDA SDK
#include <helper_math.h>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// GigaVoxels
#include <GvCache/GsCacheManagerKernel.h>
#include <GvCore/GsArray.h>
#include <GvCore/GsLinearMemory.h>
#include <GvCore/GsFunctionalExt.h>
#include <GvCache/GsCacheManager.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

//// GigaVoxels
//namespace GvUtils
//{
//	class GvProxyGeometryHandler;
//}

// Project
class ParticleSystem;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
* @class CacheManager
*
* @brief The CacheManager class provides...
*
* @ingroup ...
*
* This class is used to manage a cache on the GPU
*
* Aide PARAMETRES TEMPLATES :
* dans VolumeTreeCache.h :
* - PageTableArrayType == GsLinearMemory< uint >
* - PageTableType == PageTableNodes< ... GsLinearMemoryKernel< uint > ... > ou PageTableBricks< ... >
* - GPUProviderType == IProvider< 1, GPUProducer > ou bien avec 0
*/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename TDataStructureType >
class CacheManager : public GvCache::GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the parent class
	 */
	typedef GvCache::GsCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType > ParentClass;

	/**
	 * Type definition for the GPU side associated object
	 *
	 * @todo pass this parameter as a template parameter in order to be able to overload this component easily
	 */
	typedef typename ParentClass::KernelType KernelType;

	/**
	 * Data structure
	 */
	TDataStructureType* _dataStructure;

	/**
	 * VBO
	 */
	//GvUtils::GvProxyGeometryHandler* _vbo;

		/**
	 * Particle system
	 */
	ParticleSystem* _particleSystem;
	
	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param cachesize
	 * @param pageTableArray
	 * @param graphicsInteroperability
	 */
	CacheManager( uint3 cachesize, PageTableArrayType* pageTableArray, /*GvUtils::GvProxyGeometryHandler* _vbo,*/ uint graphicsInteroperability = 0 );

	/**
	 * Destructor
	 */
	virtual ~CacheManager();

	/**
	 * Update VBO
	 */
	uint updateVBO( bool manageUpdatesOnly );

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
	 * VBO ...
	 */
	thrust::device_vector< uint >* _d_vboBrickList;

	/**
	 * VBO ...
	 */
	thrust::device_vector< uint >* _d_vboIndexOffsetList;
	
	/**
	 * CUDPP vbo SCAN PLAN
	 *
	 * A plan is a data structure containing state and intermediate storage space that CUDPP uses to execute algorithms on data.
	 */
	 CUDPPHandle _vboScanPlan;
	 uint _vboScanPlanSize;

	/******************************** METHODS *********************************/

	/**
	 * Get a CUDPP plan given a number of elements to be processed.
	 *
	 * @param pSize The maximum number of elements to be processed
	 *
	 * @return a handle on the plan
	 */
	CUDPPHandle getVBOScanPlan( uint pSize );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "CacheManager.inl"

#endif
