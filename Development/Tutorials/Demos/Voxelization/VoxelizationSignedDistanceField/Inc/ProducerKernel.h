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

#ifndef _PRODUCER_KERNEL_H_
#define _PRODUCER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvUtils/GsUtils.h>

// Cuda
#include <cuda_runtime.h>

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
 * @class ProducerKernel
 *
 * @brief The ProducerKernel class provides the mecanisms to produce data
 * on device following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from GPU, for instance,
 * procedurally generating data (apply noise patterns, etc...).
 *
 * This class is implements the mandatory functions of the GsIProviderKernel base class.
 *
 * @param NodeRes Node tile resolution
 * @param BrickRes Brick resolution
 * @param BorderSize Border size of bricks
 * @param VolTreeKernelType Device-side data structure
 */
template< typename TDataStructureType >
class ProducerKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/** @name Mandatory type definitions
	 *
	 * Due to the use of templated classes,
	 * type definitions are used to ease use development.
	 * Some are aloso required to follow the GigaVoxels pipeline flow sequences.
	 */
	///@{

	/**
	 * MACRO
	 * 
	 * Useful and required type definition for producer kernels
	 * - it is used to access the DataStructure typedef passed in argument
	 *
	 * @param TDataStructureType a data structure type (should be the template parameter of a Producer Kernel)
	 */
	GV_MACRO_PRODUCER_KERNEL_REQUIRED_TYPE_DEFINITIONS( TDataStructureType )

	/**
	 * CUDA block dimension used for nodes production (kernel launch)
	 */
	typedef GvCore::GsVec3D< 32, 1, 1 > NodesKernelBlockSize;

	/**
	 * CUDA block dimension used for bricks production (kernel launch)
	 */
	typedef GvCore::GsVec3D< 16, 8, 1 > BricksKernelBlockSize;

	///@}

	/**
	 * Defines the data type list
	 */
	typedef typename TDataStructureType::DataTypeList DataTList;

	/**
	 * Brick full resolution
	 */
	typedef GvCore::GsVec3D
	<
		BrickRes::x + 2 * BorderSize,
		BrickRes::y + 2 * BorderSize,
		BrickRes::z + 2 * BorderSize
	>
	BrickFullRes;

	/**
	 * Brick pool kernel type
	 */
	typedef GvCore::GPUPoolKernel< GvCore::GsLinearMemoryKernel, DataTList >	BricksPoolKernelType;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Initialize the producer
	 * 
	 * @param volumeTreeKernel Reference on a volume tree data structure
	 */
	inline void initialize( DataStructureKernel& pDataStructure );

	/**
	 * Produce data on device.
	 * Implement the produceData method for the channel 0 (nodes)
	 *
	 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
	 *
	 * In the function, user has to produce data for a node tile or a brick of voxels :
	 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
	 * etc...
	 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
	 * user had previously defined (color, normal, density, etc...)
	 *
	 * @param pGpuPool The device side pool (nodes or bricks)
	 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
	 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
	 * @param pNewElemAddress The address at which to write the produced data in the pool
	 * @param pParentLocInfo The localization info used to locate an element in the pool
	 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
	 *
	 * @return A feedback value that the user can return.
	 */
	template< typename GPUPoolKernelType >
	__device__
	inline uint produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID,
							 uint3 newElemAddress, const GvCore::GsLocalizationInfo& parentLocInfo, Loki::Int2Type< 0 > );

	/**
	 * Produce data on device.
	 * Implement the produceData method for the channel 1 (bricks)
	 *
	 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
	 *
	 * In the function, user has to produce data for a node tile or a brick of voxels :
	 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
	 * etc...
	 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
	 * user had previously defined (color, normal, density, etc...)
	 *
	 * @param pGpuPool The device side pool (nodes or bricks)
	 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
	 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
	 * @param pNewElemAddress The address at which to write the produced data in the pool
	 * @param pParentLocInfo The localization info used to locate an element in the pool
	 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
	 *
	 * @return A feedback value that the user can return.
	 */
	template< typename GPUPoolKernelType >
	__device__
	inline uint produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID,
							 uint3 newElemAddress, const GvCore::GsLocalizationInfo& parentLocInfo, Loki::Int2Type< 1 > );

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

	/**
	 * Data Structure device-side associated object
	 *
	 * Note : use this element if you need to sample data in cache
	 */
	//DataStructureKernel _dataStructureKernel;

	/******************************** METHODS *********************************/

	/**
	 * Helper fonction to compute the normal from potential function
	*
	* @param dataPool The device side pool (nodes or bricks)
	* @param newElemAddress The address at which to write the produced data in the pool
	* @param locOffset Position index
	* @param levelResInc Inv of the number of voxels at the current brick resolution
	* @param elemSize 
	*
	* @return ...
	*/
	template< typename GPUPoolKernelType >
	__device__
	inline float3 getNormal( GPUPoolKernelType& dataPool, uint3 newElemAddress, uint3 locOffset, float3 levelResInv, uint3 elemSize );

	/**
	* Helper function to translate offset if we are in the border
	*
	* @param offset
	*
	* @return offset translated
	*/
	__device__
	inline void tranlateOffset( uint3& offset, const uint3& elemSize );
};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ProducerKernel.inl"

#endif // !_PRODUCER_KERNEL_H_
