/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/** 
 * @version 1.0
 */

#ifndef _SPHEREPRODUCERKERNEL_H_
#define _SPHEREPRODUCERKERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GPUVoxelProducer.h>

// CUDA
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
 * @class SphereProducerKernel
 *
 * @brief The SphereProducerKernel class provides the mecanisms to produce data
 * on device following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from GPU, for instance,
 * procedurally generating data (apply noise patterns, etc...).
 *
 * This class is implements the mandatory functions of the GvIProviderKernel base class.
 *
 * @param NodeRes Node tile resolution
 * @param BrickRes Brick resolution
 * @param BorderSize Border size of bricks
 * @param VolTreeKernelType Device-side data structure
 */
template< typename NodeRes, typename BrickRes, uint BorderSize, typename VolTreeKernelType >
class SphereProducerKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

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
							uint3 newElemAddress, const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 0 > );

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
							uint3 newElemAddress, const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 1 > );

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
	 * Helper function used to determine the type of zones in the data structure.
	 *
	 * The data structure is made of regions containing data, empty or constant regions.
	 * Besides, this function can tell if the maximum resolution is reached in a region.
	 *
	 * @param regionCoords region coordinates
	 * @param regionDepth region depth
	 *
	 * @return the type of the region
	 */
	__device__
	inline GPUVoxelProducer::GPUVPRegionInfo getRegionInfo( uint3 regionCoords, uint regionDepth );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "SphereProducerKernel.inl"

#endif // !_SPHEREPRODUCERKERNEL_H_
