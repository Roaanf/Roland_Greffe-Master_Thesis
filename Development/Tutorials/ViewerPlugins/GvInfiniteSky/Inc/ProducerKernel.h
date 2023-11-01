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
#include <GvCore/GsOracleRegionInfo.h>

// CUDA
#include <cuda_runtime.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
* Spheres ray-tracing parameters
*/
__constant__ unsigned int cNbSpheres;
__constant__ unsigned int cMinLevelOfResolutionToHandle;
__constant__ unsigned int cSphereBrickIntersectionType;
__constant__ bool cGeometricCriteria;
__constant__ unsigned int cMinNbSpheresPerBrick;
__constant__ bool cAbsoluteSizeCriteria;
__constant__ bool cFixedSizeSphere;
__constant__ bool cMeanSizeOfSpheres;
__constant__ double cCoeffAbsoluteSizeCriteria;
__constant__ float cSphereRadiusFader;

/**
 * Coefficient used to approximate a brick by a sphere
 *
 * brickRadius = ( 0.5f * sqrtf( 3.f ) ) * brickWidth;
 *
 * 0.5f * sqrtf( 3.f ) = 0,86602540378443864676372317075294
 */
__device__ static const float cBrickWidth2SphereRadius = 0.866025f;

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

	/**
	 * Data Structure device-side associated object
	 */
	typedef typename TDataStructureType::VolTreeKernelType DataStructureKernel;

	/**
	 * Type definition of the node tile resolution
	 */
	typedef typename TDataStructureType::NodeTileResolution NodeRes;

	/**
	 * Type definition of the brick resolution
	 */
	typedef typename TDataStructureType::BrickResolution BrickRes;

	/**
	 * Enumeration to define the brick border size
	 */
	enum
	{
		BorderSize = TDataStructureType::BrickBorderSize
	};

	/**
	 * CUDA block dimension used for nodes production (kernel launch)
	 */
	typedef GvCore::GsVec3D< 32, 1, 1 > NodesKernelBlockSize;

	/**
	 * CUDA block dimension used for bricks production (kernel launch)
	 */
	//typedef GvCore::GsVec3D< 16, 8, 1 > BricksKernelBlockSize;
	typedef GvCore::GsVec3D< 10, 10, 10 > BricksKernelBlockSize;
	/******************************* ATTRIBUTES *******************************/

	/**
	 * Initialize the producer
	 * 
	 * @param volumeTreeKernel Reference on a volume tree data structure
	 */
	inline void initialize( DataStructureKernel& pDataStructure );


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



	/**
	 * Set the buffer of spheres
	 *
	 * @param pSpheresBuffer the buffer of spheres (position and radius)
	 */
    inline void setPositionBuffer( float4* pSpheresBuffer );
    //inline void setPositionBuffer( std::vector<float4*> pSpheresBuffer );


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
	 * ...
	 */
	//GvCore::ArrayKernel3DLinear< float4 > _posBuf;
    //std::vector<float4*> _posBuf;
    float4* _posBuf;

	/**
	 * Data Structure device-side associated object
	 *
	 * Note : use this element if you need to sample data in cache
	 */
	DataStructureKernel _dataStructureKernel;

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
	inline GsOracleRegionInfo::OracleRegionInfo getRegionInfo( uint3 regionCoords, uint regionDepth );

	/**
	 * Test the intersection between a sphere and a brick
	 *
	 * @param pSphere sphere (position and and radius)
	 * @param pBrickCenter brick center
	 * @param pBoxExtent pBrickWidth brick width
	 *
	 * @return a flag to tell wheter or not intersection occurs
	 */
	__device__
	static inline bool intersectBrick( const float4 pSphere, const float3 pBrickCenter, const float pBrickWidth );

	/**
	 * Sphere-Sphere intersection test
	 *
	 * @param pSphereCenter 1st sphere center
	 * @param pSphereRadius 1stsphere radius
	 * @param pSphere 2nd sphere (position and and radius)
	 *
	 * @return a flag to tell wheter or not intersection occurs
	 */
	__device__
	static inline bool intersectSphereSphere( const float3 pSphereCenter, const float pSphereRadius, const float4 pSphere );

	/**
	 * Sphere-Box intersection test
	 *
	 * @param pBoxCenter box center
	 * @param pBoxExtent box extent (distance from center to one side)
	 * @param pSphere sphere (position and and radius)
	 *
	 * @return a flag to tell wheter or not intersection occurs
	 */
	__device__
	static inline bool intersectSphereBox( const float3 pBoxCenter, const float pBoxExtent, const float4 pSphere );

	/**
	 * Test wheter or not geometric criteria passes
	 *
	 * Note : the node subdivision process is stopped if there is no more than a given number of spheres inside
	 *
	 * @param pNbSpheresInBrick number of spheres in a given brick
	 *
	 * @return a flag to tell wheter or not the criteria passes
	 */
	__device__
	static inline bool isGeometricCriteriaValid( const unsigned int pNbSpheresInBrick );

	/**
	 * Test wheter or not screen based criteria passes
	 *
	 * Note : the node subdivision process is stopped if ...
	 *
	 * @return a flag to tell wheter or not the criteria passes
	 */
	__device__
	static inline bool isScreenSpaceCriteriaValid();

	/**
	 * Test wheter or not absolute size criteria passes
	 *
	 * Note : the node subdivision process is stopped if ...
	 *
	 * @return a flag to tell wheter or not the criteria passes
	 */
	__device__
    static inline bool isAbsoluteSizeCriteriaValid(float sphereRadius, float brickSize);
	
};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ProducerKernel.inl"

#endif // !_PRODUCER_KERNEL_H_
