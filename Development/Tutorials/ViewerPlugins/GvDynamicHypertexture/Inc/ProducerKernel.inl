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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvStructure/GsNode.h>
#include <GvUtils/GsNoiseKernel.h>
//#include <GvStructure/GsVolumeTree.h>

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/**
 * Volume texture (signed distance field : 3D normal + distance)
 */
texture< float4, cudaTextureType3D, cudaReadModeElementType > volumeTex;

/******************************************************************************
 * Get the RGBA data of distance field + noise's upper bound.
 *
 * @param voxelPosF 3D position in the current brick
 * @param levelRes number of voxels at the current brick resolution
 * @param levelDepth Depth of the current node
 *
 * @return computed RGBA color
 ******************************************************************************/
__device__
uchar getRGBA( float3 voxelPosF, uint3 levelRes, uint levelDepth )
{
	// Retrieve "normal" and "distance" from signed distance fied of user's 3D model
	float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );

	// Normalize the retrieved normal
	// float3 voxelNormal = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );

	// Putting red opaque everywhere
	//float4 color = make_float4( 1.f , 0.f , 0.f , 1.f );
	uchar color = 255;
	
	float noiseShellWidth = cNoiseShellWidthUpperBound * 0.5f;

	// We put zero in the opacity if the point is out of the reach of the noise
	float testDistance = 0.f;
	float DistanceTested = 0.f;

	switch ( cNoiseFillTheShape )
	{
		case 0:
			// Normal Displacement
			// - with fill the shape
			testDistance = voxelNormalAndDist.w;
			DistanceTested = noiseShellWidth;
			break;

		case 1:
			// Color Displacement
			testDistance = voxelNormalAndDist.w;
			noiseShellWidth = 0;
			// TO DO : erreur ?
			// - noiseShellWidth = 0 => why ? AND DistanceTested not initialized ?
			break;

		case 2:
			// Normal Displacement
			// - without fill the shape
			testDistance = fabs( voxelNormalAndDist.w );
			DistanceTested = noiseShellWidth;
			break;

		default:
			// Error
			//assert( false );
			break;
	}
	
	if ( testDistance > DistanceTested  )
	{
		color = 0;
	}

	return color;
}

/******************************************************************************
 * Get the normal of distance field + noise
 *
 * @param voxelPosF 3D position in the current brick
 * @param levelRes number of voxels at the current brick resolution
 *
 * @return ...
 ******************************************************************************/
__device__
float3 getNormal( float3 voxelPosF )
{
	// Retrieve "normal" and "distance" from signed distance fied of user's 3D model
	float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );

	float3 voxelNormal = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );

	return voxelNormal;
}

/******************************************************************************
 * Initialize the producer
 * 
 * @param volumeTreeKernel Reference on a volume tree data structure
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::initialize( DataStructureKernel& pDataStructure )
{
	//_dataStructureKernel = pDataStructure;
}

/******************************************************************************
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
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GsLocalizationInfo& parentLocInfo, Loki::Int2Type< 0 > )
{
	// NOTE :
	// In this method, you are inside a node tile.
	// The goal is to determine, for each node of the node tile, which type of data it holds.
	// Data type can be :
	// - a constant region,
	// - a region with data,
	// - a region where max resolution is reached.
	// So, one thread is responsible of the production of one node of a node tile.
	
	// Retrieve current node tile localization information code and depth
	const GvCore::GsLocalizationInfo::CodeType *parentLocCode = &parentLocInfo.locCode;
	const GvCore::GsLocalizationInfo::DepthType *parentLocDepth = &parentLocInfo.locDepth;

	// Process ID gives the 1D index of a node in the current node tile
	if ( processID < NodeRes::getNumElements() )
	{
		// First, compute the 3D offset of the node in the node tile
		uint3 subOffset = NodeRes::toFloat3( processID );

		// Node production corresponds to subdivide a node tile.
		// So, based on the index of the node, find the node child.
		// As we want to sudbivide the curent node, we retrieve localization information
		// at the next level
		uint3 regionCoords = parentLocCode->addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth->addLevel().get();

		// Create a new node for which you will have to fill its information.
		GvStructure::GsNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		// Call what we call an oracle that will determine the type of the region of the node accordingly
		GsOracleRegionInfo::OracleRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		// Now that the type of the region is found, fill the new node information
		if ( nodeinfo == GsOracleRegionInfo::GPUVP_CONSTANT )
		{
			newnode.setTerminal( true );
		}
		else if ( nodeinfo == GsOracleRegionInfo::GPUVP_DATA )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GsOracleRegionInfo::GPUVP_DATA_MAXRES )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}

		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// newElemAddress.x + processID : is the adress of the new node in the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( newElemAddress.x + processID, newnode.childAddress );
		nodePool.getChannel( Loki::Int2Type< 1 >() ).set( newElemAddress.x + processID, newnode.brickAddress );
	}

	return 0;
}

/******************************************************************************
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
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GsLocalizationInfo& parentLocInfo, Loki::Int2Type< 1 > )
{
	// NOTE :
	// In this method, you are inside a brick of voxels.
	// The goal is to determine, for each voxel of the brick, the value of each of its channels.
	//
	// Data type has previously been defined by the user, as a
	// typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;
	//
	// In this tutorial, we have choosen two channels containing color at channel 0 and normal at channel 1.

	// Retrieve current brick localization information code and depth
	const GvCore::GsLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GsLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;
	//printf( "\nDepth = %d", parentLocDepth.get() );

	// Shared memory declaration
	//
	// Each threads of a block process one and unique brick of voxels.
	// We store in shared memory common useful variables that will be used by each thread.
	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes;
	__shared__ float3 levelResInv;
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;
	__shared__ uint3 regionDepth;

	// TO DO :
	// - use synchonization to compute values by one unique thread
	// ...

	// Compute useful variables used for retrieving positions in 3D space
	brickRes = BrickRes::get();
	regionDepth = make_uint3( 1 << parentLocDepth.get() );
	levelRes =  regionDepth * brickRes;
	levelResInv = make_float3( 1.f ) / make_float3( levelRes );

	brickPos = make_int3( parentLocCode.get() * brickRes) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	// Real brick size (with borders)
	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );

	// The original KERNEL execution configuration on the HOST has a 2D block size :
	// dim3 blockSize( 16, 8, 1 );
	//
	// Each block process one brick of voxels.
	//
	// One thread iterate in 3D space given a pattern defined by the 2D block size
	// and the following "for" loops. Loops take into account borders.
	// In fact, each thread of the current 2D block compute elements layer by layer
	// on the z axis.
	//
	// One thread process only a subset of the voxels of the brick.
	//
	// Iterate through z axis step by step as blockDim.z is equal to 1
	uint3 elemOffset;
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x )
			{
				// Compute position index
				const uint3 locOffset = elemOffset + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z );

				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					const float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					const float3 voxelPosF = brickPosF + voxelPosInBrickF;

					// Compute data
					const uchar voxelColor = getRGBA( voxelPosF, levelRes, regionDepth.x );

					// Compute the new element's address
					const uint3 destAddress = newElemAddress + locOffset;
					// Write the voxel's color in the first field
					dataPool.template setValue< 0 >( destAddress, voxelColor );
				}
			}
		}
	}

	return 0;
}

/******************************************************************************
 * Helper function used to determine the type of zones in the data structure.
 *
 * The data structure is made of regions containing data, empty or constant regions.
 * Besides, this function can tell if the maximum resolution is reached in a region.
 *
 * @param regionCoords region coordinates
 * @param regionDepth region depth
 *
 * @return the type of the region
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline GsOracleRegionInfo::OracleRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth )
{
	//if (regionDepth <= 4)
	//return GsOracleRegionInfo::GPUVP_DATA;

	// Limit the depth.
	// Currently, 32 is the max depth of the GigaVoxels engine.
	if ( regionDepth >= 32 )
	{
		return GsOracleRegionInfo::GPUVP_DATA_MAXRES;
	}

	// TO DO :
	// - some parameter are constants to nodes (brickRes, levelRes and levelResInv)
	// ...
	uint3 brickRes;
	uint3 levelRes;
	float3 levelResInv;
	int3 brickPos;
	float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3( 1 << regionDepth ) * brickRes;
	levelResInv = make_float3( 1.f ) / make_float3( levelRes );

	brickPos = make_int3( regionCoords * brickRes ) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	uint3 elemOffset;

	bool isEmpty = true;

	// Iterate through voxels
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z && isEmpty; elemOffset.z++ )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y && isEmpty; elemOffset.y++ )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x && isEmpty; elemOffset.x++ )
			{
				uint3 locOffset = elemOffset;// + make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					const float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					const float3 voxelPosF = brickPosF + voxelPosInBrickF;

					// Test opacity to determine if there is data
					const uchar voxelColor = getRGBA( voxelPosF, levelRes, regionDepth );
					if ( voxelColor > 0 )
					{
						isEmpty = false;

						// TO DO
						// - here, we can stop and exit
						// ...
						// NOTE :
						// - attention, il y a un test bool�en isEmpty � chaque fois
						// - chercher quel est le plus rapide
						// ...
					}
				}
			}
		}
	}

	if ( isEmpty )
	{
		return GsOracleRegionInfo::GPUVP_CONSTANT;
	}

	return GsOracleRegionInfo::GPUVP_DATA;
}
