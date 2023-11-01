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

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

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
		//
		// TO DO
		// Question : is addLevel() valid for general N-Tree ? In the code, there are "*" and "/" by log2X log2y et log2z.
		const uint3 regionCoords = parentLocCode->addLevel< NodeRes >( subOffset ).get();
		const uint regionDepth = parentLocDepth->addLevel().get();

		/*if ( requestID == 0 )
		{
			printf( "produce node: %d, %d, %d. depth = %d\n", regionCoords.x, regionCoords.y, regionCoords.z, regionDepth );
		}*/

		// Create a new node for which you will have to fill its information.
		GvStructure::GsNode newNode;
		newNode.childAddress = 0;
		newNode.brickAddress = 0;

		// Call what we call an oracle that will determine the type of the region of the node accordingly
		GsOracleRegionInfo::OracleRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		// Now that the type of the region is found, fill the new node information
		if ( nodeinfo == GsOracleRegionInfo::GPUVP_CONSTANT )
		{
			newNode.setTerminal( true );
		}
		else if ( nodeinfo == GsOracleRegionInfo::GPUVP_DATA )
		{
			newNode.childAddress = newElemAddress.x;

			newNode.setStoreBrick();
			newNode.setTerminal( false );
		}
		else if ( nodeinfo == GsOracleRegionInfo::GPUVP_DATA_MAXRES )
		{
			newNode.setStoreBrick();
			newNode.setTerminal( true );
		}

		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// newElemAddress.x + processID : is the adress of the new node in the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( newElemAddress.x + processID, newNode.childAddress );
		nodePool.getChannel( Loki::Int2Type< 1 >() ).set( newElemAddress.x + processID, newNode.brickAddress );
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
	// typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;
	//
	// In this tutorial, we have choosen one channel containing color at channel 0.
	
	// Retrieve current brick localization information code and depth
	//const GvCore::GsLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GsLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	/*if ( processID == 0 )
	{
		printf( "produce brick: %d, %d, %d. depth = %d\n", parentLocCode.get().x, parentLocCode.get().y, parentLocCode.get().z, parentLocDepth.get() );
	}*/

	// Shared memory declaration
	//
	// Each threads of a block process one and unique brick of voxels.
	// We store in shared memory common useful variables that will be used by each thread.
	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	
	int3 brickPos;
	float3 brickPosF;

	// Compute useful variables used for retrieving positions in 3D space
	brickRes = BrickRes::get();
	levelRes = make_uint3( powf( 3.0f, static_cast< float >( parentLocDepth.get() ) ) ) * brickRes;
	levelResInv = make_float3( 1.0f ) / make_float3( levelRes );

	brickPos = make_int3( 0 );

	// FIXME : __shared__
	GvCore::GvLocalizationCode locBrickCode;
	uint3 locBrickIdx;
	uint3 locBrickRes;

	locBrickCode = parentLocInfo.locCode;
	locBrickIdx = locBrickCode.get();
	locBrickRes = brickRes;

	while ( locBrickIdx.x != 0 || locBrickIdx.y != 0 || locBrickIdx.z != 0 )
	{
		brickPos += make_int3( ( locBrickIdx.x & 3 ) * locBrickRes.x, ( locBrickIdx.y & 3 ) * locBrickRes.y, ( locBrickIdx.z & 3 ) * locBrickRes.z );
		locBrickCode = locBrickCode.removeLevel< NodeRes >();
		locBrickIdx = locBrickCode.get();
		locBrickRes *= 3;
	}

	brickPos -= BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	// Real brick size (with borders)
	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	
	// The original KERNEL execution configuration on the HOST has a 2D block size :
	// dim3 blockSize( 8, 8, 1 );
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
				uint3 locOffset = elemOffset + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z );

				// Test if the computed position index is inside the brick (with borders)
				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					float3 voxelPosF = brickPosF + voxelPosInBrickF;
					// Position of the current voxel's center (scaled to the range [-1.0;1.0])
					float3 posF = voxelPosF * 2.0f - 1.0f;

					// Position into the brick.
					// float3 posInBrickF = ( make_float3( locOffset ) + 0.5f ) / make_float3( brickRes );

					float4 voxelColor = make_float4( 0.0f );
					//float4 voxelColor = make_float4( 0.7f, 0.8f, 0.6f, 0.0f );
					//float4 voxelColor = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
					//float4 voxelNormal = make_float4( normalize( posInBrickF * 2.0f - 1.0f ), 1.0f );

					if ( posF.x >= -1.f && posF.x <= 1.f &&
						posF.y >= -1.f && posF.y <= 1.f &&
						posF.z >= -1.f && posF.z <= 1.f )
					{
						if ( locOffset.x > 0 && locOffset.x < 9 &&
							locOffset.y > 0 && locOffset.y < 9 &&
							locOffset.z > 0 && locOffset.z < 9 )
						{
							// voxelColor.w = 1.0f;
							voxelColor = make_float4( 0.7f, 0.8f, 0.6f, 1.0f );
						}
					}

					//// Alpha pre-multiplication used to avoid the "color bleeding" effect
					//voxelColor.x *= voxelColor.w;
					//voxelColor.y *= voxelColor.w;
					//voxelColor.z *= voxelColor.w;

					// Compute the new element's address
					uint3 destAddress = newElemAddress + locOffset;
					// Write the voxel's color in the first field
					dataPool.template setValue< 0 >( destAddress, voxelColor );
					//// Write the voxel's normal in the second field
					//dataPool.template setValue< 1 >( destAddress, voxelNormal );
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
	//// Limit the depth.
	//// Currently, 32 is the max depth of the GigaVoxels engine.
	//if ( regionDepth >= 32 )
	//{
	//	return GsOracleRegionInfo::GPUVP_DATA_MAXRES;
	//}

	// Empty positions of a 3x3 cube (of a Menger Sponge's model)
	uint3 emptyFaces[ 7 ] =
	{
		// center
		{1, 1, 1},
		// left & right,
		{0, 1, 1},
		{2, 1, 1},
		// front & back,
		{1, 1, 0},
		{1, 1, 2},
		// top & bottom
		{1, 0, 1},
		{1, 2, 1}
	};

	// Extract the location of the current level
	uint3 localCoords = make_uint3( regionCoords.x & 3, regionCoords.y & 3, regionCoords.z & 3 );

	// Positions of holes in the cube should be marked as constant
	// in order to never produce data.
	for ( int i = 0; i < 7; ++i )
	{
		if ( localCoords.x == emptyFaces[ i ].x &&
			localCoords.y == emptyFaces[ i ].y &&
			localCoords.z == emptyFaces[ i ].z )
		{
			return GsOracleRegionInfo::GPUVP_CONSTANT;
		}
	}

	// By default, there is data in the cube
	return GsOracleRegionInfo::GPUVP_DATA;
}
