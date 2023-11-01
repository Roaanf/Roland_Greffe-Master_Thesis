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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include "GvUtils/Noise.inl"

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/


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
template< typename NodeRes, typename BrickRes, uint BorderSize, typename VolTreeKernelType >
template< typename GPUPoolKernelType >
__device__
inline uint ToreProducerKernel< NodeRes, BrickRes, BorderSize, VolTreeKernelType >
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 0 > )
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
	const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	// Process ID gives the 1D index of a node in the current node tile
	if ( processID < NodeRes::getNumElements() )
	{
		// First, compute the 3D offset of the node in the node tile
		uint3 subOffset = NodeRes::toFloat3( processID );

		// Node production corresponds to subdivide a node tile.
		// So, based on the index of the node, find the node child.
		// As we want to sudbivide the curent node, we retrieve localization information
		// at the next level
		uint3 regionCoords = parentLocCode.addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth.addLevel().get();

		// Create a new node for which you will have to fill its information.
		GvStructure::OctreeNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		// Call what we call an oracle that will determine the type of the region of the node accordingly
		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		// Now that the type of the region is found, fill the new node information
		if ( nodeinfo == GPUVoxelProducer::GPUVP_CONSTANT )
		{
			newnode.setTerminal( true );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA_MAXRES )
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
template< typename NodeRes, typename BrickRes, uint BorderSize, typename VolTreeKernelType >
template< typename GPUPoolKernelType >
__device__
inline uint ToreProducerKernel< NodeRes, BrickRes, BorderSize, VolTreeKernelType >
::produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 1 > )
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
	const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	// Shared memory declaration
	//
	// Each threads of a block process one and unique brick of voxels.
	// We store in shared memory common useful variables that will be used by each thread.
	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	// Compute useful variables used for retrieving positions in 3D space
	brickRes = BrickRes::get();
	levelRes = make_uint3( 1 << parentLocDepth.get() ) * brickRes;
	levelResInv = make_float3( 1.0f ) / make_float3( levelRes );

	brickPos = make_int3( parentLocCode.get() * brickRes ) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	// Real brick size (with borders)
	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	

#define MY_NOISE_SWITCH 4
#if MY_NOISE_SWITCH == 0
		FractalNoise< PerlinNoise > toto( log2f(levelRes.x), 0.75f, 1.f );
#endif
#if MY_NOISE_SWITCH == 1
		WoodPatern< FractalNoise< PerlinNoise > >
				toto(FractalNoise< PerlinNoise >( log2f(levelRes.x), 0.75f, 1.f ),
					 50);
#endif
#if MY_NOISE_SWITCH == 2
		MarblePatern1< FractalNoise< PerlinNoise > >
				toto(FractalNoise< PerlinNoise >( log2f(levelRes.x), 0.95f, 0.5f )
					 );
#endif
#if MY_NOISE_SWITCH == 3
//* Magic Switch (tm)
		MarblePatern2< FractalNoise< PerlinNoise > >
				toto(FractalNoise< PerlinNoise >( log2f(levelRes.x), 0.5f, 0.5f )
					 );
/*/
		MarblePatern2< ClampedFractalNoise< PerlinNoise > >
				toto(ClampedFractalNoise< PerlinNoise >(
						 -2.0f, 2.0f,
						 log2f(levelRes.x), 0.5f, 0.5f
						 )
					 );
//*/
#endif
#if MY_NOISE_SWITCH == 4
/* Magic Switch (tm)
//		ClampedFractalNoise< 1, PerlinNoise >
//				toto(-0.5f, 0.5f,
//					 parentLocDepth.get() + 3, 0.5f, 1.0f
//					 );
		GvUtils::ClampedFractalNoise< 1, GvUtils::PerlinNoise >
				toto(0.03f, 0.1f,
					 parentLocDepth.get() + 3, 0.5f, 1.0f
					 );
/*/
		GvUtils::ClampedFractalNoise< 1, GvUtils::PerlinNoise >
				toto(-1.0f, 1.0f,
					 parentLocDepth.get() + 3, 0.75f, 1.0f
					 );
//		ClampedFractalNoise< 2, PerlinNoise >
//				toto(make_float2( -1.0f,  0.2f ),
//					 make_float2( -0.2f,  1.0f ),
//					 log2f(levelRes.x), 0.5f, 1.0f
//					 );
//*/
#endif
//	const float alpha = parentLocDepth.get() < 2 ? 1.0f : 1.0f / (1 << (parentLocDepth.get() - 1));
	const float alpha = parentLocDepth.get() < 5 ? 1.0f : 1.0f / (parentLocDepth.get() - 3);//1.f / parentLocDepth.get();//²(1 << (parentLocDepth.get() - 1));

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

	// init feedback variables
	__shared__ uint feedback;
	if ( threadIdx.z == 0 &&  threadIdx.y == 0 &&  threadIdx.x == 0 )
		feedback = 3;
	__syncthreads();
	bool threadEmpty = true;
	bool threadFull = true;

	// brick production
	uint3 elemOffset;
	for ( elemOffset.z = threadIdx.z; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z )
	{
		for ( elemOffset.y = threadIdx.y; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y )
		{
			for ( elemOffset.x = threadIdx.x; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x )
			{
                // Position of the current voxel's center (relative to the brick)
                float3 voxelPosInBrickF = ( make_float3( elemOffset ) + 0.5f ) * levelResInv;
                // Position of the current voxel's center (absolute, in [0.0;1.0] range)
                float3 voxelPosF = brickPosF + voxelPosInBrickF;
                // Position of the current voxel's center (scaled to the range [-1.0;1.0])
                float3 posF = voxelPosF * 2.0f - 1.0f;

				// color computation
				float noise1 = __saturatef(fabsf(       toto( posF ) ));
				float noise2 = __saturatef(fabsf( 2.5f * toto( posF ) ));
				float noise3 = __saturatef(fabsf( 1.5f * toto( posF ) ));
				float4 voxelColor = make_float4(noise1,
                                                noise2,//fabs(toto(posF.z, posF.x, posF.y)),
                                                noise3,//fabs(toto(posF.y, posF.z, posF.x)),
                                                0.00f);

				// normal computation
				float2 xyNorm = make_float2( posF.x, posF.y );
				xyNorm = xyNorm - _R * normalize(xyNorm);
                float4 voxelNormal = normalize(make_float4( xyNorm.x, xyNorm.y, posF.z, 0.0f ));
				voxelNormal.w = 1.0f;

                // Test if the voxel is located inside the unit Tore
                if ( isInTore( posF ) )//&& parentLocDepth.get() < 6 )
                {
					voxelColor.w = 1.0f;
                }

                // Alpha pre-multiplication used to avoid the "color bleeding" effect
                voxelColor.x *= voxelColor.w;
                voxelColor.y *= voxelColor.w;
                voxelColor.z *= voxelColor.w;

                // Compute the new element's address
                uint3 destAddress = newElemAddress + elemOffset;
                // Write the voxel's color in the first field
                dataPool.template setValue< 0 >( destAddress, voxelColor );
                // Write the voxel's normal in the second field
                dataPool.template setValue< 1 >( destAddress, voxelNormal );
			}
		}
	}

	return 0;
}



/******************************************************************************
 * Helper class to test if a point is inside the Tore
 *
 * @param pPoint the point to test
 *
 * @return flag to tell wheter or not the point is insied the Tore
 ******************************************************************************/
template< typename NodeRes, typename BrickRes, uint BorderSize, typename VolTreeKernelType >
__device__
bool ToreProducerKernel< NodeRes, BrickRes, BorderSize, VolTreeKernelType >
::isInTore( float3 pPoint ) const
{
//	return length( pPoint ) < 1.f ;
//  (R - sqrt(x2+y2) )2 + z2 = r2
//	(x2 + y2 + z2 + R2 -r2)2 = 4R2 ( x2 + y2)
	float x = pPoint.x, y = pPoint.y, z = pPoint.z;
	float tmp = (_R - sqrtf(x*x + y*y));
	return (tmp * tmp + z*z) < (_r * _r);
}


/******************************************************************************
 * Helper function used to determine the type of zones in the data structure.
 *
 * The data structure is made of regions containing data, empty or constant regions.
 * Besides, this function can tell if the maximum resolution is reached in a region.
 *
 * @param regionCoords region coordinates
 * @param regionDepth region deptj
 *
 * @return the type of the region
 ******************************************************************************/
template< typename NodeRes, typename BrickRes, uint BorderSize, typename VolTreeKernelType >
__device__
inline GPUVoxelProducer::GPUVPRegionInfo ToreProducerKernel< NodeRes, BrickRes, BorderSize, VolTreeKernelType >
::getRegionInfo( uint3 regionCoords, uint regionDepth )
{
	// Limit the depth.
	// Currently, 32 is the max depth of the GigaVoxels engine.
	if ( regionDepth >= 32 )
	{
		return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	}

	// Produce data in the first levels which, indeed, contains data even if
	// all the corners are outside of the tore
	if (regionDepth < 2)
	{
		return GPUVoxelProducer::GPUVP_DATA;
	}

	// Shared memory declaration
	__shared__ uint3 brickRes;
	__shared__ float3 brickSize;
	__shared__ uint3 levelRes;
	__shared__ float3 levelResInv;

	brickRes = BrickRes::get();

	levelRes = make_uint3( 1 << regionDepth ) * brickRes;
	levelResInv = make_float3( 1.f ) / make_float3( levelRes );

	int3 brickPos = make_int3(regionCoords * brickRes) - BorderSize;
	float3 brickPosF = make_float3( brickPos ) * levelResInv;

	// Since we work in the range [-1;1] below, the brick size is two time bigger
	brickSize = make_float3( 1.f ) / make_float3( 1 << regionDepth ) * 2.f;

	// Build the eight brick corners of a Tore centered in [0;0;0]
	float3 q000 = make_float3( regionCoords * brickRes ) * levelResInv * 2.f - 1.f;
	float3 q001 = make_float3( q000.x + brickSize.x, q000.y,			   q000.z);
	float3 q010 = make_float3( q000.x,				 q000.y + brickSize.y, q000.z);
	float3 q011 = make_float3( q000.x + brickSize.x, q000.y + brickSize.y, q000.z);
	float3 q100 = make_float3( q000.x,				 q000.y,			   q000.z + brickSize.z);
	float3 q101 = make_float3( q000.x + brickSize.x, q000.y,			   q000.z + brickSize.z);
	float3 q110 = make_float3( q000.x,				 q000.y + brickSize.y, q000.z + brickSize.z);
	float3 q111 = make_float3( q000.x + brickSize.x, q000.y + brickSize.y, q000.z + brickSize.z);

	// Test if any of the eight brick corner lies in the Tore
	if ( isInTore( q000 ) || isInTore( q001 ) || isInTore( q010 ) || isInTore( q011 ) ||
		isInTore( q100 ) || isInTore( q101 ) || isInTore( q110 ) || isInTore( q111 ) )
	{
		return GPUVoxelProducer::GPUVP_DATA;
	}

	return GPUVoxelProducer::GPUVP_CONSTANT;
}
