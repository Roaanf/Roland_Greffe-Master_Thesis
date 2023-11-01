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
 * Transfer function texture
 */
texture< float4, cudaTextureType1D, cudaReadModeElementType > transferFunctionTexture;

/**
 * Volume texture (3D normal + signed distance field)
 */
texture< float4, cudaTextureType3D, cudaReadModeElementType > volumeTex;

/******************************************************************************
 * Map a distance to a color (with a transfer function)
 *
 * @param dist the distance to map
 *
 * @return The color corresponding to the distance
 ******************************************************************************/
__device__
inline float4 distToColor( float dist )
{
	return tex1D( transferFunctionTexture, dist );
}

/******************************************************************************
 * Get the RGBA data of distance field + noise
 *
 * @param p 3D position
 *
 * @return ...
 ******************************************************************************/
__device__ const float noise_first_frequency = 32.0f;
__device__ const float noise_strength = 1.0f;

__device__
float4 getRGBA( float3 voxelPosF, uint3 levelRes)
{
	// Type definition for the noise
	typedef GvUtils::GsNoiseKernel Noise;

	float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );
	float3 voxelNormal = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );

	// compute noise
	float dist_noise = 0.0f;
	for ( float frequency = noise_first_frequency; frequency < levelRes.x ; frequency *= 2.f )
	{
		dist_noise += noise_strength / frequency * Noise::getValue( frequency * voxelPosF );
	}

	// compute color
	float4 voxelRGBA = distToColor( clamp( 0.5f - 0.5f * ( voxelNormalAndDist.w + dist_noise ) * noise_first_frequency, 0.f, 1.f ) );

	return voxelRGBA;
}

/******************************************************************************
 * Get the normal of distance field + noise
 *
 * @param p 3D position
 *
 * @return ...
 ******************************************************************************/
__device__
float3 getNormal(float3 voxelPosF, uint3 levelRes)
{
	// Type definition for the noise
	typedef GvUtils::GsNoiseKernel Noise;

	float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );
	float3 voxelNormal = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );

	float eps = 0.5f / (float) levelRes.x;

	// compute noise
	float3 grad_noise = make_float3(0.0f);
	for ( float frequency = noise_first_frequency; frequency < levelRes.x ; frequency *= 2.f )
	{
		grad_noise.x +=  noise_strength / frequency * Noise::getValue( frequency * ( voxelPosF + make_float3( eps, 0.0f, 0.0f ) ) )
						-noise_strength / frequency * Noise::getValue( frequency * ( voxelPosF - make_float3( eps, 0.0f, 0.0f ) ) );
		grad_noise.y +=  noise_strength / frequency * Noise::getValue( frequency * ( voxelPosF + make_float3( 0.0f, eps, 0.0f ) ) )
						-noise_strength / frequency * Noise::getValue( frequency * ( voxelPosF - make_float3( 0.0f, eps, 0.0f ) ) );
		grad_noise.z +=  noise_strength / frequency * Noise::getValue( frequency * ( voxelPosF + make_float3( 0.0f, 0.0f, eps ) ) )
						-noise_strength / frequency * Noise::getValue( frequency * ( voxelPosF - make_float3( 0.0f, 0.0f, eps ) ) );
	}

	grad_noise *= 0.5f / eps;

	voxelNormal = normalize( voxelNormal + grad_noise - dot( grad_noise, voxelNormal ) * voxelNormal );
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
	const GvCore::GsLocalizationInfo::CodeType *parentLocCode = &parentLocInfo.locCode;
	const GvCore::GsLocalizationInfo::DepthType *parentLocDepth = &parentLocInfo.locDepth;

	if ( processID < NodeRes::getNumElements() )
	{
		uint3 subOffset = NodeRes::toFloat3( processID );

		uint3 regionCoords = parentLocCode->addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth->addLevel().get();

		GvStructure::GsNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		GsOracleRegionInfo::OracleRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

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

		// Write node info into the node pool
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
	const GvCore::GsLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GsLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	//printf( "\nDepth = %d", parentLocDepth.get() );

	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3( 1 << parentLocDepth.get() ) * brickRes;
	levelResInv = make_float3( 1.f ) / make_float3( levelRes );

	brickPos = make_int3( parentLocCode.get() * brickRes) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	uint3 elemOffset;

	for ( elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x )
			{
				uint3 locOffset = elemOffset + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z );

				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					float3 voxelPosF = brickPosF + voxelPosInBrickF;

					// Compute data
					float4 voxelColor = getRGBA( voxelPosF, levelRes );
					float3 voxelNormal = getNormal( voxelPosF, levelRes );					

					// Compute the new element's address
					uint3 destAddress = newElemAddress + locOffset;
					// Write the voxel's color in the first field
					dataPool.template setValue< 0 >( destAddress, voxelColor );
					// Write the voxel's normal in the second field
					dataPool.template setValue< 1 >( destAddress, make_float4( voxelNormal, 0.f ) );
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
 * @param regionDepth region deptj
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

	// Limit the depth
	if ( regionDepth >= 32 )
	{
		return GsOracleRegionInfo::GPUVP_DATA_MAXRES;
	}

	//const GvCore::GsLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	//const GvCore::GsLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

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

	//float brickSize = 1.0f / (float)( 1 << regionDepth );

	for ( elemOffset.z = 0; elemOffset.z < elemSize.z && isEmpty; elemOffset.z++ )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y  && isEmpty; elemOffset.y++ )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x && isEmpty; elemOffset.x++ )
			{
				uint3 locOffset = elemOffset;// + make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					float3 voxelPosF = brickPosF + voxelPosInBrickF;

					float4 voxelColor = getRGBA( voxelPosF, levelRes );

					if ( voxelColor.w > 0.0f )
					{
						isEmpty = false;
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
