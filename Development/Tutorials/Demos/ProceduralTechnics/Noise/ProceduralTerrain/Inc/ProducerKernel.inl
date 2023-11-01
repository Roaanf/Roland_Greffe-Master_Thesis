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

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
float getDensity( const float3 posInWorld )
{
	// Type definition for the noise
	typedef GvUtils::GsNoiseKernel Noise;

	//posInWorld.x += 4.f;

	//const int PI = 3.141592f;
	//return cosf(posInWorld.x * PI);
	float density = -posInWorld.y;

	density -= 0.5f;
	density += 0.5000f * Noise::getValue( 1.f * posInWorld.x, 1.f * posInWorld.y, 1.f * posInWorld.z );
	density += 0.2500f * Noise::getValue( 2.f * posInWorld.x, 2.f * posInWorld.y, 2.f * posInWorld.z );
	density += 0.1250f * Noise::getValue( 4.f * posInWorld.x, 4.f * posInWorld.y, 4.f * posInWorld.z );
	density += 0.0625f * Noise::getValue( 8.f * posInWorld.x, 8.f * posInWorld.y, 8.f * posInWorld.z );

	return density;
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
	//parentLocDepth++;

	if ( processID < NodeRes::getNumElements() )
	{
		uint3 subOffset;
		subOffset.x = processID & NodeRes::xLog2;
		subOffset.y = (processID >> NodeRes::xLog2) & NodeRes::yLog2;
		subOffset.z = (processID >> (NodeRes::xLog2 + NodeRes::yLog2)) & NodeRes::zLog2;

		uint3 regionCoords = parentLocCode->addLevel<NodeRes>(subOffset).get();
		uint regionDepth = parentLocDepth->addLevel().get();

		GvStructure::GsNode newnode;
		newnode.childAddress=0;

		GsOracleRegionInfo::OracleRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		if ( nodeinfo == GsOracleRegionInfo::GPUVP_CONSTANT )
		{
		//	newnode.data.setValue(0.0f);
		//	newnode.setStoreValue();
			newnode.setTerminal(true);
		}
		else if ( nodeinfo == GsOracleRegionInfo::GPUVP_DATA )
		{
		//	newnode.data.brickAddress = 0;
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GsOracleRegionInfo::GPUVP_DATA_MAXRES )
		{
		//	newnode.data.brickAddress = 0;
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}

		// Write node info into the node pool
		nodePool.getChannel(Loki::Int2Type<0>()).set(newElemAddress.x + processID, newnode.childAddress);
		//nodePool.getChannel(Loki::Int2Type<1>()).set(newElemAddress.x + processID, newnode.data.brickAddress);
		nodePool.getChannel(Loki::Int2Type<1>()).set(newElemAddress.x + processID, newnode.brickAddress);
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
	// typedef Loki::TL::MakeTypelist< half4 >::Result DataType;
	//
	// In this tutorial, we have choosen one channel containing color at channel 0.

	// Retrieve current brick localization information code and depth
	const GvCore::GsLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GsLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	//parentLocDepth++;

	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3(1 << parentLocDepth.get()) * brickRes;
	levelResInv = make_float3(1.0f) / make_float3(levelRes);

	brickPos = make_int3(parentLocCode.get() * brickRes) - BorderSize;
	brickPosF = make_float3(brickPos) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3(2 * BorderSize);
	uint3 elemOffset;

	for (elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z)
	for (elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y)
	for (elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x)
	{
		uint3 locOffset = elemOffset + make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

		if (locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z)
		{
			// Position of the current voxel's center (relative to the brick)
			float3 voxelPosInBrickF = (make_float3(locOffset) + 0.5f) * levelResInv;
			// Position of the current voxel's center (absolute, in [0.0;1.0] range)
			float3 voxelPosF = brickPosF + voxelPosInBrickF;
			// Position of the current voxel's center (scaled to the range [-1.0;1.0])
			float3 posF = voxelPosF * 2.0f - 1.0f;

			// the final voxel's data
			float4 data;

			// w component hold the density
			data.w = getDensity(posF);

			// xyz components holds the gradient
			float step = 1.0f / levelRes.x;

			float3 grad;
			grad.x = getDensity(posF + make_float3(step, 0.0f, 0.0f)) - getDensity(posF - make_float3(step, 0.0f, 0.0f));
			grad.y = getDensity(posF + make_float3(0.0f, step, 0.0f)) - getDensity(posF - make_float3(0.0f, step, 0.0f));
			grad.z = getDensity(posF + make_float3(0.0f, 0.0f, step)) - getDensity(posF - make_float3(0.0f, 0.0f, step));
			grad = normalize(-grad);

			// compute the new element's address
			uint3 destAddress = newElemAddress + locOffset;
			// write the voxel's data in the first field
			dataPool.template setValue< 0 >( destAddress, data );
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
	//__shared__ float3 levelRes;
	//__shared__ float3 nodeSize;

	//if (regionDepth >= 31)
	//	return GsOracleRegionInfo::GPUVP_DATA_MAXRES;
	////else
	////	return GsOracleRegionInfo::GPUVP_DATA;

	//levelRes = make_float3(1 << regionDepth);
	//nodeSize = make_float3(1.0f) / levelRes;

	//float3 nodePosInLocal = make_float3(regionCoords) * nodeSize;
	//float3 nodeCenterInLocal = nodePosInLocal + nodeSize / 2.0f;
	//float3 nodeCenterInWorld = nodeCenterInLocal * 2.0f - 1.0f;

	//float maxDensity = -1.f;
	//float3 offset;

	//// get the minimal distance
	//for (offset.z = -1.0f; offset.z <= 1.0f; offset.z += 1.0f)
	//for (offset.y = -1.0f; offset.y <= 1.0f; offset.y += 1.0f)
	//for (offset.x = -1.0f; offset.x <= 1.0f; offset.x += 1.0f)
	//{
	//	maxDensity = max(maxDensity, getDensity(nodeCenterInWorld + offset * nodeSize));
	//}

	////if (maxDensity > 0.0f)
	//return GsOracleRegionInfo::GPUVP_DATA;
	////else
	////	return GsOracleRegionInfo::GPUVP_CONSTANT;
	//parentLocDepth++;

	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3(1 << regionDepth) * brickRes;
	levelResInv = make_float3(1.0f) / make_float3(levelRes);

	brickPos = make_int3(regionCoords * brickRes) - BorderSize;
	brickPosF = make_float3(brickPos) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3(2 * BorderSize);
	uint3 elemOffset;

	float maxDensity = 0.f;

	for (elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z)
	for (elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y)
	for (elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x)
	{
		uint3 locOffset = elemOffset + make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

		if (locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z)
		{
			// Position of the current voxel's center (relative to the brick)
			float3 voxelPosInBrickF = (make_float3(locOffset) + 0.5f) * levelResInv;
			// Position of the current voxel's center (absolute, in [0.0;1.0] range)
			float3 voxelPosF = brickPosF + voxelPosInBrickF;
			// Position of the current voxel's center (scaled to the range [-1.0;1.0])
			float3 posF = voxelPosF * 2.0f - 1.0f;

			// the final voxel's data
			maxDensity = max(maxDensity, getDensity(posF));
		}
	}

	if ( maxDensity > 0.f )
	{
		return GsOracleRegionInfo::GPUVP_DATA;
	}
	else
	{
		return GsOracleRegionInfo::GPUVP_CONSTANT;
	}
}
