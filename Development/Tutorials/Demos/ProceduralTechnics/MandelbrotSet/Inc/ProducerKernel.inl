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
#include <GvRendering/GsNodeVisitorKernel.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * HELPER function
 *
 * @param depth ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline uint getNumIter( uint depth )
{
	return static_cast< uint >( ceilf( ( log2f( static_cast< float >( depth ) ) + 1.0f ) * 1.5f ) );
}

/******************************************************************************
 * Distance estimation
 *
 * @param z ...
 * @param grad ...
 * @param maxIter ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline float compMandelbrotNewOptim( float3 z, float3& grad, int pMaxIter )
{
	// Power of fractal
	// - note : p > 0.0f
	const float cPower = 8.0f;

	const float threshold = 4.0f;
	const float m_min_dst = 9999.0f;
	const int m_max_iterations = pMaxIter;
	
	// Constant term in iterative formula
	const float3 c = z;
	
	// Divergence threshold nearly infinite for negative powers
	const float d_threshold = min( 2.0f, __powf( threshold, 1.0f / cPower ) );

	// orbit trapping continues at existing m_min_dist
	float min_dst = m_min_dst;

	// point z polar coordinates
	float rho = length( z );
	float theta = atan2f( z.y, z.x );
	float phi = asinf( z.z / rho );

	// orbit trapping relative to point (0,0,0)
	if ( rho < min_dst )
	{
		min_dst = rho;
	}

	float rho_dz = 1.0f;
	//float phi_dz = 0.0f;
	//float theta_dz = 0.0f;
	
	// Iterate to compute the distance estimator.
	int i = m_max_iterations;
	while ( i-- )
	{
		// Note : cPower > 0.0f

		// purely scalar dz iteration (Thanks to Enforcer for the great tip)
		float zr = __powf( rho, cPower - 1 );
		rho_dz = zr * rho_dz * cPower + 1;

		// z iteration
		float P_ = zr * rho; // equivalent to __powf( rho, cPower );
		float s1, c1; __sincosf( cPower * phi, &s1, &c1 );
		float s2, c2; __sincosf( cPower * theta, &s2, &c2 );
		z.x = P_ * c1 * c2 + c.x;
		z.y = P_ * c1 * s2 + c.y;
		z.z = P_ * s1 + c.z;

		// compute new length of z
		rho = length( z );

		// results are not stored for the "extra" iteration at i == 0
		// orbit trapping relative to point (0,0,0)
		if ( rho < min_dst )
		{
			min_dst = rho;
		}

		// Stop when we know the point diverges and return the result.
		if( rho > d_threshold )
		{
			break;
		}

		// compute remaining polar coordinates of z and iterate further
		theta = atan2f( z.y, z.x );
		phi = asinf( z.z / rho );
	}

	// return the result if we reached convergence
	///m_min_dst = min_dst;
	grad.x = min_dst;

	return /*0.5f **/ rho * __logf( rho ) / rho_dz;
}

/******************************************************************************
 * ...
 *
 * @param x ...
 * @param grad ...
 * @param maxIter ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline float compMandelbrot( float3 x, float3& grad, int maxIter )
{
	return compMandelbrotNewOptim( x, grad, maxIter );
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
	_dataStructureKernel = pDataStructure;
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
	// typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;
	//
	// In this tutorial, we have choosen one channel containing normal and opacity at channel 0.

	// Retrieve current brick localization information code and depth
	uint3 parentLocCode = parentLocInfo.locCode.get();
	uint parentLocDepth = parentLocInfo.locDepth.get();

	// Shared memory declaration
	//
	// Each threads of a block process one and unique brick of voxels.
	// We store in shared memory common useful variables that will be used by each thread.
	__shared__ uint level;
	__shared__ uint3 blockRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;
	
	__shared__ uint maxIter;
	__shared__ uint maxIterGrad;

	// Compute useful variables used for retrieving positions in 3D space
	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );

	//level=(uint)parentLocDepth+1;
	level = parentLocDepth;
	blockRes = BrickRes::get();
	levelRes = make_uint3( 1 << parentLocDepth ) * blockRes;
	levelResInv = make_float3( 1.0f ) / make_float3( levelRes );
	brickPos = make_int3( parentLocCode * blockRes ) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	maxIter = getNumIter( level ) * evalIterCoef;
	maxIterGrad = getNumIter( level ) * gradIterCoef;

	// Shared Memory declaration
	//
	// - optimization to be able to modify the "content" of the node
	//   => if it is "empty", it returns 2 to modify node "state"
	__shared__ bool nonNull;
	if ( threadIdx.x == 0 && threadIdx.y == 0 )
	{
		nonNull = false;
	}
	// Thread Synchronization
	__syncthreads();

	// Shared Memory declaration
	//
	//__shared__ bool b0_OK;
	//__shared__ float b0_nodeSize;
	//__shared__ float3 b0_nodePos;
	//__shared__ float3 b0_brickPosInPool;
	//__shared__ float b0_brickScaleInPool;

	float3 samplePos = brickPosF + ( make_float3( elemSize ) * 0.5f + 1.0f ) * levelResInv;

	GvStructure::GsNode pnode;
	float pnodeSize = 1.0f;
	float3 pnodePos = make_float3( 0.0f );
	uint pnodeDepth = 0;
	uint pbrickAddressEnc = 0;
	float3 pbrickPos = make_float3( 0.0f );
	float pbrickScale = 1.0f;

	// TO DO
	//
	// - this method seems to make the program crash in Debug mode
	GvRendering::GsNodeVisitorKernel::visit( _dataStructureKernel, parentLocDepth - 3, samplePos, _dataStructureKernel._rootAddress,
											 pnode, pnodeSize, pnodePos, pnodeDepth, pbrickAddressEnc, pbrickPos, pbrickScale );

	//b0_OK = pbrickAddressEnc;
	//b0_nodeSize = pnodeSize;
	//b0_nodePos = pnodePos; 

	//b0_brickScaleInPool = pbrickScale;

	//b0_brickPosInPool = make_float3( GvStructure::GsNode::unpackBrickAddress( pbrickAddressEnc ) ) *
		_dataStructureKernel.brickCacheResINV + pbrickPos * _dataStructureKernel.brickSizeInCacheNormalized.x;

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
	int3 decal;
	for ( decal.z = 0; decal.z< (int)(elemSize.z); decal.z += blockDim.z )
	{
		for ( decal.y = 0; decal.y< (int)(elemSize.y); decal.y += blockDim.y )
		{
			for ( decal.x = 0; decal.x< (int)(elemSize.x); decal.x += blockDim.x )
			{
				int3 locDecal = decal + make_int3( threadIdx.x, threadIdx.y, threadIdx.z );

				if ( locDecal.x < (int)(elemSize.x) && locDecal.y < (int)(elemSize.y) && locDecal.z < (int)(elemSize.z) )
				{
					//int3 voxelPos= brickPos + locDecal ;
					//float3 voxelPosF=(make_float3(voxelPos)+0.5f)*levelResInv;

					float3 voxelPosInBrickF = ( make_float3( locDecal ) + 0.5f ) * levelResInv;
					float3 voxelPosF = brickPosF + voxelPosInBrickF;

					// Transform coordinates from [ 0.0, 1.0 ] to [ -1.0, 1.0 ]
					float3 pos = voxelPosF * 2.0f - 1.0f;

					// Estimate distance from point to fractal (shortest distance)
					float3 dz = make_float3( 0.0f, 0.0f, 0.0f );
					float dist = compMandelbrot( pos, dz, maxIter );

					float val = dist * 0.5f;

					// Compute derivative (i.e. gradient)
					float step = levelResInv.x; // vodel size
					float3 ndz;
					float3 grad;
					grad.x = compMandelbrot( pos + make_float3( +step, 0.0f, 0.0f ), ndz, maxIterGrad )
						   - compMandelbrot( pos + make_float3( -step, 0.0f, 0.0f ), ndz, maxIterGrad );
					grad.y = compMandelbrot( pos + make_float3( 0.0f, +step, 0.0f ), ndz, maxIterGrad )
						   - compMandelbrot( pos + make_float3( 0.0f, -step, 0.0f ), ndz, maxIterGrad );
					grad.z = compMandelbrot( pos + make_float3( 0.0f, 0.0f, +step ), ndz, maxIterGrad )
						   - compMandelbrot( pos + make_float3( 0.0f, 0.0f, -step ), ndz, maxIterGrad );

					float vis = 1.0f;

					// FIXME: Broken, need to figure out why.
					//if ( b0_OK )
					//{
					//	float3 voxelPosInNode = voxelPosF - b0_nodePos;

					//	//float dpscale;
					//	//float3 voxelPosInBrick = _dataStructureKernel.getPosInBrick( b0_nodeSize, voxelPosInNode, dpscale );
					//	float3 voxelPosInBrick = voxelPosInNode * _dataStructureKernel.brickSizeInCacheNormalized.x / b0_nodeSize;
					//	voxelPosInBrick = voxelPosInBrick * b0_brickScaleInPool;

					//	float4 pVox = _dataStructureKernel.template getSampleValueTriLinear< 0 >( b0_brickPosInPool, voxelPosInBrick );

					//	float d = pVox.w / float( distanceMultiplier );
					
					//	float curD = val;

					//	vis = d / curD;
					//	vis = vis * vis;

					//	vis = clamp( vis, 0.0f, 1.0f );
					//}	

					grad = normalize( grad ) * vis;
					grad = ( grad + 1.0f ) * 0.5f;

					val = val * float( distanceMultiplier );
					if ( val < 1.0f )
					{
						nonNull = true;
					}

					// set opacity in [ 0.0, 1.0 ]
					val = max( min( val, 1.0f ), 0.0f );

					// Voxel data
					//
					// - normal and opacity
					float4 finalValue;
					finalValue.x = grad.x;
					finalValue.y = grad.y;
					finalValue.z = grad.z;
					finalValue.w = val;

					// Compute the new element's address in cache (i.e. data pool)
					uint3 destAddress = newElemAddress + make_uint3( locDecal );
					// Write the voxel's data in the first field
					dataPool.template setValue< 0 >( destAddress, finalValue );
				}
			}
		}
	}

	// Thread Synchronization
	__syncthreads();
	// Optimization to be able to modify the "content" of the node
	//   => if it is "empty", it returns 2 to modify node "state"
	if ( ! nonNull )
	{
		return 2;
	}
	
	// Default value
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
::getRegionInfo( uint3 pRegionCoords, uint pRegionDepth )
{
	// GigaVoxels work in Model space,
	// i.e. a BBox lying in [ 0.0, 1.0 ] x [ 0.0, 1.0 ] x [ 0.0, 1.0 ]

	// Number of nodes in each dimension (at current depth)
	float3 levelRes = make_float3( static_cast< float >( 1 << pRegionDepth ) );
	// Size of node
	float3 nodeSize = make_float3( 1.0f ) / levelRes;
	// Bottom left corner of node
	float3 nodePos = make_float3( pRegionCoords ) * nodeSize;
	// Node center
	float3 nodeCenter = nodePos + nodeSize * 0.5f;

	// Transform coordinates from [ 0.0, 0.0 ] to [ -1.0, 1.0 ]
	// - node center
	float3 nodeCenterMandel = ( nodeCenter * 2.0f - 1.0f );

	// Derivative
	float3 dz = make_float3( 1.0f, 0.0f, 0.0f );

	float distMandel = 1000.0f;

	uint maxIter = getNumIter( pRegionDepth ) * regionInfoIterCoef;

	float3 offset;
	for ( offset.z =- 1.0f; offset.z <= 1.0f; offset.z += 1.0f )
	{
		for ( offset.y =- 1.0f; offset.y <= 1.0f; offset.y += 1.0f )
		{
			for ( offset.x =- 1.0f; offset.x <= 1.0f; offset.x += 1.0f )
			{
				// Compute distance estimation
				float distMandel0 = compMandelbrot( nodeCenterMandel + offset * nodeSize, dz, maxIter );

				distMandel = min( distMandel, distMandel0 );
			}
		}
	}

	// Check criteria
	if ( distMandel <= 0.0f )
	{
		if ( pRegionDepth == GvCore::GsLocalizationInfo::maxDepth )
		{
			// Region with max level of detail
			return GsOracleRegionInfo::GPUVP_DATA_MAXRES;
		}
		else
		{
			// Region with data inside
			return GsOracleRegionInfo::GPUVP_DATA; 
		}
	}
	else
	{
		// Empty region
		return GsOracleRegionInfo::GPUVP_CONSTANT;
	}
}
