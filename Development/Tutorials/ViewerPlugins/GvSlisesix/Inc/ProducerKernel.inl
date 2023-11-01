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
//#include <GvStructure/GsVolumeTree.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * ...
 ******************************************************************************/
__device__ __forceinline__
int icoolfFunc3d2( int n )
{
	n = ( n << 13 )^n;
	return ( n * ( n * n * 15731 + 789221 ) + 1376312589 ) & 0x7fffffff;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__ __forceinline__
float coolfFunc3d2( int n )
{
	return static_cast< float >( icoolfFunc3d2( n ) );
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__ __forceinline__
float noise3f( float3 p )
{
	int3 ip = make_int3(floorf(p));
	float3 u = fracf(p);
	u = u * u * (3.0f - 2.0f * u);

	int n = ip.x + ip.y * 57 + ip.z * 113;

	float res = lerp(lerp(lerp(coolfFunc3d2(n+(0+57*0+113*0)),
		coolfFunc3d2(n+(1+57*0+113*0)),u.x),
		lerp(coolfFunc3d2(n+(0+57*1+113*0)),
		coolfFunc3d2(n+(1+57*1+113*0)),u.x),u.y),
		lerp(lerp(coolfFunc3d2(n+(0+57*0+113*1)),
		coolfFunc3d2(n+(1+57*0+113*1)),u.x),
		lerp(coolfFunc3d2(n+(0+57*1+113*1)),
		coolfFunc3d2(n+(1+57*1+113*1)),u.x),u.y),u.z);

	return 1.0f - res * ( 1.0f / 1073741824.0f );
}

/******************************************************************************
 * Sum of Perlin noise functions
 ******************************************************************************/
__device__ __forceinline__
float fbm( float3 p )
{
	return 0.5000f * noise3f( p * 1.0f ) + 0.2500f * noise3f( p * 2.0f ) + 0.1250f * noise3f( p * 4.0f ) + 0.0625f * noise3f( p * 8.0f );
}

/******************************************************************************
 * Ceiling
 ******************************************************************************/
__device__ __forceinline__
float techo( float x, float y )
{
	y = 1.0f - y;

	if ( x < 0.1f || x > 0.9f )
	{
		return y;
	}

	x = x - 0.5f;

	return -( sqrtf( x * x + y * y ) - 0.4f );
}

/******************************************************************************
 * Distance function for basic primitive
 ******************************************************************************/
__device__ __forceinline__
float distToBox( float3 p, float3 abc )
{
	const float3 di = fmaxf( fabs( p ) - abc, make_float3( 0.0f ) );

	return dot( di, di );
}

/******************************************************************************
 * Compute distance to some proceduraly generated columns by blending distance fields :
 *
 * @param x position
 * @param y position
 * @param z position
 * @param mindist ...
 * @param offx ... (unused)
 *
 * @return distance to columns
 ******************************************************************************/
__device__ __forceinline__
float columna( float x, float y, float z, float mindist, float offx/*unused*/ )
{
	// Current point
	float3 p = make_float3( x, y, z );

	// Add domain distorsion
	// - twisted column : rotate around Y axis
	/*if ( cUseTwistedColumns )
	{
		const float angle = p.y * 1.7f;
		const float rc = cosf( angle );
		const float rs = sinf( angle );
		p = make_float3( p.x * rc - p.z * rs, p.y, p.x * rs + p.z * rc );
	}*/

	// ...
	const float di0 = distToBox( p, make_float3( 0.14f, 1.0f, 0.14f ) );
	if ( di0 > ( mindist * mindist ) )
	{
		return mindist + 1.0f;
	}

	// Create useful variables to build columns details
	const float y2 = y - 0.40f;
	const float y3 = y - 0.35f;
	const float y4 = y - 1.00f;

	// [ zone 1 ] - main column
	const float di1 = distToBox( p, make_float3( 0.10f, 1.00f, 0.10f ) );

	// [ zone 2 ] - add large base around zone 1 (i.e. main column)
	const float di2 = distToBox( p, make_float3( 0.12f, 0.40f, 0.12f ) );

	// [ zone 3 ] - add rectangle areas on each Z axis sides of zone 2 (i.e. base)
	const float di3 = distToBox( p, make_float3( 0.05f, 0.35f, 0.14f ) );
	// [ zone 4 ] - add rectangle areas on each X axis sides of zone 2 (i.e. base)
	const float di4 = distToBox( p, make_float3( 0.14f, 0.35f, 0.05f ) );
	
	// [ zone 9 ] - add rectangle areas on top of column (i.e. head of column)
	float di9 = distToBox( make_float3( x, y4, z ), make_float3( 0.14f, 0.02f, 0.14f ) );

	// [ zone 5 ] - add triangle areas on top of each Z axis sides of zone 2 (i.e. base)
	const float di5 = distToBox( make_float3( ( x - y2 ) * 0.7071f, ( y2 + x ) * 0.7071f, z ), make_float3( 0.10f * 0.7071f, 0.10f * 0.7071f, 0.12f ) );
	// [ zone 6 ] - add triangle areas on top of each X axis sides of zone 2 (i.e. base)
	const float di6 = distToBox( make_float3( x, ( y2 + z ) * 0.7071f, ( z - y2 ) * 0.7071f ), make_float3( 0.12f,  0.10f * 0.7071f, 0.10f * 0.7071f ) );
	
	// [ zone 7 ] - add square areas on top of each Z axis sides of zone 3 (i.e. rectangular areas on base)
	const float di7 = distToBox( make_float3( (x - y3 ) * 0.7071f, ( y3 + x ) * 0.7071f, z ), make_float3( 0.10f * 0.7071f,  0.10f * 0.7071f, 0.14f ) );
	// [ zone 8 ] - add square areas on top of each X axis sides of zone 3 (i.e. rectangular areas on base)
	const float di8 = distToBox( make_float3( x, ( y3 + z ) * 0.7071f, ( z - y3 ) * 0.7071f ), make_float3( 0.14f,  0.10f * 0.7071f, 0.10f * 0.7071f ) );

	// Combination of (instanced) distance fields can be done by taking the min of the distance fields involved
	float di = min( min( min( di1, di2 ), min( di3, di4 ) ), min( min( di5, di6 ), min( di7, di8 ) ) );
	di = min( di, di9 );

	//  di += 0.00000003 * max( fbm( 10.1 * p ), 0.0 );

	return di;
}

/******************************************************************************
 * Compute distance to a proceduraly generated monster by blending distance fields :
 * - body is a ball
 * - tentacles are tubes
 * Add also domain distorsion :
 * - elevation tentacles (near center)
 * - noise on rotation of tentacles
 *
 * @param x position
 * @param pMinDistance ... (unused)
 *
 * @return distance to monster
 ******************************************************************************/
__device__ __forceinline__
float distanceToMonster( float3 x, float pMinDistance/*unused*/ )
{
	//float ramo = noise3f( vec3( 2.0  *time, 2.3 * time, 0.0 ) );

	// Add global translation
	x -= make_float3( 0.64f, 0.5f, 1.5f );

	// Add global scale
	float r2 = dot( x, x );
	{
		float sa = smoothstep( 0.0f, 0.5f, r2 );
		float fax = 0.75f + 0.25f * sa;
		float fay = 0.80f + 0.20f * sa;
		x.x *= fax;
		x.y *= fay;
		x.z *= fax;
	}
	r2 = dot( x, x );

	// Compute length of position vector
	float r = sqrtf( r2 );

	// Add global rotation
	{
		float a1 = 1.0f - smoothstep( 0.0f, 0.75f, r );
		a1 *= 0.40f;
		float si1 = sinf( a1 );
		float co1 = cosf( a1 );
		//x.xy = mat2( co1, si1, -si1, co1 ) * x.xy;
		float2 xy;
		xy.x = co1 * x.x - si1 * x.y;
		xy.y = si1 * x.x + co1 * x.y;
		x.x = xy.x;
		x.y = xy.y;
	}

	// Generate tentacles
	//
	// - number : 6
	// - (x-z) plan is the floor
	// - rotation around y axis
	float distance = 100000.0f;
	//float rr = dot( make_float2( x.x, x.y ), make_float2( x.x, x.y ) );
	float rr = 0.05f + sqrt( dot( make_float2( x.x, x.z ), make_float2( x.x, x.z ) ) );	// check if it should be "x" and "y", not "z"
	//float elevation = 0.6f * rr * exp2f( -10.0f * rr );
	float elevation = ( 0.5f - 0.045f * 0.75f ) - 6.0f * rr * exp2f( -10.0f * rr );
	for ( int i = 1; i < 7; i++ )
	{
		// Compute current tentacle's angle (around Y axis)
		float angle = ( 6.2831f/*2 pi*/ / 7.0f ) * static_cast< float >( i );
		// Domain distorsion : add noise in angle to deform tentacle
		//angle += 0.4f * rr * noise3f( make_float3( 4.0f * rr, 6.3f * static_cast< float >( i ), angle ) );
		angle += 0.40f * rr * noise3f( make_float3( 4.0f * rr, 2.5f, angle ) );
		angle += 0.29f;
		// Rotate point around Y axis
		float rc = cosf( angle );
		float rs = sinf( angle );
		float3 q = make_float3( x.x * rc - x.z * rs, x.y, x.x * rs + x.z * rc );
		// Add elevation to tentacles at monster center
		q.y += elevation;
		// Compute distance to ... (X axis)
		const float dd = dot( make_float2( q.y, q.z ), make_float2( q.y, q.z ) );
		if ( q.x > 0.0f && q.x < 1.5f && dd < distance )
		{
			distance = dd;
		}
	}
	// Distance to tentacles
	const float dist1 = sqrtf( distance ) - 0.045f;

	// Generate monster body
	//
	// - distance to ball (sphere of radius 0.30f)
	const float dist2 = r - 0.30f;

	// Compute blending factor between body and tentacles
	// - could also use smoothstep() function
	const float blendingFactor = clamp( r * 3.0f, 0.0f, 1.0f );

	// Blend shapes by mixing them
	return dist1 * blendingFactor + dist2 * ( 1.0f - blendingFactor );
}

/******************************************************************************
 * Distance field function
 *
 * @param pos 3D position
 ******************************************************************************/
__device__ __forceinline__
float map( float3 pos, int& sid, int& submat )
{
	submat = 0;

	float minDistance = 10000000.f;
	float distance;
	float fx;
	float fz;

	//-----------------------
	// Floor (suelo)
	//-----------------------
	if ( cFloorState )
	{
		distance = pos.y;
		float2 axz = make_float2( 128.0f ) + 6.0f * make_float2( pos.x + pos.z, pos.x - pos.z );
		int2 ixz = make_int2( floorf( axz ) );
		submat = icoolfFunc3d2( ixz.x + 53 * ixz.y );
		float2 peldxz = fracf( axz );
		float peld = smoothstep( 0.975f, 1.0f, fmaxf( peldxz.x, peldxz.y ) );
		if ( ( ( ( submat >> 10 ) & 7 ) > 6 ) )
		{
			peld = 1.0f;
		}
		distance += 0.005f * peld;
		minDistance = distance;

		sid = 0;
		if ( peld > 0.0000001f )
		{
			// Update material index
			sid = 2;
		}
	}

	//-----------------------
	// Ceiling (techo)
	//-----------------------
	if ( cCeilingState )
	{
		fx = fracf( pos.x + 128.0f );
		fz = fracf( pos.z + 128.0f );
		if ( pos.y > 1.0f )
		{
			distance = max( techo( fx, pos.y ), techo( fz, pos.y ) );
			if ( distance < minDistance )
			{
				minDistance = distance;

				// Update material index
				sid = 5;
			}
		}
	}
	
	//-----------------------
	// columnas
	//-----------------------
	if ( cColumnsState )
	{
		fx = fracf( pos.x + 128.0f + .5f );
		fz = fracf( pos.z + 128.0f + .5f );

		distance = columna( fx - .5f, pos.y, fz - .5f, minDistance, /*unused parameter*/13.1f * floorf( pos.x ) + 17.7f * floorf( pos.z ) );
		if ( distance < ( minDistance * minDistance ) )
		{
			minDistance = sqrtf( distance );

			// Update material index
			sid = 1;
		}
	}

	//-----------------------
	// Distance to monster
	//-----------------------
	if ( cMonsterState )
	{
		distance = distanceToMonster( pos, minDistance/*ununsed*/ );
		if ( distance < minDistance )
		{
			minDistance = distance;

			// Update material index
			sid = 4;
		}
	}

	return minDistance;
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
__forceinline__ uint ProducerKernel< TDataStructureType >
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
template < typename TDataStructureType >
template < typename GPUPoolKernelType >
__device__
__forceinline__ uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType &dataPool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GsLocalizationInfo &parentLocInfo, Loki::Int2Type< 1 > )
{
	// Retrieve current brick localization information code and depth
	const GvCore::GsLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GsLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	// Shared memory declaration
	//
	// Each threads of a block process one and unique brick of voxels.
	// We store in shared memory common useful variables that will be used by each thread.
	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
    __shared__ float3 levelResInv;
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	brickRes = BrickRes::get();
    levelRes = make_uint3( 1 << parentLocDepth.get() ) * brickRes;
    levelResInv = make_float3( 1.0f ) / make_float3( levelRes );

    brickPos = make_int3( parentLocCode.get() * brickRes ) - BorderSize;
    brickPosF = make_float3( brickPos ) * levelResInv;

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

    // - number of voxels
    const uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
    const int nbVoxels = elemSize.x * elemSize.y * elemSize.z;
    // - number of threads
    //const int nbThreads = blockDim.x * blockDim.y * blockDim.z;
    const int nbThreads = blockDim.x;
    // - global thread index in the block (linearized)
    //const int threadIndex1D = threadIdx.z * ( blockDim.x * blockDim.y ) + ( threadIdx.y * blockDim.x + threadIdx.x ); // written in FMAD-style
    const int threadIndex1D = threadIdx.x;
    uint3 locOffset;
	for ( int index = threadIndex1D; index < nbVoxels; index += nbThreads )
	{
		// Transform 1D per block?s global thread index to associated thread?s 3D voxel position
		locOffset.x = index % elemSize.x;
		locOffset.y = ( index / elemSize.x ) % elemSize.y;
		locOffset.z = index / ( elemSize.x * elemSize.y );

		// Position of the current voxel's center (relative to the brick)
		float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
		// Position of the current voxel's center (absolute, in [0.0;1.0] range)
		float3 voxelPosF = brickPosF + voxelPosInBrickF;
		// Position of the current voxel's center (scaled to the range [-1.0;1.0])
		float3 posF = voxelPosF * 2.0f - 1.0f;

		posF += cEyePos;
		posF *= cMapScale;

		// Distance field function
		int matId = 666;
		int subMatId;
		float distance = map( posF, matId, subMatId );

		// Initial data
		float4 voxelColor = make_float4( 0.f );
		
		// Object color
		float3 rgb = make_float3( 0.f );

		if ( matId != 666 )
		{
			float eps = 0.5f * levelResInv.x;

			// unused vars
			int m1, m2;

			// Normal
			// - computed by central differences on the distance field at the shading point (gradient approximation)
			float3 normal;
			normal.x = map( make_float3( posF.x + eps, posF.y, posF.z ), m1, m2 ) - map( make_float3( posF.x - eps, posF.y, posF.z ), m1, m2 );
			normal.y = map( make_float3( posF.x, posF.y + eps, posF.z ), m1, m2 ) - map( make_float3( posF.x, posF.y - eps, posF.z ), m1, m2 );
			normal.z = map( make_float3( posF.x, posF.y, posF.z + eps ), m1, m2 ) - map( make_float3( posF.x, posF.y, posF.z - eps ), m1, m2 );
			normal = normalize( normal );

			// Bump mapping
			if ( cUseBumpMapping )
			{
				// Computed by adding the gradient of a fractal sum of Perlin noise functions to the surface normal.
				// Bump is small and depend on the material
				const float kke = 0.0001f;
				float bumpa = 0.0075f;
				if ( matId != 5 )
				{
					bumpa *= 0.75f;
				}
				if ( matId == 4 )
				{
					bumpa *= 0.50f;
				}
				bumpa /= kke;
				float kk = fbm( 32.0f * posF );
				normal.x += bumpa * ( fbm( 32.0f * make_float3( posF.x + kke, posF.y, posF.z ) ) - kk );
				normal.y += bumpa * ( fbm( 32.0f * make_float3( posF.x, posF.y + kke, posF.z ) ) - kk );
				normal.z += bumpa * ( fbm( 32.0f * make_float3( posF.x, posF.y, posF.z + kke ) ) - kk );
				normal = normalize( normal );
			}

			// Lighting
			//
			// - common diffuse term
			float3 light = make_float3( 0.5f - posF.x, 0.8f - posF.y, 1.5f - posF.z );
			float llig = dot( light, light );
			float im = rsqrtf( llig );
			light = light * im;
			float diffuse = dot( normal, light );
			if ( matId == 4 )
			{
				diffuse = 0.5f + 0.5f * diffuse;
			}
			else
			{
				diffuse = 0.1f + 0.9f * diffuse;
			}
			//if( diffuse < 0.0f )
			//{
			//	diffuse = 0.0f;
			//}
			//diffuse = max( diffuse, 0.0f );
			diffuse = clamp( diffuse, 0.0f, 1.0f );
			diffuse *= 2.5f * exp2f( -1.75f * llig );
			float dif2 = ( normal.x + normal.y ) * 0.075f;
			// - common specular term
			float specular = 0.0f;

			// Materials
			if ( matId == 0 )
			{
				float xoff = 13.1f * float( subMatId & 255 );
				float fb = fbm( 16.0f * make_float3( posF.x + xoff, posF.y, posF.z ) );
				rgb = make_float3( 0.7f ) + fb * make_float3( 0.20f, 0.22f, 0.25f );

				float baldscale = float( ( subMatId >> 9 ) & 15 ) / 14.0f;
				baldscale = 0.51f + 0.34f * baldscale;
				rgb *= baldscale;
				float fx = 1.0f;
				if ( ( subMatId & 256 ) != 0 )
				{
					fx = -1.0f;
				}
				float m = sin( 64.0f * posF.z * fx + 64.0f * posF.x + 4.0f * fb );
				m = smoothstep( 0.25f, 0.5f, m ) - smoothstep( 0.5f, 0.75f, m );
				rgb += m * make_float3( 0.15f );
			}
			else if ( matId == 2 ) // floor
			{
				rgb = make_float3( 0.0f );
			}
			else if ( matId == 1 ) // columns
			{
				float fb = fbm( 16.0f * posF );
				float m = sin( 64.0f * posF.z + 64.0f * posF.x + 4.0f * fb );
				m = smoothstep( 0.30f, 0.5f, m ) - smoothstep( 0.5f, 0.70f, m );
				rgb = make_float3( 0.59f ) + fb * make_float3( 0.17f, 0.18f, 0.21f ) + m * make_float3( 0.15 ) + make_float3( dif2 );
			}
			else if ( matId == 4 ) // monster
			{
				float ft = fbm( 16.0f * posF );
				rgb = make_float3( 0.82f, 0.73f, 0.65f ) + ft * make_float3( 0.1f );

				float fs = 0.9f + 0.1f * fbm( 32.0f * posF );
				rgb *= fs;

				float fre = 0.9f;//max( -dot( normal, rd ), 0.0f);
				rgb -= make_float3( fre * fre * 0.45f );

				// Specular
				specular = clamp( ( normal.y - normal.z ) * 0.707f, 0.0f, 1.0f );
				specular = 0.20f * powf( specular, 32.0f );
			}
			// techo
			else //if( matID==5 )
			{
				float fb = fbm( 16.0f * posF );
				rgb = make_float3( 0.64f, 0.61f, 0.59f ) + fb * make_float3( 0.21f, 0.19f, 0.19f ) + dif2;
			}

			// Ambient occlusion
			float ao = 1.f;
			if ( cUseAmbientOcclusion )
			{
				// Fake and fast Ambient Occlusion.
				// VERY CHEAP, even cheaper than primary rays! Only 5 distance evaluations
				// instead of casting thousand of rays/evaluations.
				//
				// In a regular raytracer, primary rays/AO cost is 1:2000. Here, it�s 3:1 (that�s
				// almost four orders of magnitude speedup!).
				// It�s NOT the screen space trick (SSAO), but 3D.
				// The basic technique was invented by Alex Evans, aka Statix (�Fast
				// Approximation for Global Illumnation on Dynamic Scenes�, 2006). Greets to him!
				//
				// The idea: let p be the point to shade. Sample the distance field at a few (5)
				// points around p and compare the result to the actual distance to p. That
				// gives surface proximity information that can easily be interpreted as an
				// (ambient) occlusion factor.
				float totao = 0.0f;
				float sca = 10.0f;
				for ( int aoi = 0; aoi < 5; aoi++ )
				{
					float hr = 0.01f + 0.015f * float( aoi * aoi );
					float3 aopos =  normal * hr + posF;
					int kk, kk2;
					float dd = map( aopos, kk, kk2 );
					ao = -( dd - hr );
					totao += ao * sca;
					sca *= 0.5f;
				}
				ao = 1.0f - clamp( totao, 0.0f, 1.0f );
			}

			// Soft shadows
			if ( cUseSoftShadows )
			{
				//
				// Fake and fast soft shadows.
				// Only 6 distance evaluations used instead of casting hundrends of rays.
				// Pure geometry-based, not bluring.
				// Recipe: take n points on the line from the surface to the light and evaluate
				// the distance to the closest geometry. Find a magic formula to blend the n
				// distances to obtain a shadow factor.
				float so = 0.0f;
				int kk, kk2;
				for ( int i = 0; i < 6; i++ )
				{
					const float h = static_cast< float >( i ) / 6.0f;
					const float hr = 0.01f + h;
					const float3 aopos = light * hr + posF;
					float dd = map( aopos, kk, kk2 );
					so += ( 1.0f - h ) * dd * 2.0f * ( 10.0f / 6.0f );
				}
				diffuse *= clamp( ( so - 0.40f ) * 1.5f, 0.0f, 1.0f );
			}

			// Ambient, diffuse and specular lighting model
			rgb = rgb * ( ao * make_float3( 0.25f, 0.30f, 0.35f ) + diffuse * make_float3( 1.95f, 1.65f, 1.05f ) ) + make_float3( specular );

			// Fog
			//rgb = rgb * exp2( -0.4f * t );
		}

		// Color correct
		rgb = ( make_float3( sqrtf( rgb.x ), sqrtf( rgb.y ), sqrtf( rgb.z ) ) * 0.7f + 0.3f * rgb ) * make_float3( 0.83f, 1.0f, 0.83f ) * 1.2f;

		// Vigneting
		//rgb *= 0.25f + 0.75f * clamp( 0.60f * fabsf( pixel.x - 1.0f ) * fabsf( pixel.x + 1.0f ), 0.0f, 1.0f );
		rgb *= 0.25f + 0.75f * 0.6f;

		// Final color
		voxelColor = make_float4( /*color*/rgb, /*alpha*/1.f - clamp( .5f * levelRes.x * distance, 0.f, 1.f ) );

		// Alpha pre-multiplication used to avoid "color bleeding" effect
		voxelColor.x *= voxelColor.w;
		voxelColor.y *= voxelColor.w;
		voxelColor.z *= voxelColor.w;

		// Compute the new element's address
		uint3 destAddress = newElemAddress + locOffset;
		// Write the voxel's color in the first field
		dataPool.template setValue< 0 >( destAddress, voxelColor );
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
template < typename TDataStructureType >
__device__
__forceinline__ GsOracleRegionInfo::OracleRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth )
{
	// Limit the depth.
	// Currently, 32 is the max depth of the GigaVoxels engine.
	if ( regionDepth >= 32 )
	{
		return GsOracleRegionInfo::GPUVP_DATA_MAXRES;
	}

	// Shared memory declaration
	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv;

	brickRes = BrickRes::get();
	levelRes = make_uint3(1 << regionDepth) * brickRes;
	levelResInv = make_float3(1.0f) / make_float3(levelRes);

	int3 brickPos = make_int3(regionCoords * brickRes) - BorderSize;
	float3 brickPosF = make_float3(brickPos) * levelResInv;

    uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	
	// Iterate through voxels of current node
	uint3 elemOffset;
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z++ )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y++ )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x++ )
			{
				// Position of the current voxel's center (relative to the brick)
				const float3 voxelPosInBrickF = ( make_float3( elemOffset ) + 0.5f ) * levelResInv;
				// Position of the current voxel's center (absolute, in [0.0;1.0] range)
				const float3 voxelPosF = brickPosF + voxelPosInBrickF;

				// Position of the current voxel's center (scaled to the range [-1.0;1.0])
				float3 posF = voxelPosF * 2.0f - 1.0f;

				posF += cEyePos;
				posF *= cMapScale;

				// If the distance at the position is less than the size of one voxel, the brick is not empty
				int matId;
				int subMatId;
				if ( map( posF, matId, subMatId ) <= levelResInv.x )
				{
					// Exit loop as soon as Oracle knows it can
					return GsOracleRegionInfo::GPUVP_DATA;
				}
			}
		}
	}

	return GsOracleRegionInfo::GPUVP_CONSTANT;
}
