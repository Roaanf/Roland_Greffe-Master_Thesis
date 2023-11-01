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
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * ...
 ******************************************************************************/
inline uint3 Scene::getFather( const uint3& locCode ) const
{
	return locCode / make_uint3( 2 );
}

/******************************************************************************
 * Compute global index of a node in the node buffer given its depth and code localization info
 *
 * TODO : use generic code => only valid for octree...
 *
 * @param pDepth node's depth localization info
 * @param pCode node's code localization info
 *
 * return node's global index in the node buffer
 ******************************************************************************/
inline unsigned int Scene::getIndex( unsigned int pDepth, const uint3& pCode ) const
{
	// Compute the basis of the code
	const unsigned int b = static_cast< unsigned int >( powf( 2, pDepth ) );	// nb nodes at given depth d

	return ( powf( 8.f, static_cast< float >( pDepth ) ) - 1.f ) / static_cast< float >( 7.f ) + ( pCode.x + pCode.y * b + pCode.z * b * b );
}

/******************************************************************************
 * ...
 ******************************************************************************/
inline bool Scene::triangleIntersectBick( const float3& brickPos, 
										  const float3& brickSize,  
										  unsigned int triangleIndex, 
										  const std::vector< unsigned int >& IBO, 
										  const float* vertices )
{
	// We assume here that triangle are much smaller than brick ( at least as small as voxel's brick ) so we simplify the intersection test.
	// We only test if triangle's vertices are in the bick
	
	// Test vertices
	return vertexIsInBrick( brickPos, brickSize, IBO[ triangleIndex + 0 ], vertices ) || 
		   vertexIsInBrick( brickPos, brickSize, IBO[ triangleIndex + 1 ], vertices ) ||
		   vertexIsInBrick( brickPos, brickSize, IBO[ triangleIndex + 2 ], vertices ) ;
}

/******************************************************************************
 * ...
 ******************************************************************************/
inline bool Scene::vertexIsInBrick( const float3& brickPos, 
								   const float3& brickSize, 
							       unsigned int vertexIndex,
							       const float* vertices ) 
{
	return ( ( brickPos.x <= vertices[ 3 * vertexIndex + 0 ] && brickPos.x + brickSize.x >= vertices[ 3 * vertexIndex + 0 ] ) &&
		     ( brickPos.y <= vertices[ 3 * vertexIndex + 1 ] && brickPos.y + brickSize.y >= vertices[ 3 * vertexIndex + 1 ] ) &&
			 ( brickPos.z <= vertices[ 3 * vertexIndex + 2 ] && brickPos.z + brickSize.z >= vertices[ 3 * vertexIndex + 2 ] ) ) ;

}
