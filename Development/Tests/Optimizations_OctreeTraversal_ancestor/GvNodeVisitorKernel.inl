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
#include "GvRendering/GvRendererHelpersKernel.h"
#include "GvRendering/GvRendererContext.h"
#include "GvPerfMon/GvPerformanceMonitor.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvRendering
{

/******************************************************************************
 * Descent in data structure (in general octree) until max depth is reach or current traversed node has no subnodes,
 * or cone aperture is greater than voxel size.
 *
 * @param pVolumeTree the data structure
 * @param pGpuCache the cache
 * @param node a node that user has to provide. It will be filled with the final node of the descent
 * @param pSamplePosTree A given position in tree
 * @param pConeAperture A given cone aperture
 * @param pNodeSizeTree the returned node size
 * @param pSampleOffsetInNodeTree the returned sample offset in node tree
 * @param pBrickSampler The sampler object used to sample data in the data structure, it will be initialized after the descent
 * @param pRequestEmitted a returned flag to tell wheter or not a request has been emitted during descent
 ******************************************************************************/
template< bool priorityOnBrick, class TVolTreeKernelType, class GPUCacheType >
__device__
__forceinline__ void GvNodeVisitorKernel
::visit( TVolTreeKernelType& pVolumeTree, GPUCacheType& pGpuCache, GvStructure::GvNode& pNode,
		const float3 pSamplePosTree, const float pConeAperture, float& pNodeSizeTree, float3& pSampleOffsetInNodeTree,
		GvSamplerKernel< TVolTreeKernelType >& pBrickSampler, bool& pRequestEmitted )
{
	// Useful variables initialization
	uint nodeDepth = 0;
	float3 nodePosTree = make_float3( 0.0f );
	pNodeSizeTree = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	float voxelSizeTree = pNodeSizeTree / static_cast< float >( TVolTreeKernelType::BrickResolution::maxRes );

	uint brickChildAddressEnc  = 0;
	uint brickParentAddressEnc = 0;

	float3 brickChildNormalizedOffset = make_float3( 0.0f );
	float brickChildNormalizedScale  = 1.0f;

	// Initialize the address of the first node in the "node pool".
	// While traversing the data structure, this address will be
	// updated to the one associated to the current traversed node.
	// It will be used to fetch info of the node stored in then "node pool".
	uint nodeTileAddress = pVolumeTree._rootAddress;

#ifdef FLOAT_DESCENT
	float nodeSizeTreeInv = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
#else // FLOAT_DESCENT
	const uint maxDepth = 32; // TODO 32 => constant ?
	const float scale = static_cast< float >(( unsigned int )( 1 )<< ( maxDepth - 1 ));
	uint bitsToShift = maxDepth - 1;
	uint3 initialSamplePosValue = make_uint3( pSamplePosTree * scale );
#endif // FLOAT_DESCENT

	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	bool descentSizeCriteria;
	do {
		// Bitwise tree traversal
		//http://www.merl.com/papers/docs/TR2002-41.pdf

		// [ 1 ] - Update size parameters
		pNodeSizeTree		*= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current node size
		voxelSizeTree		*= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current voxel size
#ifdef FLOAT_DESCENT
		nodeSizeTreeInv		*= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );			// current node resolution (nb nodes in a dimension)
#endif //FLOAT_DESCENT

		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"

#ifdef FLOAT_DESCENT
		uint3 nodeChildCoordinates = make_uint3( nodeSizeTreeInv * ( pSamplePosTree - nodePosTree ) );
		uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );// & nodeChildAddressMask;
		uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
		nodePosTree = nodePosTree + pNodeSizeTree * make_float3( nodeChildCoordinates );
#else // FLOAT_DESCENT
		// P* NodePosTree is the current origin which is shifted according to the chosen child node
		// P* nodeChildAddressOffset is an offset ranging from 0-7 

		// New code 0, uncomment this to run the new way with integers
		// Note (Prashant) : Modify this for a general N^3
		// --------------------------------------------
		uint3 nodeChildCoordinates;

		nodeChildCoordinates.x = initialSamplePosValue.x >> bitsToShift;
		nodeChildCoordinates.y = initialSamplePosValue.y >> bitsToShift;
		nodeChildCoordinates.z = initialSamplePosValue.z >> bitsToShift;
		--bitsToShift;

		nodePosTree = pNodeSizeTree * make_float3( nodeChildCoordinates );

		nodeChildCoordinates.x &= 0x1;
		nodeChildCoordinates.y &= 0x1;
		nodeChildCoordinates.z &= 0x1;

		uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );
		uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
#endif // FLOAT_DESCENT

		// Try to retrieve node from the node pool given its address
		pVolumeTree.fetchNode( pNode, nodeAddress );

		// Update brick info
		if ( brickChildAddressEnc ) {
			brickParentAddressEnc = brickChildAddressEnc;
			brickChildNormalizedScale  = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );		// 0.5f;
			brickChildNormalizedOffset = brickChildNormalizedScale * make_float3( nodeChildCoordinates );
		} else {
			brickChildNormalizedScale  *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// 0.5f;
			brickChildNormalizedOffset += brickChildNormalizedScale * make_float3( nodeChildCoordinates );
		}
		brickChildAddressEnc = pNode.hasBrick() ? pNode.getBrickAddressEncoded() : 0;

		// Update descent condition
		descentSizeCriteria = ( voxelSizeTree > pConeAperture ) && ( nodeDepth < k_maxVolTreeDepth );

		// Update octree depth
		++nodeDepth;

		// ---- Flag used data (the traversed one) ----

		// Set current node as "used"
		pGpuCache._nodeCacheManager.setElementUsage( nodeTileAddress );

		// Set current brick as "used"
		if ( pNode.hasBrick() )
		{
			pGpuCache._brickCacheManager.setElementUsage( pNode.getBrickAddress() );
		}

		// ---- Emit requests if needed (node subdivision or brick loading/producing) ----

		// Process requests based on traversal strategy (priority on bricks or nodes)
		if ( priorityOnBrick )
		{
			// Low resolution first						  
			if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) )
			{
				pGpuCache.loadRequest( nodeAddress );
				pRequestEmitted = true;
			}
			else if ( !pNode.hasSubNodes() && descentSizeCriteria && !pNode.isTerminal() )
			{
				pGpuCache.subDivRequest( nodeAddress );
				pRequestEmitted = true;
			}
		}
		else
		{	 // High resolution immediatly
			if ( descentSizeCriteria && !pNode.isTerminal() )
			{
				if ( ! pNode.hasSubNodes() )
				{
					pGpuCache.subDivRequest( nodeAddress );
					pRequestEmitted = true;
				}
			}
			else if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) )
			{
				pGpuCache.loadRequest( nodeAddress );
				pRequestEmitted = true;
			}
		}

		nodeTileAddress = pNode.getChildAddress().x;

		//scaleFactor *= 2.0;
	} while ( descentSizeCriteria && pNode.hasSubNodes() );	// END of the data structure traversal

	// Compute sample offset in node tree
	pSampleOffsetInNodeTree = pSamplePosTree - nodePosTree;

	// Update brickSampler properties
	if ( pNode.isBrick() )
	{
		pBrickSampler._nodeSizeTree = pNodeSizeTree;
		pBrickSampler._sampleOffsetInNodeTree = pSampleOffsetInNodeTree;
		pBrickSampler._scaleTree2BrickPool = pVolumeTree.brickSizeInCacheNormalized.x / pBrickSampler._nodeSizeTree;

		//uint3 tmpAddressUnenc = GvStructure::GvNode::unpackBrickAddress( brickParentAddressEnc ) - make_uint3(1, 1, 1);
		//uint tmpAddressEnc = GvStructure::GvNode:ackBrickAddress(tmpAddressUnenc);

		pBrickSampler._brickParentPosInPool = pVolumeTree.brickCacheResINV * make_float3( GvStructure::GvNode::unpackBrickAddress( brickParentAddressEnc)) 
			+ brickChildNormalizedOffset * pVolumeTree.brickSizeInCacheNormalized.x;

		if ( brickChildAddressEnc )
		{
			// Should be mipmapping here, between level with the parent

			//pBrickSampler.mipMapOn = true; // "true" is not sufficient :  when no parent, program is very slow.
			pBrickSampler._mipMapOn = ( brickParentAddressEnc == 0 ) ? false : true;

			pBrickSampler._brickChildPosInPool = make_float3( GvStructure::GvNode::unpackBrickAddress( brickChildAddressEnc ) ) * pVolumeTree.brickCacheResINV;
		}
		else
		{
			// No mipmapping here

			pBrickSampler._mipMapOn = false;
			pBrickSampler._brickChildPosInPool  = pBrickSampler._brickParentPosInPool;
			pBrickSampler._scaleTree2BrickPool *= brickChildNormalizedScale;
		}
	}
}

/******************************************************************************
 * Descent in volume tree until max depth is reach or current traversed node has no subnodes.
 * Perform a descent in a volume tree from a starting node tile address, until a max depth
 * Given a 3D sample position, 
 *
 * @param pVolumeTree The volume tree on which descent in done
 * @param pMaxDepth Max depth of the descent
 * @param pSamplePos 3D sample position
 * @param pNodeTileAddress ...
 * @param pNode ...
 * @param pNodeSize ...
 * @param pNodePos ...
 * @param pNodeDepth ...
 * @param pBrickAddressEnc ...
 * @param pBrickPos ...
 * @param pBrickScale ...
 ******************************************************************************/
template< class VolumeTreeKernelType >
__device__
__forceinline__ void GvNodeVisitorKernel
::visit( VolumeTreeKernelType& pVolumeTree, uint pMaxDepth, float3 pSamplePos,
		 uint pNodeTileAddress, GvStructure::GvNode& pNode, float& pNodeSize, float3& pNodePos, uint& pNodeDepth,
		 uint& pBrickAddressEnc, float3& pBrickPos, float& pBrickScale )
{
	////descent////

	float nodeSizeInv = 1.0f;

	// WARNING uint nodeAddress;
	pBrickAddressEnc = 0;

	// Descent in volume tree until max depth is reach or current traversed node has no subnodes
	int i = 0;
	do
	{
		pNodeSize	*= 1.0f / (float)VolumeTreeKernelType::NodeResolution::maxRes;
		nodeSizeInv	*= (float)VolumeTreeKernelType::NodeResolution::maxRes;

		// Retrieve current voxel position
		uint3 curVoxel = make_uint3( nodeSizeInv * ( pSamplePos - pNodePos ) );
		uint curVoxelLinear = VolumeTreeKernelType::NodeResolution::toFloat1( curVoxel );

		float3 nodeposloc = make_float3( curVoxel ) * pNodeSize;
		pNodePos = pNodePos + nodeposloc;

		// Retrieve pNode info (child and data addresses) from pNodeTileAddress address and curVoxelLinear offset
		pVolumeTree.fetchNode( pNode, pNodeTileAddress, curVoxelLinear );

		if ( pNode.hasBrick() )
		{
			pBrickAddressEnc = pNode.getBrickAddressEncoded();
			pBrickPos = make_float3( 0.0f );
			pBrickScale = 1.0f;
		}
		else
		{
			pBrickScale = pBrickScale * 0.5f;
			pBrickPos += make_float3( curVoxel ) * pBrickScale;
		}

		pNodeTileAddress = pNode.getChildAddress().x;
		i++;
	}
	while ( ( i < pMaxDepth ) && pNode.hasSubNodes() );

	pNodeDepth = i;

	//i -= 1;		// <== TODO : don't seem to be used anymore, remove it
}

/******************************************************************************
 * Descent in data structure (in general octree) until max depth is reach or current traversed node has no subnodes,
 * or cone aperture is greater than voxel size.
 *
 * @param pVolumeTree the data structure
 * @param pGpuCache the cache
 * @param node a node that user has to provide. It will be filled with the final node of the descent
 * @param pSamplePosTree A given position in tree
 * @param pConeAperture A given cone aperture
 * @param pNodeSizeTree the returned node size
 * @param pSampleOffsetInNodeTree the returned sample offset in node tree
 * @param pBrickSampler The sampler object used to sample data in the data structure, it will be initialized after the descent
 * @param pRequestEmitted a returned flag to tell wheter or not a request has been emitted during descent
 ******************************************************************************/
template< class TVolTreeKernelType >
__device__
__forceinline__ void GvNodeVisitorKernel
::getNodeFather( TVolTreeKernelType& pVolumeTree, GvStructure::GvNode& pNode, const float3 pSamplePosTree, const uint pMaxNodeDepth )
{
	// Useful variables initialization
	uint nodeDepth = 0;
	float3 nodePosTree = make_float3( 0.0f );
	float nodeSizeTree = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	float nodeSizeTreeInv = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );

	// Initialize the address of the first node in the "node pool".
	// While traversing the data structure, this address will be
	// updated to the one associated to the current traversed node.
	// It will be used to fetch info of the node stored in the "node pool".
	uint nodeTileAddress = pVolumeTree._rootAddress;

	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	do
	{
		// [ 1 ] - Update size parameters
		nodeSizeTree		*= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current node size
		nodeSizeTreeInv		*= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );			// current node resolution (nb nodes in a dimension)

		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"
		uint3 nodeChildCoordinates = make_uint3( nodeSizeTreeInv * ( pSamplePosTree - nodePosTree ) );
		uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );// & nodeChildAddressMask;
		uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
		nodePosTree = nodePosTree + nodeSizeTree * make_float3( nodeChildCoordinates );
		// Try to retrieve node from the node pool given its address
		//pVolumeTree.fetchNode( pNode, nodeTileAddress, nodeChildAddressOffset );
		pVolumeTree.fetchNode( pNode, nodeAddress );

		nodeTileAddress = pNode.getChildAddress().x;

		// Update depth
		++nodeDepth;
	}
	while ( ( nodeDepth <= pMaxNodeDepth ) && pNode.hasSubNodes() );	// END of the data structure traversal
}

} // namespace GvRendering
