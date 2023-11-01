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

// Project
#include "CustomPriorityPolicies.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Return the priority to use for a given node.
 *
 * @param pParams the priority parameters
 *
 * @return the priority for the node.
 ******************************************************************************/
__device__
__forceinline__ int CustomPriorityPoliciesManager::getNodePriorityImpl( const GvCore::GsPriorityParameters& pParams )
{
	// CRITERIA OVERVIEW
	//
	// NOTE: lower values of priority have more priority (it comes from our sorting algorithm)
	//
	// NOTE:
	// By changing priorities, as the cache is not reinitialized inthis demo,
	// you may have incoherences cause it only applies to nodes or bricks to be produced.
	// The previously produced elements are not concerned.
	//
	// NOTE:
	// - the 10000 constant is used to transform "float values" into integer ones
	// - a value like MAX_INT can't be used, because it could result in an overflow (if other multiplied value is more than 1.0)
	switch ( cPriorityPolicy )
	{
		// No priority
		case ePriorityPolicy_noPriority:
			return 0;

		// Default priority:
		// - pParams._coneAperture : the smaller it is, the nearer it is from the camera
		// - pParams._nodeDepth : the higher it is, the more details it has
		// => so, here, the nearest and more refined elements will be produced first
		// => so, highly detailed elements will continue to be detailed
		//
		// NOTE : (Physics) units are not handled...
		case ePriorityPolicy_default:
			return pParams._nodeDepth * static_cast< int >( pParams._coneAperture ) * 10000;

		// Nearest elements first
		// - pParams._coneAperture : the smaller it is, the nearer it is from the camera
		// => so, here, the nearest elements will be produced first
		case ePriorityPolicy_nearest:
			return static_cast< int >( pParams._coneAperture ) * 10000;

		// Farthest elements first
		// - pParams._coneAperture : the smaller it is, the nearer it is from the camera
		// => so, here, as the sign is negated, the farthest elements will be produced first
		case ePriorityPolicy_farthest:
			return static_cast< int >( -pParams._coneAperture ) * 10000;

		// Most detailed elements first
		// - pParams._nodeDepth : the higher it is, the more detailed the element is
		// => so, here, as the sign is negated, the more refined elements will be produced first
		// => so, highly detailed elements will continue to be produced (subdivided or produced/loaded)
		case ePriorityPolicy_mostDetailedFirst:
			return -pParams._nodeDepth;

		// Least detailed elements first
		// - pParams._nodeDepth : the higher it is, the more detailed the element is
		// => so, here, the least refined elements will be produced first
		// => so, highly detailed elements will continue to be produced (subdivided or produced/loaded)
		case ePriorityPolicy_leastDetailedFirst:
			return pParams._nodeDepth;

		// Least detailed elements first
		// - should be equivalent to ePriorityPolicy_default by gives better results
		// - pParams._coneAperture : the smaller it is, the nearer it is from the camera
		// - pParams._nodeSize : the smaller it is, the more detailed the element is
		// => here, we take the ratio between the cone aperture and the node size
		// => so, the bigger a node is compared to the cone aperture, the more priority it has to be subdivided or produced/loaded
		case ePriorityPolicy_farthestFromOptimalSize:
			return 10000 * static_cast< int >( __fdividef( pParams._coneAperture, pParams._nodeSize ) );
		
		//case xxx:
		//	return static_cast< int >( -pParams._nodeSize );

		default:
			return 0;
	}
}

/******************************************************************************
 * Return the priority to use for a given brick.
 *
 * @param pParams the priority parameters
 *
 * @return the priority for the brick.
 ******************************************************************************/
__device__
__forceinline__ int CustomPriorityPoliciesManager::getBrickPriorityImpl( const GvCore::GsPriorityParameters& pParams )
{
	// Explanations
	// - same for nodes' version (check getNodePriorityImpl() method)

	switch ( cPriorityPolicy )
	{
		case ePriorityPolicy_noPriority:
			return 0;

		case ePriorityPolicy_default:
			return pParams._nodeDepth * static_cast< int >( pParams._coneAperture ) * 10000;

		case ePriorityPolicy_nearest:
			return static_cast< int >( pParams._coneAperture ) * 10000;

		case ePriorityPolicy_farthest:
			return static_cast< int >( -pParams._coneAperture ) * 10000;

		case ePriorityPolicy_mostDetailedFirst:
			return -pParams._nodeDepth;

		case ePriorityPolicy_leastDetailedFirst:
			return pParams._nodeDepth;

		case ePriorityPolicy_farthestFromOptimalSize:
			return 10000 * static_cast< int >( __fdividef( pParams._coneAperture, pParams._nodeSize ) );

		//case xxx:
		//	return static_cast< int >( -pParams._nodeSize );

		default:
			return 0;
	}
}

/******************************************************************************
 * Return a flag telling whether or not to use priority.
 *
 * @return A flag telling whether or not to use priority
 ******************************************************************************/
__host__ __device__
__forceinline__ bool CustomPriorityPoliciesManager::usePriorityImpl()
{
	return true;
}
