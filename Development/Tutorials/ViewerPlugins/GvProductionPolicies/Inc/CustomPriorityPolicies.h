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

#ifndef _CUSTOM_PRIORITY_POLICIES_H_
#define _CUSTOM_PRIORITY_POLICIES_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/**
 * Custom type of priority policies used to produce data
 */
enum PriorityPolicies
{
	/**
	 * Don't try to sort requests.
	 */
	ePriorityPolicy_noPriority,
	/**
	 * Default formula used in the common priority policy manager.
	 */
	ePriorityPolicy_default,
	/**
	 * Give priority to the bricks/node that are near from the camera.
	 */
	ePriorityPolicy_nearest,
	/**
	 * Give priority to the bricks/node that are far from the camera.
	 */
	ePriorityPolicy_farthest,
	/**
	 * Give priority to the bricks/node that are the most detailed.
	 */
	ePriorityPolicy_mostDetailedFirst,
	/**
	 * Give priority to the bricks/node that are the least detailed.
	 */
	ePriorityPolicy_leastDetailedFirst,
	/**
	 * Give priority to the bricks/node that are far from their optimal size.
	 */
	ePriorityPolicy_farthestFromOptimalSize,
	/**
	 * Number of different types of policies
	 */
	eNbPriorityPolicies
};

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#endif // !_CUSTOM_PRIORITY_POLICIES_H_
