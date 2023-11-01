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

#ifndef _CUSTOM_PRIORITY_POLICIES_MANAGER_KERNEL_H_
#define _CUSTOM_PRIORITY_POLICIES_MANAGER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GsIPriorityPoliciesManagerKernel.h"

// Project
#include "CustomPriorityPolicies.h"

// STL
#include <string>

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Select which priority to use.
 */
__constant__ PriorityPolicies cPriorityPolicy;

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class CustomPriorityPoliciesManager
 *
 * This class implements the interface to provide custom priority management.
 */
class CustomPriorityPoliciesManager : public GvCore::GsIPriorityPoliciesManagerKernel< CustomPriorityPoliciesManager >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Return the priority to use for a given node.
	 *
	 * @param pParams the priority parameters
	 *
	 * @return the priority for the node.
	 */
	__device__ __forceinline__
	static int getNodePriorityImpl( const GvCore::GsPriorityParameters& pParams );

	/**
	 * Return the priority to use for a given brick.
	 *
	 * @param pParams the priority parameters
	 *
	 * @return the priority for the brick.
	 */
	__device__ __forceinline__
	static int getBrickPriorityImpl( const GvCore::GsPriorityParameters& pParams );

	/**
	 * Return a flag telling whether or not to use priority.
	 *
	 * @return A flag telling whether or not to use priority
	 */
	__host__ __device__ __forceinline__
	static bool usePriorityImpl();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
				
};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "CustomPriorityPoliciesManager.inl"

#endif // !_CUSTOM_PRIORITY_POLICIES_MANAGER_KERNEL_H_
