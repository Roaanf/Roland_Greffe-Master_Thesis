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

#ifndef _GPU_Tree_BVH_Common_h_
#define _GPU_Tree_BVH_Common_h_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

//////////////////////////////////
//BVH intersection
#define BVH_NUM_THREADS_PER_BLOCK_X 16
#define BVH_NUM_THREADS_PER_BLOCK_Y 12
///////////////////////////////////

/**
 * ...
 */
#define BVH_TRAVERSAL_STACK_SIZE 16

/**
 * ...
 */
#define BVH_DATA_PAGE_SIZE 32

/**
 * ...
 */
#define BVH_NODE_POOL_SIZE 4147426 //1048576  //NB real: 908460  //262144
/**
 * ...
 */
#define BVH_VERTEX_POOL_SIZE  4000000//983040 //7864320 //491520 //15728640 //30720 //52428800 //41943040 //2097152

/**
 * ...
 */
#define BVH_USE_COLOR 1
/**
 * ...
 */
#define BVH_TRAVERSAL_USE_MASKSTACK 0

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

#endif // !
