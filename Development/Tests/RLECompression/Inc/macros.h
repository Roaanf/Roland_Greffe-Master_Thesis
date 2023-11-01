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

#ifndef _MACRO_H_
#define _MACRO_H_

// Variables used for the tests.
#define N_BRICKS 2000
#define MAX_COMPRESSION 250

// Size of the blocs of data used to compress (no link with the GigaVoxels bricks).
#define BRICK_SIZE 250
#if BRICK_SIZE >= UCHAR_MAX
#error "BRICK_SIZE is too big"
#endif

// Max size for the compressed brick (if it's bigger, we gain nothing by compressing it). 
#define MAX_COMPRESSED_BRICK_SIZE (( BRICK_SIZE * sizeof( unsigned int )) / ( sizeof( unsigned int ) + sizeof( unsigned char ) ))


// Size of the array containing compressed data.
#define STR_SIZE N_BRICKS * BRICK_SIZE

// Variables managing the number of threads.
#define GRID_SIZE 1000
#define N_THREADS_PER_BLOCS 192
#define WARP_SIZE 32
#define N_WARPS_PER_BLOCS N_THREADS_PER_BLOCS / WARP_SIZE

#endif // _MACRO_H_
