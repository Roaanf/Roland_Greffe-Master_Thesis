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

#include "SampleCore.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
//#include <cuda_runtime.h>
#include <cutil_math.h>

// Gigavoxels
#include <GvCore/StaticRes3D.h>
#include <GvStructure/GvVolumeTree.h>
#include <GvRenderer/VolumeTreeRendererCUDA_instancing.h>
#include <GvPerfMon/CUDAPerfMon.h>

// Simple Tore
#include "ToreProducer.h"
#include "ToreShader.h"

//#include <GvUtils/BuiltInVector.inl>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	(80U*1024U*1024U)		// 8 Mo
#define BRICKPOOL_MEMSIZE	(512U*1024U*1024U)		// 256 Mo

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
:	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_colorBuffer( 0 )
,	_depthBuffer( 0 )
,	_colorResource( 0 )
,	_depthResource( 0 )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 5 )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _volumeTreeRenderer;
	delete _volumeTreeCache;
	delete _volumeTree;
	delete _producer;
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Initialize CUDA with OpenGL Interoperability
	cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
	CUT_CHECK_ERROR( "cudaGLSetGLDevice" );

	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = NodeRes::numElements * sizeof( GvStructure::OctreeNode );
	size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< DataType >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( nodePoolNumElems ), 1.0f / 3.0f ) ) ) ) * NodeRes::get();
	uint3 brickPoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( brickPoolNumElems ), 1.0f / 3.0f ) ) ) ) * RealBrickRes::get();
	
	std::cout << "nodePoolRes: " << nodePoolRes << std::endl;
	std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Producer initialization
	_producer = new ProducerType();

	// Data structure initialization
	_volumeTree = new VolumeTreeType( nodePoolRes, brickPoolRes, 0 );
	_volumeTree->setMaxDepth( _maxVolTreeDepth );

	// Cache initialization
	_volumeTreeCache = new VolumeTreeCacheType( _volumeTree, _producer, nodePoolRes, brickPoolRes );

	// Renderer initialization
	_volumeTreeRenderer = new VolumeTreeRendererType( _volumeTree, _volumeTreeCache, _producer );
}















//template< bool rowMajor >
//__global__
//void inverseModelView( void *out_p, const void *in_p, size_t _n_ )
//{
////	const int couple[] = {
//	// thread identification in matrix
//	const int tidm = threadIdx.x;
//	const int tidmx = tidm & 0xF;
//	const int tidmy = tidm >> 4;
//	const int j = tidmx;
//	const int i = tidmy;
//	// matrix identification in bloc
//	const int midb = threadIdx.y;
//	// thread identification in bloc
//	const int tidb = tidm + midb * 16; // blockDim.x == 16
//	// matrix identification in grid
//	const int mid = midb + blockIdx.x * blockDim.x;
//	// thread identification in grid
//	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
//	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
//	const int tid = tidx + tidy * gridDim.x * blockDim.x;
//	// map from threadIdx/BlockIdx to pixel position
//	const int offset = tidb + blockIdx.x * 16 * blockDim.y; // blockDim.x == 16

//	__shared__ float ins[ MAT_PER_STEP ][ 16 ];
//	__shared__ float sout[ MAT_PER_STEP ][ 16 ];
//	const float *in = static_cast< const float* >( in_p );
//	float *out = static_cast< float* >( out_p );

//	// copy matrices in shared memory
//	if ( midb < MAT_PER_STEP )
//	{
//		if ( rowMajor )
//		{
//			ins[ midb ][ tidm ] = in[ offset ];
//		}
//		else
//		{
//			const int transposeShift[16] = {
//				 0,  3,  6,  9,
//				-3,  0,  3,  6,
//				-6, -6,  0,  3,
//				-9, -6, -3,  0
//			};
//			ins[ midb ][ tidm ] = in[ offset + transposeShift[tidm] ];
//		}
//	}

//	__syncthreads();

//	// inverse matrix
//	__shared__ float tmp[ MAT_PER_STEP ][ 9 ];
//	int n = tidb, nend = 16 * MAT_PER_STEP;
//	while( n < nend )
//	{
////		const int m = n >> 3; // corresponding matrix
////		const int el = n & 7; // corresponding operation
//		const int operands[ 6 * 12 ] = {
//			// similarity
//			5, 10,  9,  6, 12, 12,
//			2,  9, 10,  1, 12, 12,
//			1,  6,  5,  2, 12, 12,
//			6,  8, 10,  4, 12, 12,
//			0, 10,  8,  2, 12, 12,
//			2,  4,  6,  0, 12, 12,
//			4,  9,  8,  5, 12, 12,
//			1,  8,  9,  0, 12, 12,
//			0,  8,  4,  1, 12, 12,
//			// translation
//			0,  3,  1,  7,  2, 11,
//			4,  3,  5,  7,  6, 11,
//			8,  3,  9,  7, 10, 11
//		};


//		if ( tidm < 12 )
//		sout[ midb ][ tidm  ] = ins[ midb ][ operands[tidm+0] ] * ins[ midb ][ operands[tidm+1] ]
//							  - ins[ midb ][ operands[tidm+2] ] * ins[ midb ][ operands[tidm+3] ];
//							  - ins[ midb ][ operands[tidm+4] ] * ins[ midb ][ operands[tidm+5] ];


//	}

//	// translation component


//	// copy result in output
//	out[ offset ] = sout[ mid ][ tidm ];
//}

__device__
float4x4 inverseCM( const float4x4 &mat )
{
    float4x4 ret;

    float idet = 1.0f / det(mat);
    ret._array[0] =  (mat._array[5] * mat._array[10] - mat._array[9] * mat._array[6]) * idet;
    ret._array[1] = -(mat._array[1] * mat._array[10] - mat._array[9] * mat._array[2]) * idet;
    ret._array[2] =  (mat._array[1] * mat._array[6] - mat._array[5] * mat._array[2]) * idet;
    ret._array[3] = 0.0;
    ret._array[4] = -(mat._array[4] * mat._array[10] - mat._array[8] * mat._array[6]) * idet;
    ret._array[5] =  (mat._array[0] * mat._array[10] - mat._array[8] * mat._array[2]) * idet;
    ret._array[6] = -(mat._array[0] * mat._array[6] - mat._array[4] * mat._array[2]) * idet;
    ret._array[7] = 0.0;
    ret._array[8] =  (mat._array[4] * mat._array[9] - mat._array[8] * mat._array[5]) * idet;
    ret._array[9] = -(mat._array[0] * mat._array[9] - mat._array[8] * mat._array[1]) * idet;
    ret._array[10] =  (mat._array[0] * mat._array[5] - mat._array[4] * mat._array[1]) * idet;
    ret._array[11] = 0.0;
    ret._array[12] = -(mat._array[12] * ret._array[0] + mat._array[13] * ret._array[4] + mat._array[14] * ret._array[8]);
    ret._array[13] = -(mat._array[12] * ret._array[1] + mat._array[13] * ret._array[5] + mat._array[14] * ret._array[9]);
    ret._array[14] = -(mat._array[12] * ret._array[2] + mat._array[13] * ret._array[6] + mat._array[14] * ret._array[10]);
    ret._array[15] = 1.0;

    return ret;
}

__device__
float4x4 inverseRM( const float4x4 &mat )
{
    float4x4 ret;
    const int operands[ 6 * 12 ] = {
        // similarity
        5, 10,  9,  6, 12, 12,
        2,  9, 10,  1, 12, 12,
        1,  6,  5,  2, 12, 12,
        6,  8, 10,  4, 12, 12,
        0, 10,  8,  2, 12, 12,
        2,  4,  6,  0, 12, 12,
        4,  9,  8,  5, 12, 12,
        1,  8,  9,  0, 12, 12,
        0,  5,  4,  1, 12, 12,
        // translation
        0,  3,  1,  7,  2, 11,
        4,  3,  5,  7,  6, 11,
        8,  3,  9,  7, 10, 11
    };

//	float invDet = 1.0f / ( mat._array[2] * ret._array[0] + mat._array[1] * ret._array[4] + mat._array[0] * ret._array[8] );
    float idet = 1.0f / det(mat);
//	float idet = 1.0f / ( mat._array[2] * (mat._array[5] * mat._array[10] - mat._array[9] * mat._array[6])
//						  + mat._array[1] * (mat._array[6] * mat._array[8] - mat._array[10] * mat._array[4])
//						  + mat._array[0] * (mat._array[4] * mat._array[9] - mat._array[8] * mat._array[5]) );
    ret._array[0] = (mat._array[5] * mat._array[10] - mat._array[9] * mat._array[6]) * idet;
    ret._array[1] = (mat._array[2] * mat._array[9] - mat._array[10] * mat._array[1]) * idet;
    ret._array[2] = (mat._array[1] * mat._array[6] - mat._array[5] * mat._array[2]) * idet;
    ret._array[3] = 0.0;
    ret._array[4] = (mat._array[6] * mat._array[8] - mat._array[10] * mat._array[4]) * idet;
    ret._array[5] = (mat._array[0] * mat._array[10] - mat._array[8] * mat._array[2]) * idet;
    ret._array[6] = (mat._array[2] * mat._array[4] - mat._array[6] * mat._array[0]) * idet;
    ret._array[7] = 0.0;
    ret._array[8] = (mat._array[4] * mat._array[9] - mat._array[8] * mat._array[5]) * idet;
    ret._array[9] = (mat._array[1] * mat._array[8] - mat._array[9] * mat._array[0]) * idet;
    ret._array[10] = (mat._array[0] * mat._array[5] - mat._array[4] * mat._array[1]) * idet;
    ret._array[11] = 0.0;
    ret._array[12] = (mat._array[3] * ret._array[0] + mat._array[7] * ret._array[1] + mat._array[11] * ret._array[2]);
    ret._array[13] = (mat._array[3] * ret._array[4] + mat._array[7] * ret._array[5] + mat._array[11] * ret._array[6]);
    ret._array[14] = (mat._array[3] * ret._array[8] + mat._array[7] * ret._array[9] + mat._array[11] * ret._array[10]);
    ret._array[15] = 1.0;
    return ret;
}

template< bool rowMajor >
__global__
void inverseModelView( void *out_p, const void *in_p, size_t n )
{
    const float4x4 *in = static_cast< const float4x4* >( in_p );
    float4x4 *out = static_cast< float4x4* >( out_p );

    // thread identification in grid
    const int gridSize = gridDim.x * blockDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;


    while ( tid < n )
    {
        if ( rowMajor )
            out[ tid ] = transpose( inverse( in[tid] ));
        else
            out[ tid ] = inverse( in[tid] );

        tid += gridSize;
    }
}


//	//	const int couple[] = {
//	// thread identification in matrix
//	const int tidm = threadIdx.x;
//	const int tidmx = tidm & 0xF;
//	const int tidmy = tidm >> 4;
//	const int j = tidmx;
//	const int i = tidmy;
//	// matrix identification in bloc
//	const int midb = threadIdx.y;
//	// thread identification in bloc
//	const int tidb = tidm + midb * blockDim.x; // blockDim.x == 16
//	// matrix identification in grid
//	const int mid = midb + blockIdx.x * blockDim.x;
//	// thread identification in grid
//	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
//	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
//	const int tid = tidx + tidy * gridDim.x * blockDim.x;
//	// map from threadIdx/BlockIdx to pixel position
//	const int moffset = blockIdx.x * blockDim.x * blockDim.y; // blockDim.x == 16
//	const int offset = tidb + moffset; // blockDim.x == 16

//	__shared__ float ins[ MAT_PER_STEP ][ 16 ];
//	__shared__ float sout[ MAT_PER_STEP ][ 16 ];

//	// copy matrices in shared memory
//	if ( midb < MAT_PER_STEP )
//	{
//		if ( rowMajor )
//		{
//			ins[ midb ][ tidm ] = in[ offset ];
//		}
//		else
//		{
//			const int transposeShift[16] = {
//				 0,  3,  6,  9,
//				-3,  0,  3,  6,
//				-6, -6,  0,  3,
//				-9, -6, -3,  0
//			};
//			ins[ midb ][ tidm ] = in[ offset + transposeShift[tidm] ];
//		}
//	}

//	__syncthreads();






/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
__global__
void memcpy_24_to_32( unsigned * dest, const unsigned * src, size_t size );
__global__
void memcpy_24_to_32( uchar4 * dest, const unsigned * src, size_t size );
void SampleCore::draw(uchar4 color, float3 pos)
{
//	CUDA_SAFE_CALL( cudaMemcpyToSymbol(
//						toto_color,
//						&color,
//						sizeof(uchar4),
//						cudaMemcpyHostToDevice ) );

	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT(frame);
	CUDAPM_START_EVENT(app_init_frame);

//	glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer);

//	glClearColor(0.0f, 0.1f, 0.3f, 0.0f);
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);

	// draw the octree where the Tore will be
	glPushMatrix();
	glTranslatef(-0.5f, -0.5f, -0.5f);

	if ( _displayOctree )
	{
		_volumeTree->displayDebugOctree();
	}

	glPopMatrix();

	// copy the current scene into PBO
    glBindBuffer(GL_PIXEL_PACK_BUFFER, _colorBuffer);
	glReadPixels(0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	CUT_CHECK_ERROR_GL();

    glBindBuffer(GL_PIXEL_PACK_BUFFER, _depthBuffer);
	glReadPixels(0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	CUT_CHECK_ERROR_GL();

//	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix._array);
	glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix._array);

	// extract viewport
	GLint params[4];
	glGetIntegerv(GL_VIEWPORT, params);
	int4 viewport = make_int4(params[0], params[1], params[2], params[3]);

	// render the scene into textures
	CUDAPM_STOP_EVENT(app_init_frame);

	static float4x4 matrixArray[ 1 + 1000 ];
	static float4x4 &modelMatrix = matrixArray[0];
//*
	glPushMatrix();
	glLoadIdentity();
	glTranslatef(-0.5f, -0.5f, -0.5f);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelMatrix._array);
	glPopMatrix();
    modelMatrix = transpose( ( modelMatrix ) );
	static int n = 0;
	if ( n == 0 )
	for (int nz = 0 ; nz < 3 && n < 1 ; nz++)
	for (int ny = 0 ; ny < 6 && n < 1 ; ny+=2)
	for (int nx = 0 ; nx < 3 && n < 1 ; nx++)
	{
		n++;

		float4x4 tmp;

		glPushMatrix();
		glLoadIdentity();
        glTranslatef(nx * 1.5f - 0.0f,
					 ny * 0.75f - 2.0f + 0.5f*(ny==0),
                     nz * 1.5f - 0.0f);
		glScalef( (1+ny)*0.25f, (1+ny)*0.25f, (1+ny)*0.25f );
		glRotatef( rand()/(double)RAND_MAX*360, 1.f, 0.f, 0.f );
		glRotatef( rand()/(double)RAND_MAX*360, 0.f, 1.f, 0.f );
		glRotatef( rand()/(double)RAND_MAX*360, 0.f, 0.f, 1.f );
		glTranslatef( -0.50f, -0.50f, -0.50f );
		glGetFloatv(GL_MODELVIEW_MATRIX, tmp._array);
		glPopMatrix();

		matrixArray[n] = transpose( inverse( tmp ));

//		matrixArray[n] = modelMatrix;
//		matrixArray[n]._array[3] += nx * 1.f - 2.f ;
//		matrixArray[n]._array[7] += ny * 1.f - 2.f ;
//		matrixArray[n]._array[11] += nz * 1.f - 2.f ;
	}
    matrixArray[0]._array[0] = float(n);
/*/
	glPushMatrix();
	glLoadIdentity();
//	glTranslatef(-0.5f, -0.5f, -0.5f);
	glTranslatef(-2.5f, -2.5f, -2.5f);
	glScalef(1.f, 2.f, 1.f);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelMatrix._array);
	glPopMatrix();
//	matrixArray[1] = transpose( ( modelMatrix ) );
//*/
		// render
	_volumeTreeRenderer->render(matrixArray[0], viewMatrix, projectionMatrix, viewport);


//		glPushMatrix();
//		glLoadIdentity();
////		glTranslatef(x, y, z);
////		glTranslatef(pos.x, pos.y, pos.z);
//		glTranslatef(x + pos.x, y + pos.y, z + pos.z);
//		glGetFloatv(GL_MODELVIEW_MATRIX, modelMatrix._array);
//		glPopMatrix();

//		// render
//		_volumeTreeRenderer->render(modelMatrix, viewMatrix, projectionMatrix, viewport);


/*/
#if 1 // look at the depth
	cudaGraphicsMapResources( 1, &_depthResource, 0);
	unsigned *ptr;
	size_t tmp_size;
	cudaError_t error = cudaGraphicsResourceGetMappedPointer(
							(void**)&ptr, &tmp_size, _depthResource );
	Gv_GLbuffer oPixelBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,
							 GL_DYNAMIC_DRAW_ARB,
							 4 * _width * _height,
							 cudaGraphicsMapFlagsNone
							 );
	oPixelBuffer.map();
	memcpy_24_to_32 <<< 32, 32 >>> ( oPixelBuffer.getMappedPointer<uchar4>(),
									 ptr,
									 4 * _width * _height );
	oPixelBuffer.unmap();

	cudaGraphicsUnmapResources(1, &_depthResource, 0) ;

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, oPixelBuffer._bufferObj);
	glDrawPixels( _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#else //normal color
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _colorBuffer);
	glDrawPixels( _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#endif

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _depthBuffer);
	glDrawPixels( _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
/*/
	// upload changes into the textures
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _colorBuffer);
    glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _colorTex);
	glTexSubImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, 0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _depthBuffer);
    glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _depthTex);
	glTexSubImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// render the result to the screen
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glEnable(GL_TEXTURE_RECTANGLE_EXT);
	glDisable(GL_DEPTH_TEST);

	glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _colorTex);

	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = _width;
	GLint tMax = _height;

	glBegin(GL_QUADS);
		glColor3f(1.0f, 1.0f, 1.0f);
		glTexCoord2i(sMin, tMin); glVertex2i(-1, -1);
		glTexCoord2i(sMax, tMin); glVertex2i( 1, -1);
		glTexCoord2i(sMax, tMax); glVertex2i( 1,  1);
		glTexCoord2i(sMin, tMax); glVertex2i(-1,  1);
	glEnd();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);

	glDisable(GL_TEXTURE_RECTANGLE_EXT);

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
//*/

	_volumeTreeRenderer->nextFrame();

	CUDAPM_STOP_EVENT(frame);
	CUDAPM_STOP_FRAME;

	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrameGL( _displayPerfmon - 1 );
	}
}



/******************************************************************************
 * Resize the frame
 *
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void SampleCore::resize( int width, int height )
{
	_width = width;
	_height = height;

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(_width, _height));

	// Create frame-dependent objects
    if (_colorResource)
    {
        CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(_colorResource));
    }
    if (_depthResource)
    {
        CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(_depthResource));
    }

    if (_colorBuffer)
    {
        glDeleteBuffers(1, &_colorBuffer);
    }
    if (_depthBuffer)
    {
        glDeleteBuffers(1, &_depthBuffer);
    }

    if (_colorTex)
    {
        glDeleteTextures(1, &_colorTex);
    }
    if (_depthTex)
    {
        glDeleteTextures(1, &_depthTex);
    }

	if (_frameBuffer)
    {
        glDeleteFramebuffers(1, &_frameBuffer);
    }

    glGenBuffers(1, &_colorBuffer);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, _colorBuffer);
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLubyte) * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    CUT_CHECK_ERROR_GL();

    glGenTextures(1, &_colorTex);
    glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _colorTex);
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
    CUT_CHECK_ERROR_GL();

    glGenBuffers(1, &_depthBuffer);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, _depthBuffer);
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    CUT_CHECK_ERROR_GL();

    glGenTextures(1, &_depthTex);
    glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _depthTex);
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL);
    glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
    CUT_CHECK_ERROR_GL();

    glGenFramebuffers(1, &_frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _colorTex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    CUT_CHECK_ERROR_GL();

	// Create CUDA resources from OpenGL objects
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&_colorResource, _colorBuffer, cudaGraphicsRegisterFlagsNone));
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&_depthResource, _depthBuffer, cudaGraphicsRegisterFlagsNone));

	// Pass resources to the renderer
    _volumeTreeRenderer->setColorResource(_colorResource);
    _volumeTreeRenderer->setDepthResource(_depthResource);
//	_volumeTreeRenderer->setIO_GL( GvRenderer::ioColor, _colorBuffer );
//	_volumeTreeRenderer->setIO_GL( GvRenderer::ioDepth, _depthBuffer );
}

/******************************************************************************
 * Clear the GigaVoxels cache
 ******************************************************************************/
void SampleCore::clearCache()
{
	_volumeTreeRenderer->clearCache();
}

/******************************************************************************
 * Toggle the display of the N-tree (octree) of the data structure
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	_displayOctree = !_displayOctree;
}

/******************************************************************************
 * Toggle the GigaVoxels dynamic update mode
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	_volumeTreeRenderer->dynamicUpdateState() = !_volumeTreeRenderer->dynamicUpdateState();
}

/******************************************************************************
 * Toggle the display of the performance monitor utility if
 * GigaVoxels has been compiled with the Performance Monitor option
 *
 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
{
	if ( _displayPerfmon )
	{
		_displayPerfmon = 0;
	}
	else
	{
		_displayPerfmon = mode;
	}
}

/******************************************************************************
 * Increment the max resolution of the data structure
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if ( _maxVolTreeDepth < 32 )
	{
		_maxVolTreeDepth++;
	}

	_volumeTree->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * Decrement the max resolution of the data structure
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if ( _maxVolTreeDepth > 0 )
	{
		_maxVolTreeDepth--;
	}

	_volumeTree->setMaxDepth( _maxVolTreeDepth );
}








