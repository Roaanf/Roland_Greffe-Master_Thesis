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
#include "GvCore/GsError.h"

// Cuda SDK
#include <helper_cuda.h>

using namespace GvRendering;
/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/


/******************************************************************************
 * Constructor
 *
 * @param pVolumeTree data structure to render
 * @param pVolumeTreeCache cache
 * @param pProducer producer of data
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::RendererCUDA( TVolumeTreeType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache )
:	GvRendering::GsRenderer< TVolumeTreeType, TVolumeTreeCacheType >( pVolumeTree, pVolumeTreeCache )
,	_graphicsInteroperabiltyHandler( NULL )
{
	GV_CHECK_CUDA_ERROR( "GsRendererCUDA::GsRendererCUDA prestart" );

	// Init frame size
	_frameSize = make_uint2( 0, 0 );

	// Init frame dependant buffers//
	// Deferred lighting infos
	// ...

	_numUpdateFrames		= 1;
	_frameNumAfterUpdate	= 0;

	_fastBuildMode			= true;

	// Do CUDA initialization
	this->initializeCuda();

	// Initialize graphics interoperability
	_graphicsInteroperabiltyHandler = new GvRendering::GsGraphicsInteroperabiltyHandler();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::~RendererCUDA()
{
	// Destroy internal buffers used during rendering
	deleteFrameObjects();

	// Finalize graphics interoperability
	delete _graphicsInteroperabiltyHandler;
}

/******************************************************************************
 * Get the graphics interoperability handler
 *
 * @return the graphics interoperability handler
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
GvRendering::GsGraphicsInteroperabiltyHandler* RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getGraphicsInteroperabiltyHandler()
{
	return _graphicsInteroperabiltyHandler;
}

/******************************************************************************
 * Initialize Cuda objects
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::initializeCuda()
{
	GV_CHECK_CUDA_ERROR( "VoxelSceneRenderer::cuda_Init pre-start" );

	// Retrieve device properties
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties( &deviceProps, gpuGetMaxGflopsDeviceId() );	// TO DO : handle the case where user could want an other device
	std::cout << "\nDevice properties" << std::endl;
	std::cout << "- name : " << deviceProps.name << std::endl;
	std::cout << "- compute capability : "<< deviceProps.major << "." << deviceProps.minor << std::endl;
	std::cout << "- compute mode : " << deviceProps.computeMode << std::endl;
	std::cout << "- can map host memory : " << deviceProps.canMapHostMemory << std::endl;
	std::cout << "- can overlap transfers and kernels : " << deviceProps.deviceOverlap << std::endl;
	std::cout << "- kernels timeout : " << deviceProps.kernelExecTimeoutEnabled << std::endl;
	std::cout << "- integrated chip : " << deviceProps.integrated << std::endl;
	std::cout << "- global memory : " << deviceProps.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
	std::cout << "- shared memory : " << deviceProps.sharedMemPerBlock / 1024 << "KB" << std::endl;
	std::cout << "- clock rate : " << deviceProps.clockRate / 1000 << "MHz" << std::endl;
	
	GV_CHECK_CUDA_ERROR( "VoxelSceneRenderer::cuda_Init start" );

	GV_CHECK_CUDA_ERROR( "VoxelSceneRenderer::cuda_Init end" );
}

/******************************************************************************
 * Finalize Cuda objects
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::finalizeCuda()
{
}

/******************************************************************************
 * Initialize internal buffers used during rendering
 * (i.e. input/ouput color and depth buffers, ray buffers, etc...).
 * Buffers size are dependent of the frame size.
 *
 * @param pFrameSize the frame size
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::initFrameObjects( const uint2& pFrameSize )
{
	// Check if frame size has been modified
	if ( _frameSize.x != pFrameSize.x || _frameSize.y != pFrameSize.y )
	{
		_frameSize = pFrameSize;

		// Destruct frame based objects
		deleteFrameObjects();
	}

	GV_CHECK_CUDA_ERROR( "VoxelSceneRenderer::initFrameObjects" );
}

/******************************************************************************
 * Destroy internal buffers used during rendering
 * (i.e. input/ouput color and depth buffers, ray buffers, etc...)
 * Buffers size are dependent of the frame size.
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::deleteFrameObjects()
{
	// Destroy input/output color and depth buffers
	// ...
}

/******************************************************************************
 * Attach an OpenGL buffer object (i.e. a PBO, a VBO, etc...) to an internal graphics resource 
 * that will be mapped to a color or depth slot used during rendering.
 *
 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
 * @param pBuffer the OpenGL buffer
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::connect( GvRendering::GsGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot, GLuint pBuffer )
{
	return _graphicsInteroperabiltyHandler->connect( pGraphicsResourceSlot, pBuffer );
}

/******************************************************************************
 * Attach an OpenGL texture or renderbuffer object to an internal graphics resource 
 * that will be mapped to a color or depth slot used during rendering.
 *
 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
 * @param pImage the OpenGL texture or renderbuffer object
 * @param pTarget the target of the OpenGL texture or renderbuffer object
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::connect( GvRendering::GsGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot, GLuint pImage, GLenum pTarget )
{
	return _graphicsInteroperabiltyHandler->connect( pGraphicsResourceSlot, pImage, pTarget );
}

/******************************************************************************
 * Dettach an OpenGL buffer object (i.e. a PBO, a VBO, etc...), texture or renderbuffer object
 * to its associated internal graphics resource mapped to a color or depth slot used during rendering.
 *
 * @param pGraphicsResourceSlot the internal graphics resource slot (color or depth)
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::disconnect( GvRendering::GsGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot )
{
	return _graphicsInteroperabiltyHandler->disconnect( pGraphicsResourceSlot );
}

/******************************************************************************
 * Disconnect all registered graphics resources
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::resetGraphicsResources()
{
	return _graphicsInteroperabiltyHandler->reset();
}

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GsIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::render( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Start rendering
	doRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
}

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GsIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::preRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Initialize frame objects
	uint2 frameSize = make_uint2( pViewport.z - pViewport.x, pViewport.w - pViewport.y );
	initFrameObjects( frameSize );

	// NOTE
	// mapResources()
	// This function provides the synchronization guarantee that any graphics calls issued before cudaGraphicsMapResources() will complete before any subsequent CUDA work in stream begins.

	// Map graphics resources
	CUDAPM_START_EVENT( vsrender_pre_frame_mapbuffers );
	_graphicsInteroperabiltyHandler->mapResources();
	bindGraphicsResources();
	CUDAPM_STOP_EVENT( vsrender_pre_frame_mapbuffers );

	// TEST : put here beacuse the "call" blocks
	_graphicsInteroperabiltyHandler->setRendererContextInfo( viewContext );	// TO DO : this call blocks
}

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GsIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::postRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Unmap graphics resources
	CUDAPM_START_EVENT( vsrender_post_frame_unmapbuffers );
	unbindGraphicsResources();
	_graphicsInteroperabiltyHandler->unmapResources();
	CUDAPM_STOP_EVENT( vsrender_post_frame_unmapbuffers );
}

/******************************************************************************
 * Start the rendering process.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::doRender( const float4x4& modelMatrix, const float4x4& viewMatrix, const float4x4& projMatrix, const int4& pViewport )
{
	// Extract zNear, zFar as well as the distance in view space
	// from the center of the screen to each side of the screen.
	float fleft   = projMatrix._array[ 14 ] * ( projMatrix._array[ 8 ] - 1.0f ) / ( projMatrix._array[ 0 ] * ( projMatrix._array[ 10 ] - 1.0f ) );
	float fright  = projMatrix._array[ 14 ] * ( projMatrix._array[ 8 ] + 1.0f ) / ( projMatrix._array[ 0 ] * ( projMatrix._array[ 10 ] - 1.0f ) );
	float ftop    = projMatrix._array[ 14 ] * ( projMatrix._array[ 9 ] + 1.0f ) / ( projMatrix._array[ 5 ] * ( projMatrix._array[ 10 ] - 1.0f ) );
	float fbottom = projMatrix._array[ 14 ] * ( projMatrix._array[ 9 ] - 1.0f ) / ( projMatrix._array[ 5 ] * ( projMatrix._array[ 10 ] - 1.0f ) );
	float fnear   = projMatrix._array[ 14 ] / ( projMatrix._array[ 10 ] - 1.0f );
	float ffar    = projMatrix._array[ 14 ] / ( projMatrix._array[ 10 ] + 1.0f );
	
	float2 viewSurfaceVS[ 2 ];
	viewSurfaceVS[ 0 ] = make_float2( fleft, fbottom );
	viewSurfaceVS[ 1 ] = make_float2( fright, ftop );

	float3 viewPlane[ 2 ];
	viewPlane[ 0 ] = make_float3( fleft, fbottom, fnear );
	viewPlane[ 1 ] = make_float3( fright, ftop, fnear );
	// float3 viewSize = ( viewPlane[ 1 ] - viewPlane[ 0 ] );

	// Projected 2D Bounding Box of the GigaVoxels 3D BBox.
	// It holds its bottom left corner and its size.
	// ( bottomLeftCorner.x, bottomLeftCorner.y, frameSize.x, frameSize.y )
	viewContext._projectedBBox = this->_projectedBBox;
	
	float2 viewSurfaceVS_Size = viewSurfaceVS[ 1 ] - viewSurfaceVS[ 0 ];
	
	// Transform matrices
	float4x4 invModelMatrixT = transpose( inverse( modelMatrix ) );
	float4x4 invViewMatrixT = transpose( inverse( viewMatrix ) );

	float4x4 modelMatrixT = transpose( modelMatrix );
	float4x4 viewMatrixT = transpose( viewMatrix );

	//float4x4 viewMatrix=(inverse(invViewMatrix));

	viewContext.invViewMatrix = invViewMatrixT;
	viewContext.viewMatrix = viewMatrixT;
	viewContext.invModelMatrix = invModelMatrixT;
	viewContext.modelMatrix = modelMatrixT;

	// Store frustum parameters
	viewContext.frustumNear = fnear;
	viewContext.frustumNearINV = 1.0f / fnear;
	viewContext.frustumFar = ffar;
	viewContext.frustumRight = fright;
	viewContext.frustumTop = ftop;
	viewContext.frustumC = projMatrix._array[ 10 ]; // - ( ffar + fnear ) / ( ffar - fnear );
	viewContext.frustumD = projMatrix._array[ 14 ]; // ( -2.0f * ffar * fnear ) / ( ffar - fnear );

	// Graphics resource settings
	viewContext._clearColor = this->_clearColor;
	viewContext._clearDepth = this->_clearDepth;
	
	// WORLD
	float3 viewPlanePosWP = mul( viewContext.invViewMatrix, make_float3( fleft, fbottom, -fnear ) );
	viewContext.viewCenterWP = mul( viewContext.invViewMatrix, make_float3( 0.0f, 0.0f, 0.0f ) );
	viewContext.viewPlaneDirWP = viewPlanePosWP - viewContext.viewCenterWP;
	// TREE
	float3 viewPlanePosTP = mul( viewContext.invModelMatrix, viewPlanePosWP );
	viewContext.viewCenterTP = mul( viewContext.invModelMatrix, viewContext.viewCenterWP );
	viewContext.viewPlaneDirTP = viewPlanePosTP - viewContext.viewCenterTP;

	// Resolution dependant stuff
	viewContext.frameSize = _frameSize;
	float2 pixelSize = viewSurfaceVS_Size / make_float2( static_cast< float >( viewContext.frameSize.x ), static_cast< float >( viewContext.frameSize.y ) );
	viewContext.pixelSize = pixelSize;
	/*viewContext.viewPlaneXAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fright, fbottom, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.x );
	viewContext.viewPlaneYAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fleft, ftop, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.y );*/
	// WORLD
	viewContext.viewPlaneXAxisWP = mul( viewContext.invViewMatrix, make_float3( fright, fbottom, -fnear ) );
	viewContext.viewPlaneYAxisWP = mul( viewContext.invViewMatrix, make_float3( fleft, ftop, -fnear ) );
	// TREE
	viewContext.viewPlaneXAxisTP = mul( viewContext.invModelMatrix, viewContext.viewPlaneXAxisWP );
	viewContext.viewPlaneYAxisTP = mul( viewContext.invModelMatrix, viewContext.viewPlaneYAxisWP );
	// WORLD
	viewContext.viewPlaneXAxisWP = ( viewContext.viewPlaneXAxisWP - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.x );
	viewContext.viewPlaneYAxisWP = ( viewContext.viewPlaneYAxisWP - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.y );
	// TREE
	viewContext.viewPlaneXAxisTP = ( viewContext.viewPlaneXAxisTP - viewPlanePosTP ) / static_cast< float >( viewContext.frameSize.x );
	viewContext.viewPlaneYAxisTP = ( viewContext.viewPlaneYAxisTP - viewPlanePosTP ) / static_cast< float >( viewContext.frameSize.y );

	CUDAPM_START_EVENT( vsrender_copyconsts_frame );

	uint maxVolTreeDepth = this->_volumeTree->getMaxDepth();

	// TEST, OPTIM : early preRender pass
	//this->_volumeTreeCache->preRenderPass();

	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbolAsync( k_renderViewContext, &viewContext, sizeof( viewContext ) ) );
	
	// TO DO : optimization
	// - place this value in the big k_renderViewContext structure, to avoid launching a copy.
	// - Usually, graphics cards have only one copy engine, copies are serialized ? => not sure, check this point.
	//
	// - at least, move this call at beginning to be able to overlap cpu computation to fill k_renderViewContext
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbolAsync( k_currentTime, (&this->_currentTime), sizeof( this->_currentTime ), 0, cudaMemcpyHostToDevice) );

	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_maxVolTreeDepth, &maxVolTreeDepth, sizeof( maxVolTreeDepth ), 0, cudaMemcpyHostToDevice) );
	
		// TO DO : move this "Performance monitor" piece of code in another place
	#ifdef USE_CUDAPERFMON
		// Performance monitor
		if ( GvPerfMon::CUDAPerfMon::get()._requestResize ) 
		{
			// Update device memory
			GvCore::GsLinearMemory< GvCore::uint64 >* my_d_timersArray = GvPerfMon::CUDAPerfMon::get().getKernelTimerArray();
			GvCore::GsLinearMemoryKernel< GvCore::uint64 > h_timersArray = my_d_timersArray->getDeviceArray();
			GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvPerfMon::k_timersArray, &h_timersArray, sizeof( h_timersArray ), 0, cudaMemcpyHostToDevice ) );
		
			// Update device memory
			uchar* my_d_timersMask = GvPerfMon::CUDAPerfMon::get().getKernelTimerMask();
			GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvPerfMon::k_timersMask, &my_d_timersMask, sizeof( my_d_timersMask ), 0, cudaMemcpyHostToDevice ) );

			// Update the performnace monitor's state
			GvPerfMon::CUDAPerfMon::get()._requestResize = false;
		}
	#endif
	
	CUDAPM_STOP_EVENT( vsrender_copyconsts_frame );

		// TO DO : optimization
	//
	// - this variable seems to beun-uned anymore => remove it
	//
	// - furthermore, this value should not change every frame => remove that and call cudaMemcpyToSymbol only when required
	//
	// - at least, move this call at beginning to be able to overlap cpu computation to fill k_renderViewContext
	//float voxelSizeMultiplier = 1.0f;
	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_voxelSizeMultiplier, (&voxelSizeMultiplier), sizeof( voxelSizeMultiplier ), 0, cudaMemcpyHostToDevice ) );
	
	CUDAPM_START_EVENT_GPU( gv_rendering );

	dim3 blockSize( RenderBlockResolution::x, RenderBlockResolution::y, 1 );
	dim3 gridSize( iDivUp( _frameSize.x, RenderBlockResolution::x ), iDivUp( _frameSize.y, RenderBlockResolution::y ), 1 );
	// FUTUR optimization
	//
	//dim3 gridSize( iDivUp( /*projectedBBoxSize*/_projectedBBox.z, RenderBlockResolution::x ), iDivUp( /*projectedBBoxSize*/_projectedBBox.w, RenderBlockResolution::y ), 1 );
	
	uint cpuDescentCount = 0;
	uint cpuSkippedCount = 0;
	uint cpuSkippedNb = 0;

#ifdef PROXY_PRINTOUT
	cudaMemcpyToSymbol( nbPixels, &cpuDescentCount, sizeof( uint ) );
	cudaMemcpyToSymbol( totalSkipped, &cpuSkippedCount, sizeof( uint ) );
	cudaMemcpyToSymbol( nbSkippedNodes, &cpuSkippedNb, sizeof( uint ) );
#endif
	if ( this->_dynamicUpdate )
	{
		if ( this->_hasPriorityOnBricks )
		{
			// Priority on brick is set to TRUE to force loading data at low resolution first
			RenderKernelSimple< RenderBlockResolution, false, true, TSampleShader >
							<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject() );
		}
		else
		{
			RenderKernelSimple< RenderBlockResolution, false, false, TSampleShader >
							<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject() );
		}
	}
	else
	{
		RenderKernelSimple< RenderBlockResolution, false, false, TSampleShader >
						<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject() );
	}

	GV_CHECK_CUDA_ERROR( "RenderKernelSimple" );

	CUDAPM_STOP_EVENT_GPU( gv_rendering );
	
	CUDAPM_RENDER_CACHE_INFO( 256, 512 );

	
	uint nbDescentCPU;
	uint nbSkippedDescentsCPU;
	uint nbSkippedCPU;

#ifdef PROXY_PRINTOUT
	cudaMemcpyFromSymbol( &nbDescentCPU, nbPixels, sizeof( uint ) );
	cudaMemcpyFromSymbol( &nbSkippedDescentsCPU, totalSkipped, sizeof( uint ) );
	cudaMemcpyFromSymbol( &nbSkippedCPU, nbSkippedNodes, sizeof( uint ) );
		
	GV_CHECK_CUDA_ERROR( "cudamemcopy jeremy" );

	printf("%u descent %u skipped ; ratio of skipped descents : %f ; average length of skipped branches : %f\n",nbDescentCPU,nbSkippedDescentsCPU,(float)(nbSkippedDescentsCPU*100)/(float)nbDescentCPU , (float)nbSkippedCPU/(float)nbSkippedDescentsCPU);
#endif
	/*{
		uint2 poolRes = make_uint2(180, 150);
		uint2 poolScale = make_uint2(2, 2);

		dim3 blockSize(10, 10, 1);
		dim3 gridSize(poolRes.x * poolScale.x / blockSize.x, poolRes.y * poolScale.y / blockSize.y, 1);
		RenderDebug<<<gridSize, blockSize, 0>>>(d_outFrameColor->getDeviceArray(), poolRes, poolScale);
	}*/
}

/******************************************************************************
 * Bind all graphics resources used by the GL interop handler during rendering.
 *
 * Internally, it binds textures and surfaces to arrays associated to mapped graphics reources.
 *
 * NOTE : this method should be in the GsGraphicsInteroperabiltyHandler but it seems that
 * there are conflicts with textures ans surfaces symbols. The binding succeeds but not the
 * read/write operations.
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::bindGraphicsResources()
{
	// Iterate through graphics resources info
	std::vector< std::pair< GvRendering::GsGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvRendering::GsGraphicsResource* > >& graphicsResources = _graphicsInteroperabiltyHandler->editGraphicsResources();
	for ( int i = 0; i < graphicsResources.size(); i++ )
	{
		// Get current graphics resources info
		std::pair< GvRendering::GsGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvRendering::GsGraphicsResource* >& graphicsResourceInfo = graphicsResources[ i ];
		GvRendering::GsGraphicsInteroperabiltyHandler::GraphicsResourceSlot graphicsResourceSlot = graphicsResourceInfo.first;
		GvRendering::GsGraphicsResource* graphicsResource  = graphicsResourceInfo.second;
		assert( graphicsResource != NULL );

		// [ 2 ] - Bind array to texture or surface if needed
		if ( graphicsResource->getMemoryType() == GsGraphicsResource::eCudaArray )
		{
			struct cudaArray* imageArray = static_cast< struct cudaArray* >( graphicsResource->getMappedAddress() );

			cudaError_t error;
			switch ( graphicsResourceSlot )
			{
			case GvRendering::GsGraphicsInteroperabiltyHandler::eColorReadSlot:
					error = cudaBindTextureToArray( GvRendering::_inputColorTexture, imageArray );
					break;

				case GvRendering::GsGraphicsInteroperabiltyHandler::eColorWriteSlot:
				case GvRendering::GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot:
					error = cudaBindSurfaceToArray( GvRendering::_colorSurface, imageArray );
					break;

				case GvRendering::GsGraphicsInteroperabiltyHandler::eDepthReadSlot:
					error = cudaBindTextureToArray( GvRendering::_inputDepthTexture, imageArray );
					break;

				case GvRendering::GsGraphicsInteroperabiltyHandler::eDepthWriteSlot:
				case GvRendering::GsGraphicsInteroperabiltyHandler::eDepthReadWriteSlot:
					error = cudaBindSurfaceToArray( GvRendering::_depthSurface, imageArray );
					break;

				default:
					assert( false );
					break;
			}
		}
	}

	return false;
}

/******************************************************************************
 * Unbind all graphics resources used by the GL interop handler during rendering.
 *
 * Internally, it unbinds textures and surfaces to arrays associated to mapped graphics reources.
 *
 * NOTE : this method should be in the GsGraphicsInteroperabiltyHandler but it seems that
 * there are conflicts with textures ans surfaces symbols. The binding succeeds but not the
 * read/write operations.
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool RendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::unbindGraphicsResources()
{
	// Iterate through graphics resources info
	std::vector< std::pair< GvRendering::GsGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvRendering::GsGraphicsResource* > >& graphicsResources = _graphicsInteroperabiltyHandler->editGraphicsResources();
	for ( int i = 0; i < graphicsResources.size(); i++ )
	{
		// Get current graphics resources info
		std::pair< GvRendering::GsGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvRendering::GsGraphicsResource* >& graphicsResourceInfo = graphicsResources[ i ];
		GvRendering::GsGraphicsInteroperabiltyHandler::GraphicsResourceSlot graphicsResourceSlot = graphicsResourceInfo.first;
		GvRendering::GsGraphicsResource* graphicsResource  = graphicsResourceInfo.second;
		assert( graphicsResource != NULL );

		// [ 2 ] - Bind array to texture or surface if needed
		if ( graphicsResource->getMemoryType() == GvRendering::GsGraphicsResource::eCudaArray )
		{
			struct cudaArray* imageArray = static_cast< struct cudaArray* >( graphicsResource->getMappedAddress() );

			cudaError_t error;
			switch ( graphicsResourceSlot )
			{
			case GvRendering::GsGraphicsInteroperabiltyHandler::eColorReadSlot:
					error = cudaUnbindTexture( GvRendering::_inputColorTexture );
					break;

				case GvRendering::GsGraphicsInteroperabiltyHandler::eColorWriteSlot:
				case GvRendering::GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot:
					// There is no "unbind surface" function in CUDA
					break;

				case GvRendering::GsGraphicsInteroperabiltyHandler::eDepthReadSlot:
					error = cudaUnbindTexture( GvRendering::_inputDepthTexture );
					break;

				case GvRendering::GsGraphicsInteroperabiltyHandler::eDepthWriteSlot:
				case GvRendering::GsGraphicsInteroperabiltyHandler::eDepthReadWriteSlot:
					// There is no "unbind surface" function in CUDA
					break;

				default:
					assert( false );
					break;
			}
		}
	}

	return false;
}

