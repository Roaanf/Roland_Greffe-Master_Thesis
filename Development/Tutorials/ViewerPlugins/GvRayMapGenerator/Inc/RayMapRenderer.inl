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
#include <GvCore/GvError.h>
#include <GvUtils/GvShaderManager.h>
#include <GvRendering/GvGraphicsResource.h>

// Cuda SDK
#include <helper_cuda.h>

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
RayMapRenderer< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::RayMapRenderer( TVolumeTreeType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache )
:	GvRendering::GvRendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >( pVolumeTree, pVolumeTreeCache )
,	_rayMapResource( NULL )
{
	// Configure texture samplers
	rayMapTexture.normalized = 0;
	rayMapTexture.addressMode[0] = cudaAddressModeClamp;
	rayMapTexture.addressMode[1] = cudaAddressModeClamp;
	rayMapTexture.addressMode[2] = cudaAddressModeClamp;
	rayMapTexture.filterMode = cudaFilterModePoint;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
RayMapRenderer< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::~RayMapRenderer()
{
	//...
}

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GvIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RayMapRenderer< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::render( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Call internal render method
	//doRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );

	//	assert( _colorResource != NULL && _depthResource != NULL ); // && "You must set the input buffers first");

	// Initialize frame objects
	uint2 frameSize = make_uint2( pViewport.z - pViewport.x, pViewport.w - pViewport.y );
	this->initFrameObjects( frameSize );

	// Map graphics resources
	CUDAPM_START_EVENT( vsrender_pre_frame_mapbuffers );
	this->_graphicsInteroperabiltyHandler->mapResources();
	this->bindGraphicsResources();
	cudaArray* rayMapArray;
	//GV_CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &_rayMapResource, 0 ) );
	_rayMapResource->map();
	//GV_CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( &rayMapArray, _rayMapResource, 0, 0 ) );
	rayMapArray = static_cast< cudaArray* >( _rayMapResource->getMappedAddress() );
	GV_CUDA_SAFE_CALL( cudaBindTextureToArray( rayMapTexture, rayMapArray ) );
	CUDAPM_STOP_EVENT( vsrender_pre_frame_mapbuffers );

	// Start rendering
	doRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );

	// Unmap graphics resources
	CUDAPM_START_EVENT( vsrender_post_frame_unmapbuffers );
	this->unbindGraphicsResources();
	this->_graphicsInteroperabiltyHandler->unmapResources();
	GV_CUDA_SAFE_CALL( cudaUnbindTexture( rayMapTexture ) );
	//GV_CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &_rayMapResource, 0 ) );
	_rayMapResource->unmap();
	CUDAPM_STOP_EVENT( vsrender_post_frame_unmapbuffers );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RayMapRenderer< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::setRayMapResource( GvRendering::GvGraphicsResource* pGraphicsResource )
{
	_rayMapResource = pGraphicsResource;
}

/******************************************************************************
 * Start the rendering process.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void RayMapRenderer< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::doRender( const float4x4& modelMatrix, const float4x4& viewMatrix, const float4x4& projMatrix, const int4& pViewport )
{
	// Create a render view context to access to useful variables during (view matrix, model matrix, etc...)
	GvRendering::GvRendererContext viewContext;

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
	/*bindGraphicsResources();*/
	this->_graphicsInteroperabiltyHandler->setRendererContextInfo( viewContext );
	// TO DO : add texture offsets !!!!!!!!!!!!
	// ...
	
	// WORLD
	float3 viewPlanePosWP = mul( viewContext.invViewMatrix, make_float3( fleft, fbottom, -fnear ) );
	viewContext.viewCenterWP = mul( viewContext.invViewMatrix, make_float3( 0.0f, 0.0f, 0.0f ) );
	viewContext.viewPlaneDirWP = viewPlanePosWP - viewContext.viewCenterWP;
	// TREE
	float3 viewPlanePosTP = mul( viewContext.invModelMatrix, viewPlanePosWP );
	viewContext.viewCenterTP = mul( viewContext.invModelMatrix, viewContext.viewCenterWP );
	viewContext.viewPlaneDirTP = viewPlanePosTP - viewContext.viewCenterTP;

	// Resolution dependant stuff
	viewContext.frameSize = this->_frameSize;
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

		int maxVolTreeDepth = this->_volumeTree->getMaxDepth();

		// TEST, OPTIM : early preRender pass
		//this->_volumeTreeCache->preRenderPass();

		GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_renderViewContext, &viewContext, sizeof( viewContext ) ) );
		GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_currentTime, (&this->_currentTime), sizeof( this->_currentTime ), 0, cudaMemcpyHostToDevice) );
		GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_maxVolTreeDepth, &maxVolTreeDepth, sizeof( maxVolTreeDepth ), 0, cudaMemcpyHostToDevice) );
	
		// TO DO : move this "Performance monitor" piece of code in another place
	#ifdef USE_CUDAPERFMON
		// Performance monitor
		if ( GvPerfMon::CUDAPerfMon::getApplicationPerfMon()._requestResize ) 
		{
			// Update device memory
			GvCore::Array3DGPULinear< GvCore::uint64 >* my_d_timersArray = GvPerfMon::CUDAPerfMon::getApplicationPerfMon().getKernelTimerArray();
			GvCore::Array3DKernelLinear< GvCore::uint64 > h_timersArray = my_d_timersArray->getDeviceArray();
			GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvPerfMon::k_timersArray, &h_timersArray, sizeof( h_timersArray ), 0, cudaMemcpyHostToDevice ) );
		
			// Update device memory
			uchar* my_d_timersMask = GvPerfMon::CUDAPerfMon::getApplicationPerfMon().getKernelTimerMask();
			GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvPerfMon::k_timersMask, &my_d_timersMask, sizeof( my_d_timersMask ), 0, cudaMemcpyHostToDevice ) );

			// Update the performnace monitor's state
			GvPerfMon::CUDAPerfMon::getApplicationPerfMon()._requestResize = false;
		}
	#endif
	
	CUDAPM_STOP_EVENT( vsrender_copyconsts_frame );

	float voxelSizeMultiplier = 1.0f;
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvRendering::k_voxelSizeMultiplier, (&voxelSizeMultiplier), sizeof( voxelSizeMultiplier ), 0, cudaMemcpyHostToDevice ) );
	
	CUDAPM_START_EVENT_GPU( vsrender_gigavoxels );

	dim3 blockSize( RenderBlockResolution::x, RenderBlockResolution::y, 1 );
	dim3 gridSize( iDivUp( this->_frameSize.x, RenderBlockResolution::x ), iDivUp( this->_frameSize.y, RenderBlockResolution::y ), 1 );
	// FUTUR optimization
	//
	//dim3 gridSize( iDivUp( /*projectedBBoxSize*/_projectedBBox.z, RenderBlockResolution::x ), iDivUp( /*projectedBBoxSize*/_projectedBBox.w, RenderBlockResolution::y ), 1 );
	
	if ( this->_dynamicUpdate )
	{
		if ( this->_hasPriorityOnBricks )
		{
			// Priority on brick is set to TRUE to force loading data at low resolution first
			RayMapRenderKernel< RenderBlockResolution, false, true, TSampleShader >
							<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject() );
		}
		else
		{
			RayMapRenderKernel< RenderBlockResolution, false, false, TSampleShader >
							<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject() );
		}
	}
	else
	{
		RayMapRenderKernel< RenderBlockResolution, false, false, TSampleShader >
						<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject() );
	}

	GV_CHECK_CUDA_ERROR( "RenderKernelSimple" );

	CUDAPM_STOP_EVENT_GPU( vsrender_gigavoxels );
	
	CUDAPM_RENDER_CACHE_INFO( 256, 512 );

	/*{
		uint2 poolRes = make_uint2(180, 150);
		uint2 poolScale = make_uint2(2, 2);

		dim3 blockSize(10, 10, 1);
		dim3 gridSize(poolRes.x * poolScale.x / blockSize.x, poolRes.y * poolScale.y / blockSize.y, 1);
		RenderDebug<<<gridSize, blockSize, 0>>>(d_outFrameColor->getDeviceArray(), poolRes, poolScale);
	}*/
}
