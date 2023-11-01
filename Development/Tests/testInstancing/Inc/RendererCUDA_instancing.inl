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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvRenderer
{

/******************************************************************************
 * Constructor
 *
 * @param volumeTree ...
 * @param gpuProd ...
 * @param nodePoolRes ...
 * @param brickPoolRes ...
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::VolumeTreeRendererCUDA_instancing( VolumeTreeType* volumeTree, VolumeTreeCacheType* volumeTreeCache, ProducerType* gpuProd )
:	VolumeTreeRenderer< VolumeTreeType, VolumeTreeCacheType, ProducerType >( volumeTree, volumeTreeCache, gpuProd )
{
	CUT_CHECK_ERROR("VolumeTreeRendererCUDA_instancing::VolumeTreeRendererCUDA_instancing prestart");

	_frameSize = make_uint2( 0, 0 );

	//Init frame dependant buffers//
	//Deferred lighting infos
	d_inFrameColor			= 0;
	d_inFrameDepth			= 0;
	d_outFrameColor			= 0;
	d_outFrameDepth			= 0;

	currentDebugRay			= make_int2(-1, -1);

	d_rayBufferT			= NULL;
	d_rayBufferTmax			= NULL;
	d_rayBufferAccCol		= NULL;

	numUpdateFrames			= 1;
	frameNumAfterUpdate		= 0;

	fastBuildMode			= true;

    _colorResource			= NULL;
    _depthResource			= NULL;

	this->cuda_Init();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::~VolumeTreeRendererCUDA_instancing()
{
	// Frame dependent buffers
	deleteFrameObjects();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
void VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::cuda_Init()
{
	CUT_CHECK_ERROR("VoxelSceneRenderer::cuda_Init pre-start");

	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps, cutGetMaxGflopsDeviceId());	// TO DO : handle the case where user could want an other device
	std::cout<<"\n******Device properties******\n";
	std::cout<<"Name: "<<deviceProps.name<<"\n";
	std::cout<<"Compute capability: "<<deviceProps.major<<"."<<deviceProps.minor<<"\n";
	std::cout<<"Compute mode: "<<deviceProps.computeMode<<"\n";
	std::cout<<"Can map host memory: "<<deviceProps.canMapHostMemory<<"\n";
	std::cout<<"Can overlap transfers and kernels: "<<deviceProps.deviceOverlap<<"\n";
	std::cout<<"Kernels timeout: "<<deviceProps.kernelExecTimeoutEnabled<<"\n";
	std::cout<<"Integrated chip: "<<deviceProps.integrated<<"\n";
	std::cout<<"Global memory: "<<deviceProps.totalGlobalMem/1024/1024<<"MB\n";
	std::cout<<"Shared memory: "<<deviceProps.sharedMemPerBlock/1024<<"KB\n";
	std::cout<<"Clock rate: "<<deviceProps.clockRate/1000<<"MHz\n";
	std::cout<<"*****************************\n\n";

	CUT_CHECK_ERROR("VoxelSceneRenderer::cuda_Init start");

	cudaStreamCreate(&cudaStream[0]);

	CUT_CHECK_ERROR("VoxelSceneRenderer::cuda_Init end");
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
void VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::cuda_Destroy()
{
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
void VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::deleteFrameObjects()
{

	if(d_inFrameColor)
		delete d_inFrameColor;
	if(d_inFrameDepth)
		delete d_inFrameDepth;
	if(d_outFrameColor)
		delete d_outFrameColor;
	if(d_outFrameDepth)
		delete d_outFrameDepth;

	if(d_rayBufferT)
		delete d_rayBufferT;
	if(d_rayBufferTmax)
		delete d_rayBufferTmax;
	if(d_rayBufferAccCol)
		delete d_rayBufferAccCol;
}

/******************************************************************************
 * ...
 *
 * @param fs ...
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
void VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::initFrameObjects( const uint2& fs )
{
	if (_frameSize.x != fs.x || _frameSize.y != fs.y)
	{
		_frameSize=fs;

		// Destruct frame based objects
		deleteFrameObjects();

		d_inFrameColor			= new GvCore::Array3DGPULinear<uchar4>(NULL, dim3(_frameSize.x, _frameSize.y, 1));
		d_inFrameDepth			= new GvCore::Array3DGPULinear<float>(NULL, dim3(_frameSize.x, _frameSize.y, 1));
		d_outFrameColor			= new GvCore::Array3DGPULinear<uchar4>(NULL, dim3(_frameSize.x, _frameSize.y, 1));
		d_outFrameDepth			= new GvCore::Array3DGPULinear<float>(NULL, dim3(_frameSize.x, _frameSize.y, 1));

		// Ray buffers
		uint2 frameResWithBlock=
				make_uint2(		iDivUp(_frameSize.x, RenderBlockResolution::x)*RenderBlockResolution::x,
								iDivUp(_frameSize.x, RenderBlockResolution::y)*RenderBlockResolution::y );
		d_rayBufferT			= new GvCore::Array3DGPULinear<float>(make_uint3(frameResWithBlock.x, frameResWithBlock.y, 1));
		d_rayBufferTmax			= new GvCore::Array3DGPULinear<float>(make_uint3(frameResWithBlock.x, frameResWithBlock.y, 1));
		d_rayBufferAccCol		= new GvCore::Array3DGPULinear<float4>(make_uint3(frameResWithBlock.x, frameResWithBlock.y, 1));
	}

	CUT_CHECK_ERROR("VoxelSceneRenderer::initFrameObjects");
}

/******************************************************************************
 * ...
 *
 * @param modelMatrix ...
 * @param viewMatrix ...
 * @param projectionMatrix ...
 * @param viewport ...
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
void VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::renderImpl(const float4x4 &modelMatrix, const float4x4 &viewMatrix, const float4x4 &projectionMatrix, const int4 &viewport)
{
    assert(_colorResource != NULL && _depthResource != NULL);// && "You must set the input buffers first");

	uint2 frameSize = make_uint2(viewport.z - viewport.x, viewport.w - viewport.y);
	initFrameObjects(frameSize);

	size_t bufferSize;

	CUDAPM_START_EVENT(vsrender_pre_frame_mapbuffers);

    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_colorResource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)d_inFrameColor->getDataStoragePtrAddress(), &bufferSize, _colorResource));

    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_depthResource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)d_inFrameDepth->getDataStoragePtrAddress(), &bufferSize, _depthResource));

    CUDAPM_STOP_EVENT(vsrender_pre_frame_mapbuffers);

    // Use the same buffer as input and output of the ray-tracing
    d_outFrameColor->manualSetDataStorage(d_inFrameColor->getPointer());
    d_outFrameDepth->manualSetDataStorage(d_inFrameDepth->getPointer());

	doRender(modelMatrix, viewMatrix, projectionMatrix);

    CUDAPM_START_EVENT(vsrender_post_frame_unmapbuffers);
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_colorResource, 0));
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_depthResource, 0));
	CUDAPM_STOP_EVENT(vsrender_post_frame_unmapbuffers);
}

/******************************************************************************
 * ...
 *
 * @param modelMatrix ...
 * @param viewMatrix ...
 * @param projMatrix ...
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
void VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::doRender(const float4x4 &modelMatrix, const float4x4 &viewMatrix, const float4x4 &projMatrix)
{
	RenderViewContext viewContext;

	if ( this->_clearRequested )
	{
		CUDAPM_START_EVENT(vsrender_clear);

		frameNumAfterUpdate = 0;
		fastBuildMode = true;

		if ( this->_volumeTreeCache )
		{
			this->_volumeTreeCache->clearCache();
		}

		CUDAPM_STOP_EVENT(vsrender_clear);

		this->_clearRequested = false;
	}

	// Extract zNear, zFar as well as the distance in view space from the center of the screen
	// to each side of the screen.
	const float
	fleft	= projMatrix._array[14] * (projMatrix._array[8]-1.0f) / (projMatrix._array[0]*(projMatrix._array[10]-1.0f)),
	fright	= projMatrix._array[14] * (projMatrix._array[8]+1.0f) / (projMatrix._array[0]*(projMatrix._array[10]-1.0f)),
	ftop	= projMatrix._array[14] * (projMatrix._array[9]+1.0f) / (projMatrix._array[5]*(projMatrix._array[10]-1.0f)),
	fbottom	= projMatrix._array[14] * (projMatrix._array[9]-1.0f) / (projMatrix._array[5]*(projMatrix._array[10]-1.0f)),
	fnear	= projMatrix._array[14] / (projMatrix._array[10]-1.0f),
	ffar	= projMatrix._array[14] / (projMatrix._array[10]+1.0f);

	///////////////////////////////////////////

	//transfor matrices
	float4x4 invModelMatrixT=transpose(inverse(modelMatrix));
	float4x4 invViewMatrixT=transpose(inverse(viewMatrix));

	float4x4 modelMatrixT=transpose(modelMatrix);
	float4x4 viewMatrixT=transpose(viewMatrix);

	//float4x4 viewMatrix=(inverse(invViewMatrix));

	viewContext.invViewMatrix=invViewMatrixT;
	viewContext.viewMatrix=viewMatrixT;
	viewContext.invModelMatrix=invModelMatrixT;
	viewContext.modelMatrix=modelMatrixT;

	viewContext.frustumNear=fnear;
	viewContext.frustumNearINV=1.0f/fnear;
	viewContext.frustumFar=ffar;
	viewContext.frustumLeft=fleft;
	viewContext.frustumRight=fright;
	viewContext.frustumBottom=fbottom;
	viewContext.frustumTop=ftop;
	viewContext.frustumC= projMatrix._array[10]; // -(ffar+fnear)/(ffar-fnear);
	viewContext.frustumD= projMatrix._array[14]; // (-2.0f*ffar*fnear)/(ffar-fnear);
	viewContext.inFrameColor=d_inFrameColor->getDeviceArray();
	viewContext.inFrameDepth=d_inFrameDepth->getDeviceArray();
	viewContext.outFrameColor=d_outFrameColor->getDeviceArray();
	viewContext.outFrameDepth=d_outFrameDepth->getDeviceArray();

	float3 viewPlanePosWP = mul( viewContext.invViewMatrix, make_float3( fleft, fbottom, -fnear ) );
	viewContext.viewCenterWP = mul( viewContext.invViewMatrix, make_float3( 0.5f*(fleft+fright), 0.5f*(fbottom+ftop), 0.f ) );
	viewContext.viewPlaneDirWP = viewPlanePosWP - viewContext.viewCenterWP;
	float3 viewPlanePosTP = mul( viewContext.invModelMatrix, viewPlanePosWP );
	viewContext.viewCenterTP = mul( viewContext.invModelMatrix, viewContext.viewCenterWP );
	viewContext.viewPlaneDirTP = viewPlanePosTP - viewContext.viewCenterTP;

	///Resolution dependant stuff///
	viewContext.frameSize = _frameSize;
	viewContext.pixelSize = make_float2( (fright - fleft) / (float)_frameSize.x,	(ftop - fbottom) / (float) _frameSize.y );
	viewContext.viewPlaneXAxisWP = mul( viewContext.invViewMatrix, make_float3( fright, fbottom, -fnear ) );
	viewContext.viewPlaneYAxisWP = mul( viewContext.invViewMatrix, make_float3( fleft, ftop, -fnear ) );
	viewContext.viewPlaneXAxisTP = mul( viewContext.invModelMatrix, viewContext.viewPlaneXAxisWP );
	viewContext.viewPlaneYAxisTP = mul( viewContext.invModelMatrix, viewContext.viewPlaneYAxisWP );
	viewContext.viewPlaneXAxisWP = (viewContext.viewPlaneXAxisWP - viewPlanePosWP) / (float)viewContext.frameSize.x;
	viewContext.viewPlaneYAxisWP = (viewContext.viewPlaneYAxisWP - viewPlanePosWP) / (float)viewContext.frameSize.y;
	viewContext.viewPlaneXAxisTP = (viewContext.viewPlaneXAxisTP - viewPlanePosTP) / (float)viewContext.frameSize.x;
	viewContext.viewPlaneYAxisTP = (viewContext.viewPlaneYAxisTP - viewPlanePosTP) / (float)viewContext.frameSize.y;

	CUDAPM_START_EVENT(vsrender_copyconsts_frame);

	int maxVolTreeDepth = this->_volumeTree->getMaxDepth();

	CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_renderViewContext, &viewContext, sizeof( viewContext ) ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_currentTime, (&this->_currentTime), sizeof( this->_currentTime ), 0, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_maxVolTreeDepth, &maxVolTreeDepth, sizeof( maxVolTreeDepth ), 0, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_userParam, (&this->_userParam), sizeof( this->_userParam ), 0, cudaMemcpyHostToDevice ) );

	// TO DO : move this "Performance monitor" piece of code in another place
#ifdef USE_CUDAPERFMON
	// Performance monitor
	if ( GvPerfMon::CUDAPerfMon::getApplicationPerfMon()._requestResize ) 
	{
		// Update device memory
		GvCore::Array3DGPULinear< GvCore::uint64 >* my_d_timersArray = GvPerfMon::CUDAPerfMon::getApplicationPerfMon().getKernelTimerArray();
		GvCore::Array3DKernelLinear< GvCore::uint64 > h_timersArray = my_d_timersArray->getDeviceArray();
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvPerfMon::k_timersArray, &h_timersArray, sizeof( h_timersArray ), 0, cudaMemcpyHostToDevice ) );
		
		// Update device memory
		uchar* my_d_timersMask = GvPerfMon::CUDAPerfMon::getApplicationPerfMon().getKernelTimerMask();
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvPerfMon::k_timersMask, &my_d_timersMask, sizeof( my_d_timersMask ), 0, cudaMemcpyHostToDevice ) );

		// Update the performnace monitor's state
		GvPerfMon::CUDAPerfMon::getApplicationPerfMon()._requestResize = false;
	}
#endif
	
	float voxelSizeMultiplier = 1.0f;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_voxelSizeMultiplier, (&voxelSizeMultiplier), sizeof( voxelSizeMultiplier ), 0, cudaMemcpyHostToDevice ) );

	CUDAPM_STOP_EVENT(vsrender_copyconsts_frame);







	/// call the render kernel

	this->_volumeTreeCache->preRenderPass();

	frameNumAfterUpdate = 0;

	fastBuildMode = false;

	CUDAPM_START_EVENT_GPU(vsrender_gigavoxels);

	dim3 blockSize(RenderBlockResolution::x, RenderBlockResolution::y, 1);
	dim3 gridSize( iDivUp(_frameSize.x,RenderBlockResolution::x), iDivUp(_frameSize.y, RenderBlockResolution::y), 1);

	int nModel = (int) modelMatrix._array[0];
	static float4x4 *matrixArray_dev = NULL;
	if (matrixArray_dev == NULL)
		CUDA_SAFE_CALL( cudaMalloc( (void**)&matrixArray_dev,
									1001 * sizeof(float4x4) ) );
	CUDA_SAFE_CALL( cudaMemcpy(matrixArray_dev,
							   &modelMatrix,
							   (nModel+1) * sizeof(float4x4),
							   cudaMemcpyHostToDevice) );


	if ( this->_dynamicUpdate )
	{
		if ( this->_hasPriorityOnBricks )
		{
			// Priority on brick is set to TRUE to force loading data at low resolution first
			RenderKernelInstancing< RenderBlockResolution, false, true, SampleShader >
							<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject(),
								1 + matrixArray_dev, nModel);
		}
		else
		{
			RenderKernelInstancing< RenderBlockResolution, false, false, SampleShader >
							<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject(),
								1 + matrixArray_dev, nModel );
		}
	}
	else
	{
		RenderKernelInstancing< RenderBlockResolution, false, false, SampleShader >
						<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject(),
								1 + matrixArray_dev, nModel );
	}

	CUT_CHECK_ERROR("RenderKernelSimple");

	CUDAPM_STOP_EVENT_GPU(vsrender_gigavoxels);

	CUDAPM_START_EVENT(vsrender_load);

	//Bricks loading
	if ( this->_dynamicUpdate ) 
	{
		// dynamicUpdate = false;

		this->_volumeTreeCache->_intraFramePass = false;

		// Post render pass
		this->_volumeTreeCache->postRenderPass();
	}

	CUDAPM_STOP_EVENT(vsrender_load);







	CUDAPM_RENDER_CACHE_INFO(256, 512);

	/*{
		uint2 poolRes = make_uint2(180, 150);
		uint2 poolScale = make_uint2(2, 2);

		dim3 blockSize(10, 10, 1);
		dim3 gridSize(poolRes.x * poolScale.x / blockSize.x, poolRes.y * poolScale.y / blockSize.y, 1);
		RenderDebug<<<gridSize, blockSize, 0>>>(d_outFrameColor->getDeviceArray(), poolRes, poolScale);
	}*/

	frameNumAfterUpdate++;
}

/******************************************************************************
 * ...
 *
 * @param colorResource ...
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
void VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::setColorResource( struct cudaGraphicsResource* colorResource )
{
    _colorResource = colorResource;
}

/******************************************************************************
 * ...
 *
 * @param depthResource ...
 ******************************************************************************/
template< typename VolumeTreeType, typename VolumeTreeCacheType, typename ProducerType, typename SampleShader >
void VolumeTreeRendererCUDA_instancing< VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader >
::setDepthResource (struct cudaGraphicsResource* depthResource )
{
    _depthResource = depthResource;
}

} // namespace GvRenderer
