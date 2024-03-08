/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 */

/** 
 * @version 1.0
 */

#include "SampleCore.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GsVector.h>
#include <GvStructure/GsVolumeTree.h>
#include <GvStructure/GsDataProductionManager.h>
#include <GvRendering/GsRendererCUDA.h>
#include <GvUtils/GsSimplePipeline.h>
#include <GvUtils/GsSimpleHostShader.h>
#include <GvUtils/GsSimplePriorityPoliciesManagerKernel.h>
#include <GvUtils/GsDataLoader.h>
#include <GvUtils/GsTransferFunction.h>
#include <GvPerfMon/GsPerformanceMonitor.h>
#include <GvCore/GsError.h>
#include <GvStructure/GsWriter.h>
#include <GvCore/GsDataTypeList.h>

// Project
#include "Producer.h"
#include "ShaderKernel.h"
#include "RawFileReader.h"

// GvViewer
#include <GvvApplication.h>

// Cuda SDK
#include <helper_math.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>

// STL
#include <vector>
#include <string>
#include <chrono>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvStructure;
using namespace GvRendering;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
// Théorie -> ça overflow (0o0)
/*
#define NODEPOOL_MEMSIZE	(64*1024*1024)		// 8 Mo
#define BRICKPOOL_MEMSIZE	(2047*1024*1024)		// 256 Mo
*/

///**
// * Tag name identifying a space profile element
// */
//const char* SampleCore::cTypeName = "GigaVoxelsPipeline";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

//template< typename TDataTypeList, int TChannel >
//void listDataTypes()
//{
//	// Typedef to access the channel in the data type list
//	typedef typename Loki::TL::TypeAt< TDataTypeList, TChannel >::Result ChannelType;
//
//	std::cout << GvCore::typeToString< ChannelType >() << std::endl;
////	// Build filename according to GigaVoxels internal syntax
////	std::stringstream filename;
////	filename << mFileName << "_BR" << mBrickSize << "_B" << mBorderSize << "_L" << mLevel
////		<< "_C" << TChannel << "_" << GvCore::typeToString< ChannelType >() << mFileExt;
////
////	// Store generated filename
////	mResult->push_back( filename.str() );
//}


/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
:	GvViewerCore::GvvPipelineInterface()
,	_pipeline( NULL )
,	_renderer( NULL )
,	_producer( NULL )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_depthBuffer( 0 )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 0 )
,	_filename()
,	_resolution( 0 )
,	_producerThresholdLow( 0.f )
,	_producerThresholdHigh( 65535.f )
,	_shaderThresholdLow( 0.f )
,	_shaderThresholdHigh(65535.f)
,	_xRayConst(40.f)
,	_fullOpacityDistance( 300.f )
,	_gradientStep( 0.f )
,	_transferFunction( NULL )
,	_minDataValue( 0.f )
,	_maxDataValue( 0.f )
,	_gradientRendering(false)
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	// Rotation used to position the GigaVoxels data structure
	_rotation[ 0 ] = 0.0f;
	_rotation[ 1 ] = 0.0f;
	_rotation[ 2 ] = 0.0f;
	_rotation[ 3 ] = 0.0f;

	// Scale used to transform the GigaVoxels data structure
	_scale = 1.0f;

	// Light position
	_lightPosition = make_float3(  1.f, 1.f, 1.f );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _pipeline;
	_pipeline = NULL;

	// Finalize the transfer function (free memory)
	finalizeTransferFunction();
}

///******************************************************************************
// * Returns the type of this browsable. The type is used for retrieving
// * the context menu or when requested or assigning an icon to the
// * corresponding item
// *
// * @return the type name of this browsable
// ******************************************************************************/
//const char* SampleCore::getTypeName() const
//{
//	return cTypeName;
//}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "RawDataLoader";
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{

	CUDAPM_INIT();

	// Define cache sizes
	_nodeMemoryPool = 64 * 1024 * 1024; // Hardcoded for now but we'll see if there is a need for more (but I don't think so)

	// TO DO :
	// Define the brick memory pool size according to the available VRAM
	// cudaMemGetInfo isn't giving me the correct svalue bruh -> might be correct actually but does swap stuff ?f
	size_t freeGPUMem, totalGPUMem;
	cudaMemGetInfo( &freeGPUMem, &totalGPUMem);
  
	_brickMemoryPool = (size_t)5000 * (size_t)1024 * (size_t)1024;
	freeGPUMem *= 0.70;
	_brickMemoryPool = freeGPUMem;

	// Temp fix because we have issue with data pools > 2Go
	/*
	if (freeGPUMem <= _brickMemoryPool){
		_brickMemoryPool = freeGPUMem;
	}
	*/
	
	std::cout << "Free mem: " << freeGPUMem << " Total mem: " << totalGPUMem << " The one I set myself " << _brickMemoryPool << std::endl;

	QString filename( get3DModelFilename().c_str() );
	QFileInfo dataFileInfo( filename );
	std::cout << dataFileInfo.suffix().toStdString() << std::endl;
	QString dataFilename;
	GvCore::GvDataTypeInspector< DataType > dataTypeInspector; // Seems to be just a vector of DataTypes

	unsigned int brickSize = BrickRes::get().x;

	// Fill the data type list used to store voxels in the data structure
	GvViewerCore::GvvDataType& dataTypes = editDataTypes();
	GvCore::StaticLoop< GvCore::GvDataTypeInspector< DataType >, GvCore::DataNumChannels< DataType >::value - 1 >::go( dataTypeInspector ); // See GsDataTypeList.h in GsCore
	dataTypes.setTypes( dataTypeInspector._dataTypes );
	
	if (dataFileInfo.suffix().toStdString() == "xml" ){
		
		dataFilename = dataFileInfo.absolutePath() + QString( "/" ) + dataFileInfo.completeBaseName();
		dataFilename += ".xml";
		
		goto fileload;
	}

	// Use custom RAW data importer
	// - user data type
	assert( GvCore::DataNumChannels< DataType >::value == 1 );
	typedef GvCore::DataChannelType< DataType, 0/*data channel index*/ >::Result UserDataType;
	// - create reader
	RawFileReader< UserDataType >* rawFileReader = new RawFileReader< UserDataType >();
	
	if ( rawFileReader != NULL )
	{
		// Initialize reader
		// - file path + name
		QString dataFilename = dataFileInfo.absolutePath() + QDir::separator() + dataFileInfo.completeBaseName();
		rawFileReader->setFilename( dataFilename.toLatin1().constData() );
		// - data resolution
		rawFileReader->setDataResolution( get3DModelResolution() );
		// - file mode
		//rawFileReader->setMode( static_cast< GvVoxelizer::GsIRAWFileReader::Mode >( dataLoaderDialog.getModelFileMode() ) );
		rawFileReader->setMode( GvVoxelizer::GsIRAWFileReader::eBinary );
		// data type
		const char* dataTypeName = GvCore::typeToString< UserDataType >();
		GvVoxelizer::GsDataTypeHandler::VoxelDataType voxelDataType;
		std::cout << "DataTypeName : " << dataTypeName << std::endl;
		if ( strcmp( dataTypeName, "uchar" ) == 0 )
		{
			voxelDataType = GvVoxelizer::GsDataTypeHandler::gvUCHAR;
			std::cout << "Uchar" << std::endl;
		}
		else if ( strcmp( dataTypeName, "ushort" ) == 0 )
		{
			voxelDataType = GvVoxelizer::GsDataTypeHandler::gvUSHORT;
			std::cout << "UShort" << std::endl;
		}
		else if ( strcmp( dataTypeName, "float" ) == 0 )
		{
			voxelDataType = GvVoxelizer::GsDataTypeHandler::gvFLOAT;
		}
		else
		{
			// TO DO
			// Handle error
			std::cout << "Error : the data type is not handle by the internal voxelizer : " << dataTypeName << std::endl;
			assert( false );
		}
		rawFileReader->setDataType( voxelDataType );

		// Read file and build GigaSpace mip-map pyramid files
		// BRICK SIZE
		auto start = std::chrono::high_resolution_clock::now();
		rawFileReader->read(brickSize, _trueX, _trueY, _trueZ, _radius);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
		std::cout << "\nConversion time : " << duration.count() << std::endl;
		// Update internal info
		_minDataValue = static_cast< float >( rawFileReader->getMinDataValue() );
		_maxDataValue = static_cast< float >( rawFileReader->getMaxDataValue() );

		// Clean reader resources
		delete rawFileReader;
		rawFileReader = NULL;
	}

	// Use GigaSpace Meta Data file writer
	GsWriter* gigaSpaceWriter = new GsWriter();

	dataFilename = dataFileInfo.absolutePath() + /*QDir::separator()*/QString( "/" ) + dataFileInfo.completeBaseName();
	dataFilename += ".xml";
	// - exporter configuration
	gigaSpaceWriter->setModelDirectory( dataFileInfo.absolutePath().toLatin1().constData() );
	gigaSpaceWriter->setModelName( dataFileInfo.completeBaseName().toLatin1().constData() );
	std::cout << "get3DModelResolution : " << get3DModelResolution() << std::endl;
	const unsigned int modelMaxResolution = static_cast< unsigned int >( log( static_cast< float >( get3DModelResolution() / brickSize ) ) / log( static_cast< float >( 2 ) ) );
	std::cout << "\nModel max resolution : "<< modelMaxResolution << std::endl;
	gigaSpaceWriter->setModelMaxResolution( modelMaxResolution );
	const unsigned int brickWidth = PipelineType::BrickTileResolution::get().x;	/*BEWARE : bricks are uniforms*/
	gigaSpaceWriter->setBrickWidth( brickWidth );
	gigaSpaceWriter->setNbDataChannels( GvCore::DataNumChannels< DataType >::value );
	// - fill data types
	//std::vector< std::string > dataTypeNames;
	//GvCore::GvDataTypeInspector< DataType > dataTypeInspector;
	//GvCore::StaticLoop< GvCore::GvDataTypeInspector< DataType >, GvCore::DataNumChannels< DataType >::value - 1 >::go( dataTypeInspector );
	gigaSpaceWriter->setDataTypeNames( dataTypeInspector._dataTypes );
	// - write Meta Data file
	bool statusOK = gigaSpaceWriter->write();
	// Handle error(s)
	assert( statusOK );
	if ( ! statusOK )
	{
		std::cout << "Error during writing GigaSpace's Meta Data file" << std::endl;
	}
	// Destroy GigaSpace Meta Data file writer
	delete gigaSpaceWriter;
	gigaSpaceWriter = NULL;
	
fileload:

	// Compute the size of one node in the cache 
	size_t nodeElemSize = PipelineType::NodeTileResolution::numElements * sizeof(GvStructure::GsNode);
	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = _nodeMemoryPool / nodeElemSize; // Seulement utilisé pour calculer nodePoolRes
	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3((uint)floorf(powf((float)nodePoolNumElems, 1.0f / 3.0f))) * NodeRes::get(); // Donnée au producer
	
	// TO DO :
	// Test empty and existence of filename
	std::cout << dataFilename.toStdString() << std::endl;
	GvUtils::GsDataLoader< DataType >* dataLoader = new GvUtils::GsDataLoader< DataType >(
														dataFilename.toStdString(),
														PipelineType::BrickTileResolution::get(), PipelineType::BrickTileBorderSize, true );
	unsigned int dataResolution = dataLoader->getDataResolution().x;
	std::cout << "DataResolution : " << dataResolution << std::endl;
	// Shader creation
	ShaderType* shader = new ShaderType();
	GS_CUDA_SAFE_CALL(cudaMemcpyToSymbol(cDataResolution, &dataResolution, sizeof(dataResolution), 0, cudaMemcpyHostToDevice));

	// Pipeline initialization
	_pipeline = new PipelineType();
	_pipeline->initialize( _nodeMemoryPool, _brickMemoryPool, shader ); // Juste donnée à la pipeline

	// Producer initialization
	_producer = new ProducerType( 64 * 1024 * 1024, nodePoolRes.x * nodePoolRes.y * nodePoolRes.z ); // Hardcoded cache size ?
	assert( _producer != NULL );
	_producer->attachProducer( dataLoader );
	_pipeline->addProducer( _producer );

	// Renderer initialization
	_renderer = new RendererType( _pipeline->editDataStructure(), _pipeline->editCache() );
	assert( _renderer != NULL );
	_pipeline->addRenderer( _renderer );

	// Pipeline configuration
	_maxVolTreeDepth = 10; // Hardcoded again ?
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
	
	////------------------------------------------------------------
	//// Typedef to access the channel in the data type list
	//for ( int i = 0; i < GvCore::DataNumChannels< DataType >::value; i++ )
	//{
	//	listDataTypes< DataType, Loki::Int2Type< i > ) >();
	//}
	////------------------------------------------------------------

	//-----------------------------------------------
	//TEST
	//_pipeline->editCache()->setMaxNbNodeSubdivisions( 100 );
	//_pipeline->editCache()->setMaxNbBrickLoads( 100 );
	//-----------------------------------------------

	// Initialize the transfer function
	initializeTransferFunction();
	
	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	// Need to initialize CUDA memory with light position
	float x,y,z;
	getLightPosition( x, y, z );
	setLightPosition( x, y, z );
	setProducerThresholdLow( 0.f );	// no threshold by default
	setProducerThresholdHigh( 65535.f );	// no threshold by default
	setShaderThresholdLow( 0.f );	// no threshold by default
	setShaderThresholdHigh(65535.f);	// no threshold by default
	setFullOpacityDistance( dataResolution ); // the distance ( 1 / FullOpacityDistance ) is the distance after which opacity is full.
	setGradientStep( 0.25f );
	setGradientRenderingBool(false);
	setXRayConst(40.f);
	_transferFunction->updateDeviceMemory();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

	glMatrixMode( GL_MODELVIEW );

	if ( _displayOctree )
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		// Translation used to position the GigaVoxels data structure
		glScalef( _scale, _scale, _scale );
		glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
		_pipeline->editDataStructure()->render();
		glPopMatrix();
		glDisable( GL_DEPTH_TEST );

		// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
		glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		GV_CHECK_GL_ERROR();
	}
	else
	{
		glClear( GL_COLOR_BUFFER_BIT );
	}

	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );

	// extract viewport
	GLint params[4];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4(params[0], params[1], params[2], params[3]);

	// render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	// Translation used to position the GigaVoxels data structure
	glScalef( _scale, _scale, _scale );
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render
	_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );

	// Render the result to the screen
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	glDisable( GL_DEPTH_TEST );
	glEnable( GL_TEXTURE_RECTANGLE_EXT );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );

	// Draw a full screen quad
	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = _width;
	GLint tMax = _height;
	glBegin( GL_QUADS );
	glColor3f( 1.0f, 1.0f, 1.0f );
	glTexCoord2i( sMin, tMin ); glVertex2i( -1, -1 );
	glTexCoord2i( sMax, tMin ); glVertex2i(  1, -1 );
	glTexCoord2i( sMax, tMax ); glVertex2i(  1,  1 );
	glTexCoord2i( sMin, tMax ); glVertex2i( -1,  1 );
	glEnd();

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	glDisable( GL_TEXTURE_RECTANGLE_EXT );
	
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	//_volumeTreeRenderer->doPostRender();
	
	// Update GigaVoxels info
	/*_pipeline->editRenderer()*/_renderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL( _displayPerfmon - 1 );
		
		// SORTIE CONSOLE
		// GvPerfMon::CUDAPerfMon::get().displayFrame();

		// HOST
		//GvPerfMon::CUDAPerfMon::get().displayFrameGL( 1 );
		
		// DEVICE
		//GvPerfMon::CUDAPerfMon::get().displayFrameGL( 0 );
	}
}

/******************************************************************************
 * ...
 *
 * @param width ...
 * @param height ...
 ******************************************************************************/
void SampleCore::resize(int width, int height)
{
	_width = width;
	_height = height;

	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(_width, _height));

	/*uchar *timersMask = GvPerfMon::CUDAPerfMon::get().getKernelTimerMask();
	cudaMemset(timersMask, 255, _width * _height);*/

	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	// Create frame-dependent objects

	if (_depthBuffer)
		glDeleteBuffers(1, &_depthBuffer);

	if (_colorTex)
		glDeleteTextures(1, &_colorTex);
	if (_depthTex)
		glDeleteTextures(1, &_depthTex);

	if (_frameBuffer)
		glDeleteFramebuffers(1, &_frameBuffer);

	glGenTextures(1, &_colorTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _colorTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenBuffers(1, &_depthBuffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, _depthBuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	GV_CHECK_GL_ERROR();

	glGenTextures(1, &_depthTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _depthTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenFramebuffers(1, &_frameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _colorTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GV_CHECK_GL_ERROR();

	// Create CUDA resources from OpenGL objects
	if ( _displayOctree )
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::clearCache()
{
	_pipeline->clear();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	_displayOctree = !_displayOctree;

	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	if ( _displayOctree )
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	const bool status = _pipeline->hasDynamicUpdate();
	_pipeline->setDynamicUpdate( ! status );
}

/******************************************************************************
 * ...
 *
 * @param mode ...
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
{
	if (_displayPerfmon){
		_displayPerfmon = 0;
		GvPerfMon::CUDAPerfMon::_isActivated = false;
	} else {
		_displayPerfmon = mode;
		GvPerfMon::CUDAPerfMon::_isActivated = true;
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if (_maxVolTreeDepth < 32)
		_maxVolTreeDepth++;

	//mVolumeTree->setMaxDepth( _maxVolTreeDepth );
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if (_maxVolTreeDepth > 0)
		_maxVolTreeDepth--;

	//mVolumeTree->setMaxDepth( _maxVolTreeDepth );
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * Get the node tile resolution of the data structure.
 *
 * @param pX the X node tile resolution
 * @param pY the Y node tile resolution
 * @param pZ the Z node tile resolution
 ******************************************************************************/
void SampleCore::getDataStructureNodeTileResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
	const uint3& nodeTileResolution = _pipeline->editDataStructure()->getNodeTileResolution().get();

	pX = nodeTileResolution.x;
	pY = nodeTileResolution.y;
	pZ = nodeTileResolution.z;
}

/******************************************************************************
 * Get the brick resolution of the data structure (voxels).
 *
 * @param pX the X brick resolution
 * @param pY the Y brick resolution
 * @param pZ the Z brick resolution
 ******************************************************************************/
void SampleCore::getDataStructureBrickResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
	const uint3& brickResolution = _pipeline->editDataStructure()->getBrickResolution().get();

	pX = brickResolution.x;
	pY = brickResolution.y;
	pZ = brickResolution.z;
}

/******************************************************************************
 * Get the max depth.
 *
 * @return the max depth
 ******************************************************************************/
unsigned int SampleCore::getRendererMaxDepth() const
{
	return _pipeline->editDataStructure()->getMaxDepth();
}

/******************************************************************************
 * Set the max depth.
 *
 * @param pValue the max depth
 ******************************************************************************/
void SampleCore::setRendererMaxDepth( unsigned int pValue )
{
	_pipeline->editDataStructure()->setMaxDepth( pValue );
}

/******************************************************************************
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbNodeSubdivisions() const
{
	return _pipeline->editCache()->getMaxNbNodeSubdivisions();
}

/******************************************************************************
 * Set the max number of requests of node subdivisions.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbNodeSubdivisions( unsigned int pValue )
{
	_pipeline->editCache()->setMaxNbNodeSubdivisions( pValue );
}

/******************************************************************************
 * Get the max number of requests of brick of voxel loads.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbBrickLoads() const
{
	return _pipeline->editCache()->getMaxNbBrickLoads();
}

/******************************************************************************
 * Set the max number of requests of brick of voxel loads.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbBrickLoads( unsigned int pValue )
{
	_pipeline->editCache()->setMaxNbBrickLoads( pValue );
}

/******************************************************************************
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
void SampleCore::setRendererPriorityOnBricks( bool pFlag )
{
	/*_pipeline->editRenderer()*/_renderer->setPriorityOnBricks( pFlag );
}

/******************************************************************************
 * Specify color to clear the color buffer
 *
 * @param pRed red component
 * @param pGreen green component
 * @param pBlue blue component
 * @param pAlpha alpha component
 ******************************************************************************/
void SampleCore::setClearColor( unsigned char pRed, unsigned char pGreen, unsigned char pBlue, unsigned char pAlpha )
{
	/*_pipeline->editRenderer()*/_renderer->setClearColor( make_uchar4( pRed, pGreen, pBlue, pAlpha ) );
}

/******************************************************************************
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
 ******************************************************************************/
bool SampleCore::hasLight() const
{
	return false;
}

/******************************************************************************
 * Get the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::getLightPosition( float& pX, float& pY, float& pZ ) const
{
	pX = _lightPosition.x;
	pY = _lightPosition.y;
	pZ = _lightPosition.z;
}

/******************************************************************************
 * Set the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setLightPosition( float pX, float pY, float pZ )
{
	// Update DEVICE memory with "light position"
	//
	// WARNING
	// Apply inverse modelisation matrix applied on the GigaVoxels object to set light position correctly.
	// Here a glTranslatef( -0.5f, -0.5f, -0.5f ) has been used.
	_lightPosition.x = pX/* - _translation[ 0 ]*/;
	_lightPosition.y = pY/* - _translation[ 1 ]*/;
	_lightPosition.z = pZ/* - _translation[ 2 ]*/;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &_lightPosition, sizeof( _lightPosition ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
 ******************************************************************************/
bool SampleCore::has3DModel() const
{
	return true;
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
string SampleCore::get3DModelFilename() const
{
	return _filename;
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void SampleCore::set3DModelFilename( const string& pFilename )
{
	_filename = pFilename;
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int SampleCore::get3DModelResolution() const
{
	return _resolution;
}

/******************************************************************************
 * Set the 3D model resolution
 *
 * @param pValue the 3D model resolution
 ******************************************************************************/
void SampleCore::set3DModelResolution( unsigned int pValue )
{
	_resolution = pValue;
}

/******************************************************************************
 * Set the 3D model resolution
 *
 * @param pValue the 3D model resolution
 ******************************************************************************/
void SampleCore::setTrueResolution(unsigned int trueX, unsigned int trueY, unsigned int trueZ)
{
	_trueX = trueX;
	_trueY = trueY;
	_trueZ = trueZ;
}

void SampleCore::setRadius(unsigned int radius)
{
	_radius = radius;
}

/******************************************************************************
 * Get the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::getTranslation( float& pX, float& pY, float& pZ ) const
{
	pX = _translation[ 0 ];
	pY = _translation[ 1 ];
	pZ = _translation[ 2 ];
}

/******************************************************************************
 * Set the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::setTranslation( float pX, float pY, float pZ )
{
	_translation[ 0 ] = pX;
	_translation[ 1 ] = pY;
	_translation[ 2 ] = pZ;
}

/******************************************************************************
 * Get the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const
{
	pAngle = _rotation[ 0 ];
	pX = _rotation[ 1 ];
	pY = _rotation[ 2 ];
	pZ = _rotation[ 3 ];
}

/******************************************************************************
 * Set the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::setRotation( float pAngle, float pX, float pY, float pZ )
{
	_rotation[ 0 ] = pAngle;
	_rotation[ 1 ] = pX;;
	_rotation[ 2 ] = pY;;
	_rotation[ 3 ] = pZ;;
}

/******************************************************************************
 * Get the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::getScale( float& pValue ) const
{
	pValue = _scale;
}

/******************************************************************************
 * Set the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::setScale( float pValue )
{
	_scale = pValue;
}

/******************************************************************************
 * Get the producer's threshold
 *
 * @return the threshold
 ******************************************************************************/
float SampleCore::getProducerThresholdLow() const
{
	return _producerThresholdLow;
}

/******************************************************************************
 * Get the producer's threshold
 *
 * @return the threshold
 ******************************************************************************/
float SampleCore::getProducerThresholdHigh() const
{
	return _producerThresholdHigh;
}

/******************************************************************************
 * Set the producer's threshold
 *
 * @param pValue the threshold
 ******************************************************************************/
void SampleCore::setProducerThresholdLow( float pValue )
{
	_producerThresholdLow = pValue;

	_producer->setLowThreshold(pValue);

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cProducerThresholdLow, &_producerThresholdLow, sizeof( _producerThresholdLow ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Set the producer's threshold
 *
 * @param pValue the threshold
 ******************************************************************************/
void SampleCore::setProducerThresholdHigh( float pValue )
{
	_producerThresholdHigh = pValue;

	_producer->setHighThreshold(pValue);

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cProducerThresholdHigh, &_producerThresholdHigh, sizeof( _producerThresholdHigh ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the shader's threshold
 *
 * @return the threshold
 ******************************************************************************/
float SampleCore::getShaderThresholdLow() const
{
	return _shaderThresholdLow;
}

/******************************************************************************
 * Get the shader's threshold
 *
 * @return the threshold
 ******************************************************************************/
float SampleCore::getShaderThresholdHigh() const
{
	return _shaderThresholdHigh;
}

/******************************************************************************
 * Set the shader's threshold
 *
 * @param pValue the threshold
 ******************************************************************************/
void SampleCore::setShaderThresholdLow( float pValue )
{
	_shaderThresholdLow = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderThresholdLow, &_shaderThresholdLow, sizeof( _shaderThresholdLow ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Set the shader's threshold
 *
 * @param pValue the threshold
 ******************************************************************************/
void SampleCore::setShaderThresholdHigh( float pValue )
{
	_shaderThresholdHigh = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderThresholdHigh, &_shaderThresholdHigh, sizeof( _shaderThresholdHigh ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the full opacity distance
 *
 * @return the full opacity distance
 ******************************************************************************/
float SampleCore::getFullOpacityDistance() const
{
	return _fullOpacityDistance;
}

/******************************************************************************
 * Set the full opacity distance
 *
 * @param pValue the full opacity distance
 ******************************************************************************/
void SampleCore::setFullOpacityDistance( float pValue )
{
	_fullOpacityDistance = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cFullOpacityDistance, &_fullOpacityDistance, sizeof( _fullOpacityDistance ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the gradient step
 *
 * @return the gradient step
 ******************************************************************************/
float SampleCore::getGradientStep() const
{
	return _gradientStep;
}

/******************************************************************************
 * Set the gradient step
 *
 * @param pValue the gradient step
 ******************************************************************************/
void SampleCore::setGradientStep( float pValue )
{
	_gradientStep = pValue;
	
	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cGradientStep, &_gradientStep, sizeof( _gradientStep ), 0, cudaMemcpyHostToDevice ) );
}

float SampleCore::getXRayConst() const
{
	return _xRayConst;
}

/******************************************************************************
 * Set the XRay constant
 *
 * @param pValue the full opacity distance
 ******************************************************************************/
void SampleCore::setXRayConst(float pValue)
{
	_xRayConst = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL(cudaMemcpyToSymbol(cXRayConst, &_xRayConst, sizeof(cXRayConst), 0, cudaMemcpyHostToDevice));
}

/******************************************************************************
 * Initialize the transfer function
 *
 * @return flag to tell wheter or not it succeeded
 ******************************************************************************/
bool SampleCore::initializeTransferFunction()
{
	// Create the transfer function
	_transferFunction = new GvUtils::GsTransferFunction();
	assert( _transferFunction != NULL );
	if ( _transferFunction == NULL )
	{
		// TO DO
		// Handle error
		// ...

		return false;
	}

	// Initialize transfer fcuntion with a resolution of 256 elements
	_transferFunction->create( 256 ); // 256 because UChar ?

	// Bind the transfer function's internal data to the texture reference that will be used on device code
	textureReference* texRefPtr;
	GS_CUDA_SAFE_CALL( cudaGetTextureReference( (const textureReference **)&texRefPtr, &transferFunctionTexture ) );
	texRefPtr->normalized = true; // Access with normalized texture coordinates
	texRefPtr->filterMode = cudaFilterModeLinear;
	texRefPtr->addressMode[ 0 ] = cudaAddressModeClamp; // Wrap texture coordinates
	texRefPtr->addressMode[ 1 ] = cudaAddressModeClamp;
	texRefPtr->addressMode[ 2 ] = cudaAddressModeClamp;
	GS_CUDA_SAFE_CALL( cudaBindTextureToArray( (const textureReference *)texRefPtr, _transferFunction->_dataArray, &_transferFunction->_channelFormatDesc ) );

	// TODO RESOUDRE LE BUG DE LA TRANSFER FUNCTION (enft pas important mnt que j'y pense)
	//onTransferfunctionChanged();

	// TODO Switch to texture objects instead
	
	return true;
}

/******************************************************************************
 * Finalize the transfer function
 *
 * @return flag to tell wheter or not it succeeded
 ******************************************************************************/
bool SampleCore::finalizeTransferFunction()
{
	// Free memory
	delete _transferFunction;

	return true;
}

/******************************************************************************
 * Tell wheter or not the pipeline has a transfer function.
 *
 * @return the flag telling wheter or not the pipeline has a transfer function
 ******************************************************************************/
bool SampleCore::hasTransferFunction() const
{
	return true;
}

/******************************************************************************
 * Update the associated transfer function
 *
 * @param pData the new transfer function data
 * @param pSize the size of the transfer function
 ******************************************************************************/
void SampleCore::updateTransferFunction( float* pData, unsigned int pSize )
{
	assert( _transferFunction != NULL );
	if ( _transferFunction != NULL )
	{
		// Apply modifications on transfer function's internal data
		float4* tf = _transferFunction->editData();
		unsigned int size = _transferFunction->getResolution();
		assert( size == pSize );
		for ( unsigned int i = 0; i < size; ++i )
		{
			tf[ i ] = make_float4( pData[ 4 * i ], pData[ 4 * i + 1 ], pData[ 4 * i + 2 ], pData[ 4 * i + 3 ] );
		}

		// Apply modifications on device memory
		_transferFunction->updateDeviceMemory();

		// Update cache
		// NOTE : no need toclear the cache because the transfer funtion is applied in the Shader, not the Producer
		//_pipeline->clear();
	}
}


/******************************************************************************
 * Get the min data value
 *
 * @return the min data value
 ******************************************************************************/
float SampleCore::getMinDataValue() const
{
	return _minDataValue;
}

/******************************************************************************
 * Get the max data value
 *
 * @return the max data value
 ******************************************************************************/
float SampleCore::getMaxDataValue() const
{
	return _maxDataValue;
}

bool SampleCore::getGradientRenderingBool() const
{
	return _gradientRendering;
}

void SampleCore::setGradientRenderingBool(bool pValue)
{
	_gradientRendering = pValue;

	_producer->setGradientRendering(pValue);

	// Update device memory
	GS_CUDA_SAFE_CALL(cudaMemcpyToSymbol(cGradientRendering, &_gradientRendering, sizeof(_gradientRendering), 0, cudaMemcpyHostToDevice));

	// Clear cache
	clearCache();
}

void SampleCore::setRenderMode(int index)
{
	_renderMode = index;

	std::cout << "Switched to mode : " << _renderMode << std::endl;

	// Update device memory
	GS_CUDA_SAFE_CALL(cudaMemcpyToSymbol(cRenderMode, &_renderMode, sizeof(_renderMode), 0, cudaMemcpyHostToDevice));
}

/******************************************************************************
 * Get the nodes cache usage
 *
 * @return the nodes cache usage
 ******************************************************************************/
unsigned int SampleCore::getNodeCacheUsage() const
{
	//const unsigned int nbProducedElements = _pipeline->getCache()->getNodesCacheManager()->_totalNumLoads;
	const unsigned int nbProducedElements = _pipeline->getCache()->getNodesCacheManager()->_numElemsNotUsed;
	const unsigned int nbElements = _pipeline->getCache()->getNodesCacheManager()->getNumElements();

	const unsigned int cacheUsage = static_cast< unsigned int >( 100.0f * static_cast< float >( nbElements - nbProducedElements ) / static_cast< float >( nbElements ) );

	//std::cout << "NODE cache usage [ " << nbProducedElements << " / "<< nbElements << " : " << cacheUsage << std::endl;

	return cacheUsage;
}

/******************************************************************************
 * Get the bricks cache usage
 *
 * @return the bricks cache usage
 ******************************************************************************/
unsigned int SampleCore::getBrickCacheUsage() const
{
	//const unsigned int nbProducedElements = _pipeline->getCache()->getBricksCacheManager()->_totalNumLoads;
	const unsigned int nbProducedElements = _pipeline->getCache()->getBricksCacheManager()->_numElemsNotUsed;
	const unsigned int nbElements = _pipeline->getCache()->getBricksCacheManager()->getNumElements();

	const unsigned int cacheUsage = static_cast< unsigned int >( 100.0f * static_cast< float >( nbElements - nbProducedElements ) / static_cast< float >( nbElements ) );

	//std::cout << "BRICK cache usage [ " << nbProducedElements << " / "<< nbElements << " : " << cacheUsage << std::endl;

	return cacheUsage;
}

/******************************************************************************
 * Get the number of tree leaf nodes
 *
 * @return the number of tree leaf nodes
 ******************************************************************************/
unsigned int SampleCore::getNbTreeLeafNodes() const
{
	return _pipeline->getCache()->_nbLeafNodes;
}

/******************************************************************************
 * Get the number of tree nodes
 *
 * @return the number of tree nodes
 ******************************************************************************/
unsigned int SampleCore::getNbTreeNodes() const
{
	return _pipeline->getCache()->_nbNodes;
}

/******************************************************************************
 * Get the node cache memory
 *
 * @return the node cache memory
 ******************************************************************************/
unsigned int SampleCore::getNodeCacheMemory() const
{
	return _nodeMemoryPool / ( 1024U * 1024U );
}

/******************************************************************************
 * Get the node cache capacity
 *
 * @return the node cache capacity
 ******************************************************************************/
unsigned int SampleCore::getNodeCacheCapacity() const
{
	return _pipeline->getCache()->getNodesCacheManager()->getNumElements();
}

/******************************************************************************
 * Get the brick cache memory
 *
 * @return the brick cache memory
 ******************************************************************************/
unsigned int SampleCore::getBrickCacheMemory() const
{
	return _brickMemoryPool / ( 1024U * 1024U );
}

/******************************************************************************
 * Get the brick cache capacity
 *
 * @return the brick cache capacity
 ******************************************************************************/
unsigned int SampleCore::getBrickCacheCapacity() const
{
	return _pipeline->getCache()->getBricksCacheManager()->getNumElements();
}

/******************************************************************************
* Get the flag indicating wheter or not cache monitoring is activated
*
* @return the flag indicating wheter or not cache monitoring is activated
 ******************************************************************************/
bool SampleCore::hasCacheMonitoring() const
{
	return true;
}

