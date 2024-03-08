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

#include "CustomSectionEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QtCore/QUrl>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QVBoxLayout>
#include <QtGui/QToolBar>

// GvViewer
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPluginManager.h"
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterface.h"

// Project
#include "SampleCore.h"

// STL
#include <iostream>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Default constructor
 ******************************************************************************/
CustomSectionEditor::CustomSectionEditor( QWidget *parent, Qt::WindowFlags flags )
:	GvvSectionEditor( parent, flags )
{
	setupUi( this );

	// Editor name
	setName( tr( "RAW Data Loader" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
CustomSectionEditor::~CustomSectionEditor()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void CustomSectionEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	assert( pBrowsable != NULL );
	SampleCore* pipeline = dynamic_cast<SampleCore* >( pBrowsable );
	assert( pipeline != NULL );
	if ( pipeline != NULL )
	{
		// Data parameters
		// - filename
		_filenameLineEdit->setText( pipeline->get3DModelFilename().c_str() );
		// - data type
		const GvViewerCore::GvvDataType& dataTypes = pipeline->getDataTypes();
		const std::vector< std::string >& dataTypeNames = dataTypes.getTypes();
		assert( dataTypeNames.size() == 1 );
		if ( dataTypeNames.size() == 1 )
		{
			_dataTypeLineEdit->setText( dataTypeNames[ 0 ].c_str() );
		}
		// - min/max values
		_minDataValueLineEdit->setText( QString::number( pipeline->getMinDataValue() ) );
		_maxDataValueLineEdit->setText( QString::number( pipeline->getMaxDataValue() ) );

		// Producer parameters
		_producerThresholdDoubleSpinBoxLow->setValue( pipeline->getProducerThresholdLow() );
		_producerThresholdDoubleSpinBoxHigh->setValue( pipeline->getProducerThresholdHigh() );

		// Shader parameters
		_shaderThresholdDoubleSpinBoxLow->setValue( pipeline->getShaderThresholdLow() );
		_shaderThresholdDoubleSpinBoxHigh->setValue( pipeline->getShaderThresholdHigh() );
		_shaderFullOpacityDistanceDoubleSpinBox->setValue( pipeline->getFullOpacityDistance() );

		_xRayConst->setValue(pipeline->getXRayConst());

		// Gradient bool
		// Should be set to the same thing but I don't think I can actually
		//_gradientRenderingCheckBox->setTristate(pipeline->getGradientRenderingBool());
	}
}

/******************************************************************************
 * Slot called when producer's threshold value has changed
 ******************************************************************************/
void CustomSectionEditor::on__producerThresholdDoubleSpinBoxLow_valueChanged( double pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setProducerThresholdLow( pValue );
}

/******************************************************************************
 * Slot called when producer's threshold value has changed
 ******************************************************************************/
void CustomSectionEditor::on__producerThresholdDoubleSpinBoxHigh_valueChanged( double pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setProducerThresholdHigh( pValue );
}

/******************************************************************************
 * Slot called when shader's threshold value has changed
 ******************************************************************************/
void CustomSectionEditor::on__shaderThresholdDoubleSpinBoxLow_valueChanged( double pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );


	sampleCore->setShaderThresholdLow( pValue / 65535 );
}

/******************************************************************************
 * Slot called when shader's threshold value has changed
 ******************************************************************************/
void CustomSectionEditor::on__shaderThresholdDoubleSpinBoxHigh_valueChanged( double pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setShaderThresholdHigh( pValue / 65535 );
}

/******************************************************************************
 * Slot called when shader's full opacity distance value has changed
 ******************************************************************************/
void CustomSectionEditor::on__shaderFullOpacityDistanceDoubleSpinBox_valueChanged( double pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setFullOpacityDistance( pValue );
}

void CustomSectionEditor::on__gradientRenderingCheckBox_stateChanged(int pValue)
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast<SampleCore*>(pipeline);
	assert(sampleCore != NULL);

	sampleCore->setGradientRenderingBool(pValue != 0);
	std::cout << "Set GradientRenderingBool to : " << pValue << std::endl;
}

void CustomSectionEditor::on__renderModeComboBox_currentIndexChanged(int index)
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast<SampleCore*>(pipeline);
	assert(sampleCore != NULL);

	sampleCore->setRenderMode(index);
}

void CustomSectionEditor::on__xRayConst_valueChanged(double value)
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast<SampleCore*>(pipeline);
	assert(sampleCore != NULL);

	sampleCore->setXRayConst(value);
}

