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
#include "GvvPipelineInterfaceViewer.h"
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
 * Helper function to retrieve the SampleCore.
 *
 * @return the SampleCore
 ******************************************************************************/
SampleCore* getSampleCore()
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	return sampleCore;
}

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Default constructor
 * @param pParent parent widget
 * @param pFlags the window flags
 ******************************************************************************/
CustomSectionEditor::CustomSectionEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	GvvSectionEditor( pParent, pFlags )
{
	setupUi( this );

	// Editor name
	setName( tr( "Procedural torus on Device" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
CustomSectionEditor::~CustomSectionEditor()
{
}

///******************************************************************************
// * Return the SampleCore.
// *
// * @return SampleCore
// ******************************************************************************/
//SampleCore* CustomSectionEditor::getSampleCore()
//{
//	GvvApplication& application = GvvApplication::get();
//	GvvMainWindow* mainWindow = application.getMainWindow();
//	Gvv3DWindow* window3D = mainWindow->get3DWindow();
//	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
//	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
//
//	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
//	assert( sampleCore != NULL );
//
//	return sampleCore;
//}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void CustomSectionEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	assert( pBrowsable != NULL );
	SampleCore* pipeline = dynamic_cast< SampleCore* >( pBrowsable );
	assert( pipeline != NULL );
	if ( pipeline != NULL )
	{
		int index;

		// Hypertexture parameters
		_noiseFirstFrequencySpinBox->setValue( pipeline->getNoiseFirstFrequency() );
		_noiseStrengthSpinBox->setValue( pipeline->getNoiseStrength() );

		/*_noiseTypeComboBox->addItem( "Simplex", PipelineType::ProducerType::KernelProducer::SIMPLEX );
		_noiseTypeComboBox->addItem( "Perlin", PipelineType::ProducerType::KernelProducer::PERLIN );

		index = _noiseTypeComboBox->findData( pipeline->getNoiseType() );
		if (index != -1) {
			_noiseTypeComboBox->setCurrentIndex(index);
		}*/
		_noiseTypeComboBox->setCurrentIndex( pipeline->getNoiseType() );

		// Light parameters
		float x, y, z;
		pipeline->getLightPosition( x, y, z );
		_lightXDoubleSpinBox->setValue( x );
		_lightYDoubleSpinBox->setValue( y );
		_lightZDoubleSpinBox->setValue( z );

		_brightnessSpinBox->setValue(pipeline->getBrightness() );

		/*_lightTypeComboBox->addItem( "Phong", PHONG );
		_lightTypeComboBox->addItem( "Lambert", LAMBERT );

		index = _lightTypeComboBox->findData( pipeline->getLightingType() );
		if (index != -1) {
			_lightTypeComboBox->setCurrentIndex(index);
		}*/
		_lightTypeComboBox->setCurrentIndex( pipeline->getLightingType() );
	}
}

/******************************************************************************
 * Slot called when noise first frequency value has changed
 ******************************************************************************/
void CustomSectionEditor::on__noiseFirstFrequencySpinBox_valueChanged( int value )
{
	getSampleCore()->setNoiseFirstFrequency( value );
}

/******************************************************************************
 * Slot called when noise strength value has changed
 ******************************************************************************/
void CustomSectionEditor::on__noiseStrengthSpinBox_valueChanged( double value )
{
	getSampleCore()->setNoiseStrength( value );
}

/******************************************************************************
 * Slot called when noise type value has changed
 ******************************************************************************/
void CustomSectionEditor::on__noiseTypeComboBox_currentIndexChanged( int value )
{
	//int noiseType = _noiseTypeComboBox->itemData(value).toInt();
	//getSampleCore()->setNoiseType( noiseType );
	getSampleCore()->setNoiseType( value );
}

/******************************************************************************
 * Slot called when brightness value has changed
 ******************************************************************************/
void CustomSectionEditor::on__lightTypeComboBox_currentIndexChanged( int value )
{
	//int lightingType = _lightTypeComboBox->itemData(value).toInt();
	//getSampleCore()->setLightingType( lightingType );
	getSampleCore()->setLightingType( value );
}

/******************************************************************************
 * Slot called when brightness value has changed
 ******************************************************************************/
void CustomSectionEditor::on__brightnessSpinBox_valueChanged( double value )
{
	getSampleCore()->setBrightness( value );
}

/******************************************************************************
 * Slot called when light position X value has changed
 ******************************************************************************/
void CustomSectionEditor::on__lightXDoubleSpinBox_valueChanged( double value )
{
	SampleCore *sampleCore = getSampleCore();

	float x, y, z;
   	sampleCore->getLightPosition( x, y, z );
	x = value;
	sampleCore->setLightPosition( x, y, z );
}

/******************************************************************************
 * Slot called when light position Y value has changed
 ******************************************************************************/
void CustomSectionEditor::on__lightYDoubleSpinBox_valueChanged( double value )
{
	SampleCore* sampleCore = getSampleCore();

	float x, y, z;
	sampleCore->getLightPosition( x, y, z );
	y = value;
	sampleCore->setLightPosition( x, y, z );
}

/******************************************************************************
 * Slot called when light position Z value has changed
 ******************************************************************************/
void CustomSectionEditor::on__lightZDoubleSpinBox_valueChanged( double value )
{
	SampleCore* sampleCore = getSampleCore();

	float x, y, z;
	sampleCore->getLightPosition( x, y, z );
	z = value;
	sampleCore->setLightPosition( x, y, z );
}

