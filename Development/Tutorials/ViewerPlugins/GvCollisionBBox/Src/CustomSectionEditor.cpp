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
	setName( tr( "Collision BBox" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
CustomSectionEditor::~CustomSectionEditor()
{
	free(_timerGraphicsView->scene());
}

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

		_noiseTypeComboBox->setCurrentIndex( pipeline->getNoiseType() );

		// Light parameters
		float x, y, z;
		pipeline->getLightPosition( x, y, z );
		_lightXDoubleSpinBox->setValue( x );
		_lightYDoubleSpinBox->setValue( y );
		_lightZDoubleSpinBox->setValue( z );

		_brightnessSpinBox->setValue(pipeline->getBrightness() );

		_lightTypeComboBox->setCurrentIndex( pipeline->getLightingType() );

		// Materials properties
		_ambientDoubleSpinBox->setValue( pipeline->getAmbient() );
		_diffuseDoubleSpinBox->setValue( pipeline->getDiffuse() );
		_specularDoubleSpinBox->setValue( pipeline->getSpecular() );

		// Shape
		_torusRadiusDoubleSpinBox->setValue( pipeline->getTorusRadius() );
		_tubeRadiusDoubleSpinBox->setValue( pipeline->getTubeRadius() );

		// Perfs
		_timerGraphicsView->setScene( new QGraphicsScene() );

		// Create a timer to keep the graphics view up to date.
		QTimer *timer = new QTimer(this);
		connect(timer, SIGNAL(timeout()), this, SLOT(updateTimerGraphicsView()));
		timer->start(1000);
	}

}

/******************************************************************************
 *
 ******************************************************************************/
void CustomSectionEditor::drawPoint( pair< float, unsigned int> point, QBrush brush )
{
	_timerGraphicsView->scene()->addEllipse( point.second, -point.first, 4.0, 4.0, QPen(), brush );

	_timerGraphicsView->centerOn( point.second, -point.first );
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
	getSampleCore()->setNoiseType( value );
}

/******************************************************************************
 * Slot called when brightness value has changed
 ******************************************************************************/
void CustomSectionEditor::on__lightTypeComboBox_currentIndexChanged( int value )
{
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

/******************************************************************************
 * Slot called when the ambient reflection has changed.
 ******************************************************************************/
void CustomSectionEditor::on__ambientDoubleSpinBox_valueChanged( double value )
{
	getSampleCore()->setAmbient( value );
}

/******************************************************************************
 * Slot called when the diffuse reflection has changed.
 ******************************************************************************/
void CustomSectionEditor::on__diffuseDoubleSpinBox_valueChanged( double value )
{
	getSampleCore()->setDiffuse( value );
}

/******************************************************************************
 * Slot called when the specular reflection has changed.
 ******************************************************************************/
void CustomSectionEditor::on__specularDoubleSpinBox_valueChanged( double value )
{
	getSampleCore()->setSpecular( value );
}

/******************************************************************************
 * Slot called when the torus radius has changed.
 ******************************************************************************/
void CustomSectionEditor::on__torusRadiusDoubleSpinBox_valueChanged( double value )
{
	getSampleCore()->setTorusRadius( value );
}

/******************************************************************************
 * Slot called when the specular reflection has changed.
 ******************************************************************************/
void CustomSectionEditor::on__tubeRadiusDoubleSpinBox_valueChanged( double value )
{
	getSampleCore()->setTubeRadius( value );
}

/******************************************************************************
 * Slot called each second to keep the graphics view up to date.
 ******************************************************************************/
void CustomSectionEditor::updateTimerGraphicsView()
{
	SampleCore* sampleCore = getSampleCore();

	// Get the newest values
	vector< pair <float, unsigned int> > lastBricksTime = sampleCore->getTimeBrick();
	vector< pair <float, unsigned int> > lastNodesPoolTime = sampleCore->getTimeNodePool();

	// Draw the points
	QBrush red(Qt::red);
	QBrush green(Qt::green);
	for ( unsigned int i = 0; i < lastBricksTime.size(); ++i ) {
		drawPoint( lastBricksTime[i], red);
	}
	for ( unsigned int i = 0; i < lastNodesPoolTime.size(); ++i ) {
		drawPoint( lastNodesPoolTime[i], green);
	}
}
