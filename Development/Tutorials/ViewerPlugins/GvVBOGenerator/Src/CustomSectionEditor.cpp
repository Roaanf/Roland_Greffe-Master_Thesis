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

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>
#include <Gvv3DWindow.h>
#include <GvvPipelineInterfaceViewer.h>
#include <GvvPipelineInterface.h>

// Project
#include "SampleCore.h"

// Qt
#include <QColorDialog>
#include <QFileDialog>

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
	setName( tr( "VBO Generator" ) );
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
	SampleCore* pipeline = dynamic_cast< SampleCore* >( pBrowsable );
	assert( pipeline != NULL );
	if ( pipeline != NULL )
	{
		_nbPointsSpinBox->setValue( pipeline->getNbPoints() );

		_minLevelToHandleUserDefinedSpinBox->setValue( pipeline->getUserDefinedMinLevelOfResolutionToHandle() );
		_intersectionTypeComboBox->setCurrentIndex( pipeline->getSphereBrickIntersectionType() );
		_pointSizeFaderDoubleSpinBox->setValue( pipeline->getPointSizeFader() );
		
		_geometricCriteriaGroupBox->setChecked( pipeline->hasGeometricCriteria() );
		_minNbPointsPerBrickSpinBox->setValue( pipeline->getMinNbPointsPerBrick() );
		_minNbPointsPerBrickSpinBox->setValue( pipeline->getMinNbPointsPerBrick() );
		_geometricCriteriaGlobalUsageCheckBox->setChecked( pipeline->hasGeometricCriteriaGlobalUsage() );
		
		_apparentMinSizeCriteriaGroupBox->setChecked( pipeline->hasApparentMinSizeCriteria() );
		_apparentMinSizeDoubleSpinBox->setValue( pipeline->getApparentMinSize() );
		
		_apparentMaxSizeCriteriaGroupBox->setChecked( pipeline->hasApparentMaxSizeCriteria() );
		_apparentMaxSizeDoubleSpinBox->setValue( pipeline->getApparentMaxSize() );
		
		_shaderUseUniformColorCheckBox->setChecked( pipeline->hasShaderUniformColor() );
		_shaderUseAnimationCheckBox->setChecked( pipeline->hasShaderAnimation() );
		_shaderUseTextureCheckBox->setChecked( pipeline->hasTexture() );
		_textureLineEdit->setText( pipeline->getTextureFilename().c_str() );
	}
}

/******************************************************************************
 * Slot called when global nb spheres value has changed
 ******************************************************************************/
void CustomSectionEditor::on__nbPointsSpinBox_valueChanged( int pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setNbPoints( pValue );
}

/******************************************************************************
 * Slot called when user defined min level of resolution to handle value has changed
 ******************************************************************************/
void CustomSectionEditor::on__minLevelToHandleUserDefinedSpinBox_valueChanged( int pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setUserDefinedMinLevelOfResolutionToHandle( pValue );
}

/******************************************************************************
 * Slot called when intersection type value has changed
 ******************************************************************************/
void CustomSectionEditor::on__intersectionTypeComboBox_currentIndexChanged( int pIndex )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setSphereBrickIntersectionType( pIndex );
}

/******************************************************************************
 * Slot called when global sphere radius fader value has changed
 ******************************************************************************/
void CustomSectionEditor::on__pointSizeFaderDoubleSpinBox_valueChanged( double pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setPointSizeFader( pValue );
}

/******************************************************************************
 * Slot called when the geometric criteria group box state has changed
 ******************************************************************************/
void CustomSectionEditor::on__geometricCriteriaGroupBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setGeometricCriteria( pChecked );
}

/******************************************************************************
 * Slot called when min nb of spheres per brick value has changed
 ******************************************************************************/
void CustomSectionEditor::on__minNbPointsPerBrickSpinBox_valueChanged( int pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setMinNbPointsPerBrick( pValue );
}

/******************************************************************************
 * Slot called when Geometruc Criteria's global usage check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__geometricCriteriaGlobalUsageCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setGeometricCriteriaGlobalUsage( pChecked );
}

/******************************************************************************
 * Slot called when the screen based criteria group box state has changed
 ******************************************************************************/
void CustomSectionEditor::on__apparentMinSizeCriteriaGroupBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setApparentMinSizeCriteria( pChecked );
}

/******************************************************************************
 * Slot called when global sphere radius fader value has changed
 ******************************************************************************/
void CustomSectionEditor::on__apparentMinSizeDoubleSpinBox_valueChanged( double pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setApparentMinSize( pValue );
}

/******************************************************************************
 * Slot called when the absolute size criteria group box state has changed
 ******************************************************************************/
void CustomSectionEditor::on__apparentMaxSizeCriteriaGroupBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setApparentMaxSizeCriteria( pChecked );
}

/******************************************************************************
 * Slot called when global sphere radius fader value has changed
 ******************************************************************************/
void CustomSectionEditor::on__apparentMaxSizeDoubleSpinBox_valueChanged( double pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setApparentMaxSize( pValue );
}

/******************************************************************************
 * Slot called when shader's uniform color check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__shaderUseUniformColorCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setShaderUniformColorMode( pChecked );
}

/******************************************************************************
 * Slot called when shader's uniform color tool button is released
 ******************************************************************************/
void CustomSectionEditor::on__shaderUniformColorToolButton_released()
{
	QColor color = QColorDialog::getColor( Qt::white, this );
	if ( color.isValid() )
	{
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		if ( pipelineViewer != NULL )
		{
			GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );
			if ( sampleCore != NULL )
			{
				sampleCore->setShaderUniformColor( static_cast< float >( color.red() ) / 255.f
												, static_cast< float >( color.green() ) / 255.f
												, static_cast< float >( color.blue() ) / 255.f
												, static_cast< float >( color.alpha() ) / 255.f );
			}
		}
	}
}

/******************************************************************************
 * Slot called when shader's animation check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__shaderUseAnimationCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setShaderAnimation( pChecked );
}

/******************************************************************************
 * Slot called when shader's animation check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__shaderUseTextureCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setTexture( pChecked );
}

/******************************************************************************
 * Slot called when the 3D model file button has been clicked (released)
 ******************************************************************************/
void CustomSectionEditor::on__textureToolButton_released()
{
	// Try to open 3D model
	QString filename = QFileDialog::getOpenFileName( this, "Choose a texture", QString( "." ), tr( "Texture Files (*.png *.jpg)" ) );
	if ( ! filename.isEmpty() )
	{
		_textureLineEdit->setText( filename );
		_textureLineEdit->setToolTip( filename );

		// Update the GigaVoxels pipeline
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

		SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
		assert( sampleCore != NULL );

		sampleCore->setTextureFilename( filename.toStdString() );
	}
}
