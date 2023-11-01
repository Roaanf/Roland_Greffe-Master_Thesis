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
 * Default constructor
 *
 * @param pParent parent widget
 * @param pFlags the window flags
 ******************************************************************************/
CustomSectionEditor::CustomSectionEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	GvvSectionEditor( pParent, pFlags )
{
	setupUi( this );

	// Editor name
	setName( tr( "BVH Management" ) );
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
		// 3D model parameters
		_3DModelLineEdit->setText( pipeline->get3DModelFilename().c_str() );

		//// Screen-based criteria
		//_screenBasedCriteriaGroupBox->setChecked( pipeline->getScreenBasedCriteria() );
		//_screenBasedCriteriaCoeffDoubleSpinBox->setValue( pipeline->getScreenBasedCriteriaCoefficient() );

		//// Material parameters
		//_materialAlphaCorrectionCoefficientDoubleSpinBox->setValue( pipeline->getMaterialAlphaCorrectionCoefficient() );

		//// Hypertexture parameters
		//_noiseFirstFrequencySpinBox->setValue( pipeline->getNoiseFirstFrequency() );
		//_noiseStrengthSpinBox->setValue( pipeline->getNoiseStrength() );
	}
}

/******************************************************************************
 * Slot called when the 3D model file button has been clicked (released)
 ******************************************************************************/
void CustomSectionEditor::on__3DModelToolButton_released()
{
	// Try to open 3D model
	QString filename = QFileDialog::getOpenFileName( this, "Choose a proxy geometry", QString( "." ), tr( "3D Model Files (*.obj *.dae *.3ds)" ) );
	if ( ! filename.isEmpty() )
	{
		_3DModelLineEdit->setText( filename );
		_3DModelLineEdit->setToolTip( filename );

		// Update the GigaVoxels pipeline
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

		SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
		assert( sampleCore != NULL );

		sampleCore->set3DModelFilename( filename.toStdString() );
	}
}

/******************************************************************************
 * Slot called when the screen based criteria group box state has changed
 ******************************************************************************/
void CustomSectionEditor::on__screenBasedCriteriaGroupBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	//sampleCore->setScreenBasedCriteria( pChecked );
}

/******************************************************************************
 * Slot called when sphere diameter coefficient value has changed
 ******************************************************************************/
void CustomSectionEditor::on__screenBasedCriteriaCoeffDoubleSpinBox_valueChanged( double pValue )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

   // sampleCore->setScreenBasedCriteriaCoefficient( pValue );
}

/******************************************************************************
 * Slot called when material's alpha correction coefficient has changed
 ******************************************************************************/
void CustomSectionEditor::on__materialAlphaCorrectionCoefficientDoubleSpinBox_valueChanged( double pValue )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

  //  sampleCore->setMaterialAlphaCorrectionCoefficient( pValue );
}

/******************************************************************************
 * Slot called when noise first frequency value has changed
 ******************************************************************************/
void CustomSectionEditor::on__noiseFirstFrequencySpinBox_valueChanged( double value )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

//	sampleCore->setNoiseFirstFrequency( value );
}

/******************************************************************************
 * Slot called when noise strength value has changed
 ******************************************************************************/
void CustomSectionEditor::on__noiseStrengthSpinBox_valueChanged( double value )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	//sampleCore->setNoiseStrength( value );
}
