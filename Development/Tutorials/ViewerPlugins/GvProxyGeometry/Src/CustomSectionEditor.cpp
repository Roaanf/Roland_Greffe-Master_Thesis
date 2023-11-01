/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
	setName( tr( "Proxy Geometry" ) );
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
		//// Hypertexture parameters
		//_noiseFirstFrequencySpinBox->setValue( pipeline->getNoiseFirstFrequency() );
		//_noiseStrengthSpinBox->setValue( pipeline->getNoiseStrength() );
		
		// Light parameters
		float x;
		float y;
		float z;
		pipeline->getLightPosition( x, y, z );
		_lightXDoubleSpinBox->setValue( x );
		_lightYDoubleSpinBox->setValue( y );
		_lightZDoubleSpinBox->setValue( z );

		//// Shader parameters
		//_voxelSizeMultiplierDoubleSpinBox->setValue( pipeline->getVoxelSizeMultiplier() );

		_proxyGeometryOptimisationCheckBox->setChecked( pipeline->hasProxyGeometryOptimisation() );
	}
}

///******************************************************************************
// * Slot called when noise first frequency value has changed
// ******************************************************************************/
//void CustomSectionEditor::on__noiseFirstFrequencySpinBox_valueChanged( double value )
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
//	sampleCore->setNoiseFirstFrequency( value );
//}
//
///******************************************************************************
// * Slot called when noise strength value has changed
// ******************************************************************************/
//void CustomSectionEditor::on__noiseStrengthSpinBox_valueChanged( double value )
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
//	sampleCore->setNoiseStrength( value );
//}

/******************************************************************************
 * Slot called when light position X value has changed
 ******************************************************************************/
void CustomSectionEditor::on__lightXDoubleSpinBox_valueChanged( double value )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	float x;
	float y;
	float z;
	sampleCore->getLightPosition( x, y, z );
	x = value;
	sampleCore->setLightPosition( x, y, z );
}

/******************************************************************************
 * Slot called when light position Y value has changed
 ******************************************************************************/
void CustomSectionEditor::on__lightYDoubleSpinBox_valueChanged( double value )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	float x;
	float y;
	float z;
	sampleCore->getLightPosition( x, y, z );
	y = value;
	sampleCore->setLightPosition( x, y, z );
}

/******************************************************************************
 * Slot called when light position Y value has changed
 ******************************************************************************/
void CustomSectionEditor::on__lightZDoubleSpinBox_valueChanged( double value )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	float x;
	float y;
	float z;
	sampleCore->getLightPosition( x, y, z );
	z = value;
	sampleCore->setLightPosition( x, y, z );
}

///******************************************************************************
// *  * Slot called when shader's voxel size multiplier value has changed
// ******************************************************************************/
//void CustomSectionEditor::on__voxelSizeMultiplierDoubleSpinBox_valueChanged( double value )
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
//	sampleCore->setVoxelSizeMultiplier( value );
//}

/******************************************************************************
 * Slot called when light position Y value has changed
 ******************************************************************************/
void CustomSectionEditor::on__proxyGeometryOptimisationCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setProxyGeometryOptimisation( pChecked );
}
