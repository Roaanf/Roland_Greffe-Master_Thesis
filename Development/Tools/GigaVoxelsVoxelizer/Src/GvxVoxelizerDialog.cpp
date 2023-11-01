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

#include "GvxVoxelizerDialog.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QMessageBox>
#include <QFileDialog>
#include <QString>

// STL
#include <string>

// GigaSpace
#include <GvUtils/GsEnvironment.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvVoxelizer
using namespace Gvx;

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
 * Default constructor.
 ******************************************************************************/
GvxVoxelizerDialog::GvxVoxelizerDialog( QWidget* pParent ) 
:	QDialog( pParent )
,	_fileName()
,	_maxResolution( 512 )
,	_isGenerateNormalsOn( false )
,	_brickWidth( 8 )
,	_dataType( 0 )
,   _filterType( 0)
,   _nbFilterOperation(0)
,   _normals(false)
{

	//** Set the name
	setAccessibleName( qApp->translate( "GvxVoxelizerDialog", "Voxelizer Dialog" ) );

	//** Initalizes the dialog
	setupUi( this );

	// Update level of resolution
	unsigned int maxResolution = _maxResolutionComboBox->currentText().toUInt() - 1;
	//_nbLevelsLineEdit->setText( QString::number( maxResolution / 8 - 1) );
	_maxResolution = maxResolution;
	_nbLevelsLineEdit->setText( QString::number( ( 1 << maxResolution ) * 8 ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvxVoxelizerDialog::~GvxVoxelizerDialog()
{
}

/******************************************************************************
 * Slot called when Credits push button is released.
 ******************************************************************************/
void GvxVoxelizerDialog::on__3DModelToolButton_released()
{
	//QMessageBox::information( this, tr( "Voxelizer - START" ), tr( "Not yet implemented..." ) );

	QString defaultDirectory = GvUtils::GsEnvironment::getDataDir( GvUtils::GsEnvironment::e3DModelsDir ).c_str();
	QString fileName = QFileDialog::getOpenFileName( this, "Choose a file", defaultDirectory, tr( "3D Model Files (*.obj *.dae *.3ds)" ) );
	if ( ! fileName.isEmpty() )
	{
		_3DModelLineEdit->setText( fileName );

		_fileName = fileName;
	}
}

/******************************************************************************
 * Slot called when License push button is released.
 ******************************************************************************/
void GvxVoxelizerDialog::on__maxResolutionComboBox_currentIndexChanged( const QString& pText )
{
	//QMessageBox::information( this, tr( "Voxelizer - STOP" ), tr( "Not yet implemented..." ) );

	//unsigned int maxResolution = pText.toUInt();
	//_nbLevelsLineEdit->setText(	QString::number( maxResolution / 8 - 1 ) );

	//_maxResolution = maxResolution / 8 - 1;
	
	unsigned int maxResolution = pText.toUInt() - 1;
	_maxResolution = maxResolution;
	_nbLevelsLineEdit->setText( QString::number( ( 1 << maxResolution ) * 8 ) );
}

/******************************************************************************
 * Override QDialog accept() method
 * (Hides the modal dialog and sets the result code to Accepted)
 ******************************************************************************/
void GvxVoxelizerDialog::accept()
{
	// Retrieve all information
	_fileName = _3DModelLineEdit->text();
	if ( _fileName.isEmpty() )
	{
		// TO DO
		// Handle this case
		// ...
	}
	_maxResolution = _maxResolutionComboBox->currentText().toUInt() - 1;
	_isGenerateNormalsOn = _generateNormalsCheckBox->isChecked();
	_brickWidth = _brickWidthSpinBox->value();
	_dataType = _dataTypeComboBox->currentIndex();
	
	// Call base class implementation
	QDialog::accept();
}

	

/******************************************************************************
 * Set the Filter type
 * 0 = mean
 * 1 = gaussian
 * 2 = laplacian
 ******************************************************************************/
void GvxVoxelizerDialog::on__filterTypeComboBox_currentIndexChanged (int pValue) 
{
	_filterType = pValue;
}

/******************************************************************************
 * Set the Number of filter iteration
 ******************************************************************************/
void GvxVoxelizerDialog::on__filterIterationsSpinBox_valueChanged( int pValue)
{
	_nbFilterOperation = pValue;
}

/*****************************************************************************
 * Set whether or not we produce the normal field
 *****************************************************************************/
void GvxVoxelizerDialog::on__generateNormalsCheckBox_toggled (bool pChecked)
{
	_normals = pChecked;
}