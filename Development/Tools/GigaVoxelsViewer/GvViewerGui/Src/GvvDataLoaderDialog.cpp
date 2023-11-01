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

#include "GvvDataLoaderDialog.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QFileDialog>

// GigaSpace
#include <GvUtils/GsEnvironment.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerGui;

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
GvvDataLoaderDialog::GvvDataLoaderDialog( QWidget* pParent ) 
:	QDialog( pParent )
{
	//** Set the name
	setAccessibleName( qApp->translate( "GvvDataLoaderDialog", "Data Loader Dialog" ) );

	//** Initalizes the dialog
	setupUi( this );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvDataLoaderDialog::~GvvDataLoaderDialog()
{
}

/******************************************************************************
 * Initialize the default filename
 *
 * @param the default filename to load
 ******************************************************************************/
void GvvDataLoaderDialog::intialize( const char* pFilename )
{
	_3DModelLineEdit->setText( pFilename );
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
QString GvvDataLoaderDialog::get3DModelFilename() const
{
	//QString directory = _3DModelLineEdit->text();
	QString name = _3DModelLineEdit->text();
	
	QString filename = name;//directory + QDir::separator() + name;

	// TO DO
	// Test the existence...

	return filename;
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int GvvDataLoaderDialog::get3DModelResolution() const
{
	//unsigned int maxResolution =  ( 1 << _maxResolutionComboBox->currentIndex() ) * 8;

	return 0;//maxResolution;
}

/******************************************************************************
 * Slot called when 3D window background color tool button is released
 ******************************************************************************/
void GvvDataLoaderDialog::on__3DModelToolButton_released()
{
	QString defaultDirectory = GvUtils::GsEnvironment::getDataDir( GvUtils::GsEnvironment::eVoxelsDir ).c_str();
	QString file = QFileDialog::getOpenFileName( this, tr( "Select the XML file describing the 3D model"), defaultDirectory, "*.xml", NULL/*, QFileDialog::Option::DontUseNativeDialog*/ );
	if ( ! file.isEmpty() )
	{
		_3DModelLineEdit->setText( file );
	}
}
