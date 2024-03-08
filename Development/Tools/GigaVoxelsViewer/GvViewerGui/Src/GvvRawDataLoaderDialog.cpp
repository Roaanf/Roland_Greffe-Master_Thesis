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

#include "GvvRawDataLoaderDialog.h"

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
GvvRawDataLoaderDialog::GvvRawDataLoaderDialog( QWidget* pParent ) 
:	QDialog( pParent )
{
	//** Set the name
	setAccessibleName( qApp->translate( "GvvRawDataLoaderDialog", "RAW Data Loader Dialog" ) );

	//** Initalizes the dialog
	setupUi( this );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvRawDataLoaderDialog::~GvvRawDataLoaderDialog()
{
}

/******************************************************************************
 * Slot called when 3D window background color tool button is released
 ******************************************************************************/
void GvvRawDataLoaderDialog::on__3DModelDirectoryToolButton_released()
{
	QString defaultDirectory = GvUtils::GsEnvironment::getDataDir( GvUtils::GsEnvironment::eVoxelsDir ).c_str();
	defaultDirectory += QDir::separator();
	defaultDirectory += QString( "Raw" );
	QString filename = QFileDialog::getOpenFileName( this, "Choose a file", defaultDirectory, tr( "GigaVoxels XML Files (*.xml);;RAW data file (*.raw)" ) );
	if ( ! filename.isEmpty() )
	{
		_3DModelFilenameLineEdit->setText( filename );
	}
}

/******************************************************************************
 * Slot called when License push button is released.
 ******************************************************************************/
void GvvRawDataLoaderDialog::on__maxResolutionComboBox_currentIndexChanged( const QString& pText )
{
	unsigned int maxResolution = pText.toUInt() - 1;
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
QString GvvRawDataLoaderDialog::get3DModelFilename() const
{
	return _3DModelFilenameLineEdit->text();
}

/******************************************************************************
 * Get the file mode
 *
 * @return the file mode
 ******************************************************************************/
unsigned int GvvRawDataLoaderDialog::getModelFileMode() const
{
	// TODO : add a combo box (binary / text)
	return 0;
}

/******************************************************************************
 * Get the data type
 *
 * @return the data type mode
 ******************************************************************************/
unsigned int GvvRawDataLoaderDialog::getModelDataType() const
{
	return _3DModelDataTypeComboBox->currentIndex();
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int GvvRawDataLoaderDialog::getRadius() const
{
	unsigned int radius = (unsigned int)_radius->value();

	return radius;
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int GvvRawDataLoaderDialog::getTrueX() const
{
	unsigned int trueX = _trueXSpinBox->value();

	return trueX;
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int GvvRawDataLoaderDialog::getTrueY() const
{
	unsigned int trueY = _trueYSpinBox->value();

	return trueY;
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int GvvRawDataLoaderDialog::getTrueZ() const
{
	unsigned int trueZ = _trueZSpinBox->value();

	return trueZ;
}
