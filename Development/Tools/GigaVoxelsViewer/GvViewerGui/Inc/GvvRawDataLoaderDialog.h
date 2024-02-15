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

#ifndef _GVV_RAW_DATA_LOADER_DIALOG_H_
#define _GVV_RAW_DATA_LOADER_DIALOG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvQRawDataLoaderDialog.h"

// Qt
#include <QDialog>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

class GVVIEWERGUI_EXPORT GvvRawDataLoaderDialog : public QDialog, public Ui::GvQRawDataLoaderDialog
{
	// Qt macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvRawDataLoaderDialog( QWidget* pParent = NULL );

	/**
	 * Destructor.
	 */
	virtual ~GvvRawDataLoaderDialog();

	/**
	 * Get the 3D model filename to load
	 *
	 * @return the 3D model filename to load
	 */
	QString get3DModelFilename() const;

	/**
	 * Get the file mode
	 *
	 * @return the file mode
	 */
	unsigned int getModelFileMode() const;

	/**
	 * Get the data type
	 *
	 * @return the data type mode
	 */
	unsigned int getModelDataType() const;

	/**
	 * Get the 3D model resolution
	 *
	 * @return the 3D model resolution
	 */
	unsigned int getBrickSize() const;

	/**
	 * Get the 3D model resolution
	 *
	 * @return the 3D model resolution
	 */
	unsigned int getTrueX() const;

	/**
	 * Get the 3D model resolution
	 *
	 * @return the 3D model resolution
	 */
	unsigned int getTrueY() const;

	/**
	 * Get the 3D model resolution
	 *
	 * @return the 3D model resolution
	 */
	unsigned int getTrueZ() const;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/********************************* SLOTS **********************************/
		
protected slots:

	/**
	 * Slot called when 3D model directory tool button is released
	 */
	void on__3DModelDirectoryToolButton_released();

	/**
	 * Slot called when License push button is released
	 */
	void on__maxResolutionComboBox_currentIndexChanged( const QString& pText );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvRawDataLoaderDialog( const GvvRawDataLoaderDialog& );

	/**
	 * Copy operator forbidden.
	 */
	GvvRawDataLoaderDialog& operator=( const GvvRawDataLoaderDialog& );

	/********************************* SLOTS **********************************/

};

} // namespace GvViewerGui

#endif
