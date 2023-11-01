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

#ifndef _GVV_DATA_LOADER_DIALOG_H_
#define _GVV_DATA_LOADER_DIALOG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvQDataLoaderDialog.h"

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

class GVVIEWERGUI_EXPORT GvvDataLoaderDialog : public QDialog, public Ui::GvQDataLoaderDialog
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
	GvvDataLoaderDialog( QWidget* pParent = NULL );

	/**
	 * Destructor.
	 */
	virtual ~GvvDataLoaderDialog();

	/**
	 * Initialize the default filename
	 *
	 * @param the default filename to load
	 */
	void intialize( const char* pFilename );

	/**
	 * Get the 3D model filename to load
	 *
	 * @return the 3D model filename to load
	 */
	QString get3DModelFilename() const;

	/**
	 * Get the 3D model resolution
	 *
	 * @return the 3D model resolution
	 */
	unsigned int get3DModelResolution() const;

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
	void on__3DModelToolButton_released();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvDataLoaderDialog( const GvvDataLoaderDialog& );

	/**
	 * Copy operator forbidden.
	 */
	GvvDataLoaderDialog& operator=( const GvvDataLoaderDialog& );

	/********************************* SLOTS **********************************/

};

} // namespace GvViewerGui

#endif
