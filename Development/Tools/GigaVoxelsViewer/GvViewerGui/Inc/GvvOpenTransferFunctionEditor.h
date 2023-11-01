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

#ifndef GVVOPENTRANSFERFUNCTIONEDITOR_H
#define GVVOPENTRANSFERFUNCTIONEDITOR_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvAction.h"
#include "GvvContextListener.h"

// Qt
#include <QCoreApplication>

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

class GVVIEWERGUI_EXPORT GvvOpenTransferFunctionEditor : public GvvAction, public GvvContextListener
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The unique name of the action
	 */
	static const QString cName;
	
	/**
	 * The default text assigned to the action
	 */
	static const char* cDefaultText;

	/******************************** METHODS *********************************/

	/**
	 * Constructs an action dependant of the applications project
	 *
	 * @param	pFileName	specifies the filename of the manual
	 * @param	pText		specifies the descriptive text of this action
	 * @param	pIconName	specifies the name of the icon for this action located in the icons application path
	 *							Does nothing if the string is empty. A full file path can also be given.
	 * @param	pIsToggled	specified if the action is toggled or not
	 */
	GvvOpenTransferFunctionEditor(	const QString& pFileName, 
									const QString& pText = QCoreApplication::translate( "GvvOpenTransferFunctionEditor", cDefaultText ),
									//const QString& pIconName = QString::null, 
									const QString& pIconName = QString( "TransferFunction" ) ,
									bool pIsToggled = false );	

	/**
	 * Destructor.
	 */
	virtual ~GvvOpenTransferFunctionEditor();

	/**
	 * Overwrites the execute method
	 */
	virtual void execute();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The manual filename
	 */
	QString mFileName;

	/******************************** METHODS *********************************/

	/**
	 * This slot is called when the current browsable is changed
	 */
	virtual void onCurrentBrowsableChanged();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvOpenTransferFunctionEditor( const GvvOpenTransferFunctionEditor& );

	/**
	 * Copy operator forbidden.
	 */
	GvvOpenTransferFunctionEditor& operator=( const GvvOpenTransferFunctionEditor& );

};

} // namespace GvViewerGui

#endif
