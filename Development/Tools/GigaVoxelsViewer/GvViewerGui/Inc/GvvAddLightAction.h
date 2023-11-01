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

#ifndef GVVADDLIGHTACTION_H
#define GVVADDLIGHTACTION_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvAction.h"

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

class GVVIEWERGUI_EXPORT GvvAddLightAction : public GvvAction
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
	GvvAddLightAction(	const QString& pFileName, 
						const QString& pText = QCoreApplication::translate( "GvvAddLightAction", cDefaultText ),
						//const QString& pIconName = QString::null, 
						const QString& pIconName = QString( "Light" ) ,
						bool pIsToggled = false );	

	/**
	 * Destructor.
	 */
	virtual ~GvvAddLightAction();

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

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvAddLightAction( const GvvAddLightAction& );

	/**
	 * Copy operator forbidden.
	 */
	GvvAddLightAction& operator=( const GvvAddLightAction& );

};

} // namespace GvViewerGui

#endif
