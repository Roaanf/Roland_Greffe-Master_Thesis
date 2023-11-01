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

#ifndef GVVREMOVELEMENTACTION_H
#define GVVREMOVELEMENTACTION_H

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

class GVVIEWERGUI_EXPORT GvvRemoveBrowsableAction : public GvvAction
{

	/**************************************************************************
	 ***************************** FRIEND SECTION *****************************
	 **************************************************************************/

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
	 * Constructs the action
	 *
	 * @param	pText specifies the descriptive text of this action
	 * @param	pIconName specifies the name of the icon for this action located in the icons application path
	 *					Does nothing if the string is empty. A full file path can also be given.
	 */
	GvvRemoveBrowsableAction(	const QString& pText = QCoreApplication::translate("GvvRemoveBrowsableAction", cDefaultText ), 
							const QString& pIconName = QString::null ); 

	/**
	 * Destructor.
	 */
	virtual ~GvvRemoveBrowsableAction();

	/**
	 * Overwrites the execute method
	 */
	virtual void execute();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

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
	GvvRemoveBrowsableAction( const GvvRemoveBrowsableAction& );

	/**
	 * Copy operator forbidden.
	 */
	GvvRemoveBrowsableAction& operator=( const GvvRemoveBrowsableAction& );

};

} // namespace GvViewerGui

#endif
