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

#ifndef GVV3DWindow_H
#define GVV3DWindow_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qt
#include <QtCore/QObject>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GvViewer
namespace GvViewerGui
{
	class GvvPipelineInterfaceViewer;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvQMainWindow
 *
 * @brief The GvQMainWindow class provides ...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT Gvv3DWindow : public QObject
{
	// Qt Macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Default constructor
	 */
	Gvv3DWindow( QWidget *parent = 0, Qt::WindowFlags flags = 0 );

	/**
	 * Destructor
	 */
	virtual ~Gvv3DWindow();

	/**
	 * Get the pipeline viewer
	 *
	 * return The pipeline viewer
	 */
	GvvPipelineInterfaceViewer* getPipelineViewer();

	/**
	 * ...
	 */
	void addViewer();
	void removeViewer();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The pipelijne viewer
	 */
	GvvPipelineInterfaceViewer* mPipelineViewer;

	/******************************** METHODS *********************************/
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
		
	/******************************** METHODS *********************************/

};

} // namespace GvViewerGui

#endif

