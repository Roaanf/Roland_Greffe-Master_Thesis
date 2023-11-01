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

#ifndef GVVAPPICATION_H
#define GVVAPPICATION_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
 
// Qt
#include <QApplication>

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
	class GvvMainWindow;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvQApplication
 *
 * @brief The GvQApplication class provides ...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvApplication : public QApplication
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
	 * Initialize the application
	 *
	 * @param pArgc Number of arguments
	 * @param pArgv List of arguments
	 */
	static void initialize( int& pArgc, char** pArgv );

	/**
	 * Finalize the application
	 */
	static void finalize();

	/**
	 * Get the application
	 *
	 * return The application
	 */
	static GvvApplication& get();

	/**
	 * Execute the application
	 */
	int execute();

	/**
	 * Get the main window
	 *
	 * return The main window
	 */
	GvvMainWindow* getMainWindow();

	/**
	 *
	 */
	bool isGPUComputingInitialized() const;

	/**
	 *
	 */
	void setGPUComputingInitialized( bool pFlag );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 *
	 */
	bool mGPUComputingInitialized;
		
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pArgc Number of arguments
	 * @param pArgv List of arguments
	 */
	GvvApplication( int& pArgc, char** pArgv );

	/**
	 * Destructor
	 */
	virtual ~GvvApplication();

	/**
	 * Initialize the main wiondow
	 */
	void initializeMainWindow();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
     * The unique instance
     */
    static GvvApplication* msInstance;

	/**
	 * The main window
	 */
	GvvMainWindow* mMainWindow;
	
	/******************************** METHODS *********************************/

};

} // namespace GvViewerGui

#endif
