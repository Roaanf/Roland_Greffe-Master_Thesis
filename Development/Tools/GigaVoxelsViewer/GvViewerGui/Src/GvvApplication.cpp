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

#include "GvvApplication.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvMainWindow.h"
#include "GvvPluginManager.h"
#include "GvvEnvironment.h"

// Qt
//#include <QSplashScreen>
#include <QImageReader>

// System
#include <cassert>

// STL
#include <iostream>

// GigaSpace
#include <GvUtils/GsEnvironment.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * The unique instance of the singleton.
 */
GvvApplication* GvvApplication::msInstance = NULL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Initialize the application
 *
 * @param pArgc Number of arguments
 * @param pArgv List of arguments
******************************************************************************/
void GvvApplication::initialize( int& pArgc, char** pArgv )
{
	assert( msInstance == NULL );
	if ( msInstance == NULL )
    {
		// Initialize application
        msInstance = new GvvApplication( pArgc, pArgv );

		//// Splash screen
		//QPixmap pixmap( "J:\\Projects\\Inria\\GigaVoxels\\Development\\Library\\doc\\GigaVoxelsLogo_div2.png" );
		//QSplashScreen* splash = new QSplashScreen( pixmap, Qt::WindowStaysOnTopHint );
		//splash->show();

		//// Loading some items
		//splash->showMessage( "Loaded modules" );

		//qApp->processEvents();

		//app.processEvents();
		//...
		//	QMainWindow window;
		//window.show();
		//splash.finish(&window);

		// Load the settings
		/*bool statusOK = */GvUtils::GsEnvironment::initializeSettings();
		// TODO : check settings
		// ...

		// Load the settings
		/*bool statusOK = */GvvEnvironment::initializeSettings();
		// TODO : check settings
		// ...

		// Initialize main window
		msInstance->initializeMainWindow();
	}	
}

/******************************************************************************
 * Finalize the application
******************************************************************************/
void GvvApplication::finalize()
{
	delete msInstance;
	msInstance = NULL;
}

/******************************************************************************
 * Get the application
 *
 * @return The aplication
 ******************************************************************************/
GvvApplication& GvvApplication::get()
{
	assert( msInstance != NULL );
	
	return *msInstance;
}

/******************************************************************************
 * Constructor
 *
 * @param pArgc Number of arguments
 * @param pArgv List of arguments
 ******************************************************************************/
GvvApplication::GvvApplication( int& pArgc, char** pArgv )
:	QApplication( pArgc, pArgv )
,	mGPUComputingInitialized( false )
,	mMainWindow( NULL )
{
	// Initialize GigaSpace environment
	GvUtils::GsEnvironment::initialize( QCoreApplication::applicationDirPath().toLatin1().constData() );

	// Initialize Viewer environment
	GvvEnvironment::initialize( QCoreApplication::applicationDirPath().toLatin1().constData() );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvApplication::~GvvApplication()
{
	delete mMainWindow;
	mMainWindow = NULL;
}

/******************************************************************************
 * Execute the application
 ******************************************************************************/
int GvvApplication::execute()
{	
	mMainWindow->show();

#ifdef _DEBUG
	std::cout << "\nQt supported image formats :" << std::endl;
	QList< QByteArray > supportedImageFormats = QImageReader::supportedImageFormats();
	for ( int i = 0; i < supportedImageFormats.size(); i++ )
	{
		std::cout << "- " <<supportedImageFormats.at( i ).constData() << std::endl;
	}
#endif

	// Main Qt's event loop
	int result = exec();
	
	// Remove any plugin
	GvvPluginManager::get().unloadAll();

	// Destroy the main window
	delete mMainWindow;
	mMainWindow = NULL;

	return result;
}

/******************************************************************************
 * Initialize the main wiondow
 ******************************************************************************/
void GvvApplication::initializeMainWindow()
{
	mMainWindow = new GvvMainWindow();
	if ( mMainWindow != NULL )
	{
		mMainWindow->initialize();
	}
}

/******************************************************************************
 * Get the main window
 *
 * return The main window
 ******************************************************************************/
GvvMainWindow* GvvApplication::getMainWindow()
{
	return mMainWindow;
}

/******************************************************************************
 *
 ******************************************************************************/
bool GvvApplication::isGPUComputingInitialized() const
{
	return mGPUComputingInitialized;
}

/******************************************************************************
 *
 ******************************************************************************/
void GvvApplication::setGPUComputingInitialized( bool pFlag )
{
	mGPUComputingInitialized = pFlag;
}
