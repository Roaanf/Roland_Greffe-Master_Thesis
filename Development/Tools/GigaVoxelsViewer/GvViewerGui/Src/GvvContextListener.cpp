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

#include "GvvContextListener.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvContextManager.h"

// System
#include <cassert>

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
GvvContextListener::GvvContextListener( unsigned int pSignals )
:	_contextListenerProxy( this )
{
	//** Listen to the given signals
	listen( pSignals );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvContextListener::~GvvContextListener()
{
	disconnectAll();
}

/******************************************************************************
 * Listen the given document
 *
 * @param pSignals specifies the signals to be activated
 ******************************************************************************/
void GvvContextListener::listen( unsigned int pSignals )
{
	assert( GvvContextManager::get() != NULL );
	disconnectAll();
	connect( pSignals );
}

/******************************************************************************
 * Connect this plugin to the specified signal
 *
 * @param	pSignal specifies the signal to be activated
 ******************************************************************************/
void GvvContextListener::connect( unsigned int pSignals )
{
	//** HACK
	//** Disconnect desired signal(s) because we don't know
	//** if the listener is already connected to this/these desired signal(s)
	disconnect( pSignals );

	// Connection(s)
	if ( pSignals & eBrowsableChanged )
	{
		QObject::connect( GvvContextManager::get(), SIGNAL( currentBrowsableChanged( ) ), &_contextListenerProxy, SLOT( onCurrentBrowsableChanged( ) ) );
	}
}

/******************************************************************************
 * Disconnects all the slots
 ******************************************************************************/
void GvvContextListener::connectAll()
{
	connect( eAllSignals );
}

/******************************************************************************
 * Disconnectes this plugin to the specified signal
 *
 * @param	pSignal specifies the signal to be activated
 ******************************************************************************/
void GvvContextListener::disconnect( unsigned int pSignals )
{
	if ( pSignals & eBrowsableChanged )
	{
		QObject::disconnect( GvvContextManager::get(), SIGNAL( currentBrowsableChanged( ) ), &_contextListenerProxy, SLOT( onCurrentBrowsableChanged( ) ) );
	}
}

/******************************************************************************
 * Disconnects all the slots
 ******************************************************************************/
void GvvContextListener::disconnectAll()
{
	disconnect( eAllSignals );
}

/******************************************************************************
 * Returns whether this plugin is connected to the specified signal
 *
 * @param	pSignal specifies the signal to be checked
 *
 * @return	true if the signal is handled
 ******************************************************************************/
bool GvvContextListener::isConnected( GvvSignal pSignal )
{
	return false;
}

/******************************************************************************
 ****************************** SLOT DEFINITION *******************************
 ******************************************************************************/

/******************************************************************************
 * This slot is called when an Current Browsable Changed
 *
 ******************************************************************************/
void GvvContextListener::onCurrentBrowsableChanged( )
{
}
