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

#include "GvvContextManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowsable.h"

// STL
#include <cassert>

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
 * The unique instance of the context
 */
GvvContextManager* GvvContextManager::msInstance = NULL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Returns the context
 ******************************************************************************/
GvvContextManager* GvvContextManager::get()
{
	if ( msInstance == NULL )
	{
		msInstance = new GvvContextManager();
	}
	assert( msInstance != NULL );
	return msInstance;
}

/******************************************************************************
 * Default constructor.
 ******************************************************************************/
GvvContextManager::GvvContextManager()
:	QObject()
,	_currentBrowsable( NULL )
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvContextManager::~GvvContextManager()
{
}

/******************************************************************************
 * Sets the current browsable
 ******************************************************************************/
void GvvContextManager::setCurrentBrowsable( GvvBrowsable* pBrowsable )
{
	//** Sets the current browsable
	if ( _currentBrowsable != pBrowsable )
	{
		_currentBrowsable = pBrowsable;

		//** Emits the corresponding signal
		emit currentBrowsableChanged();
	}
}
