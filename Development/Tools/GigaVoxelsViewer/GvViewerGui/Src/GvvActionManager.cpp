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

#include "GvvActionManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <algorithm>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * The unique action manager
 */
GvvActionManager* GvvActionManager::msInstance = NULL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Default constructor.
 ******************************************************************************/
GvvActionManager::GvvActionManager( bool pDeleteActions )
:	mActions()
,	mDeleteActions( pDeleteActions )
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvActionManager::~GvvActionManager()
{
	if ( mDeleteActions )
	{
		deleteActions();
	}
	else if	( ! isEmpty() )
	{
		//GVV_ERROR( "All elements registered within the manager were not correctly destroyed before its own destruction." );
	}
}

/******************************************************************************
 * Returns the action manager
 ******************************************************************************/
GvvActionManager& GvvActionManager::get()
{
	if ( msInstance == NULL )
	{
		msInstance = new GvvActionManager();
	}
	assert( msInstance != NULL );
	return *msInstance;
}

/******************************************************************************
 * Registers the specified element to this manager
 *
 * @param	pManageable	specifies the element to be registered
 ******************************************************************************/
void GvvActionManager::registerAction( GvvAction* pAction )
{
	assert( pAction != NULL );

	//** Insert the element
	mActions.push_back( pAction );
}

/******************************************************************************
 * Unregisters the specified element from this manager
 *
 * @param	pManageable	specifies the element to be unregistered
 ******************************************************************************/
void GvvActionManager::unregisterAction( GvvAction* pAction )
{
	assert( pAction != NULL );

	//** Remove the element
	GvvActionVector::iterator lIt = std::find( mActions.begin(), mActions.end(), pAction );
	if ( lIt != mActions.end() )
	{
		mActions.erase( lIt );
	}
}

/******************************************************************************
 * Searchs for the specified element
 *
 * @param	pManageable	specifies the element to be found
 *
 * @return	the index of the element or -1 if not found
 ******************************************************************************/
unsigned int GvvActionManager::findAction( const GvvAction* pAction ) const
{
	//** Find the action
	GvvActionVector::const_iterator lIt = std::find( mActions.begin(), mActions.end(), pAction );
	if ( lIt != mActions.end() )
	{
		return lIt - mActions.begin();
	}
	return -1;
}

/******************************************************************************
 * Searchs for the element with the specified name
 *
 * @param	pName	specifies the name of the element to be found
 *
 * @return	the index of the element or -1 if not found
 ******************************************************************************/
unsigned int GvvActionManager::findAction( const QString& pName ) const
{
	GvvActionVector::const_iterator lIt = mActions.begin();
	for ( ; lIt != mActions.end(); ++lIt )
	{
		//** Search the action
		if ( (*lIt)->getName() == pName )
		{
			return lIt - mActions.begin();
		}
	}
	return -1;
}

/******************************************************************************
 * Deletes the specified element from memory
 *
 * @param	pManageable	specifies the element to delete
 ******************************************************************************/
void GvvActionManager::deleteAction( GvvAction* pAction )
{
	assert( pAction != NULL );
	delete pAction;
}

/******************************************************************************
 * Deletes all the elements from memory and clears the manager
 ******************************************************************************/
void GvvActionManager::deleteActions()
{
	GvvActionVector::reverse_iterator lIt = mActions.rbegin();
	for	( ; lIt != mActions.rend() ; ++lIt )
	{
		deleteAction( *lIt );
	}
	mActions.clear();
}
