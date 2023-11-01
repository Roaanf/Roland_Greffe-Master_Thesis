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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include <cassert>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvViewerGui
{

/******************************************************************************
 * Returns whether this manager is empty
 *
 * @return	true if this manager is empty
 ******************************************************************************/
inline bool GvvActionManager::isEmpty() const
{
	return getNbActions() == 0;
}

/******************************************************************************
 * Returns the number of elements
 *
 * @return	the number of elements
 ******************************************************************************/
inline unsigned int GvvActionManager::getNbActions() const
{
	return static_cast< unsigned int >( mActions.size() );
}

/******************************************************************************
 * Returns the i-th element
 *
 * @param	pIndex	specifies the index of the desired element
 *
 * @return	a const pointer to the pIndex-th element
 ******************************************************************************/
inline const GvvAction* GvvActionManager::getAction( unsigned int pIndex ) const
{
	assert( pIndex < getNbActions() );
	return mActions[ pIndex ];
}	

/******************************************************************************
 * Returns the i-th element
 *
 * @param	pIndex	specifies the index of the desired element
 *
 * @return	a pointer to the pIndex-th element
 ******************************************************************************/
inline GvvAction* GvvActionManager::editAction( unsigned int pIndex )
{
	return const_cast< GvvAction* >( getAction( pIndex ) );
}

/******************************************************************************
 * Returns the element represented by the specified name
 *
 * @param	pName specifies the name of the desired element
 *
 * @return	a const pointer to the element or null if not found
 ******************************************************************************/
inline const GvvAction* GvvActionManager::getAction( const QString& pName ) const
{
	unsigned int lIndex = findAction( pName );
	if
		( lIndex != -1 )
	{
		return getAction( lIndex );
	}
	return NULL;
}

/******************************************************************************
 * Returns the element represented by the specified name
 *
 * @param	pName specifies the name of the desired element
 *
 * @return	a pointer to the element or null if not found
 ******************************************************************************/
inline GvvAction* GvvActionManager::editAction( const QString& pName )
{
	return const_cast< GvvAction* >( getAction( pName ) );
}

} // namespace GvViewerGui
