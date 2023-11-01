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

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvViewerGui
{

/******************************************************************************
 * Returns the current browsable
 ******************************************************************************/
inline const GvViewerCore::GvvBrowsable* GvvContextManager::getCurrentBrowsable() const
{
	return _currentBrowsable;
}

/******************************************************************************
 * Returns the current browsable
 ******************************************************************************/
inline GvViewerCore::GvvBrowsable* GvvContextManager::editCurrentBrowsable()
{
	return _currentBrowsable;
}

} // namespace GvViewerGui
