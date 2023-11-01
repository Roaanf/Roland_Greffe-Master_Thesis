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

#ifndef GVVCONTEXTMANAGER_H
#define GVVCONTEXTMANAGER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

//Qt
#include <QObject>

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
namespace GvViewerCore
{
	class GvvBrowsable;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

class GVVIEWERGUI_EXPORT GvvContextManager : public QObject
{
	// Qt macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/********************************** ENUM **********************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Returns the unique instance of the context
	 */
	static GvvContextManager* get();

	/**
	 * Returns the current browsable
	 *
	 * @return a const pointer to the current browsable
	 */
	const GvViewerCore::GvvBrowsable* getCurrentBrowsable() const;

	/**
	 * Returns the current browsable
	 *
	 * @return a const to the current browsable
	 */
	GvViewerCore::GvvBrowsable* editCurrentBrowsable();

	/**
	 * Sets the current browsable
	 *
	 * @param pBrowsable specifies the current browsable to be set
	 */
	void setCurrentBrowsable( GvViewerCore::GvvBrowsable* pBrowsable );

	/******************************** SIGNALS *********************************/

signals:

	/**
	 * The signal is emitted when the current browsable changed
	 */
	void currentBrowsableChanged();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************** TYPEDEFS ********************************/
	
	/******************************* ATTRIBUTES *******************************/

	/**
     * The unique instance
     */
	static GvvContextManager* msInstance;

	/******************************** METHODS *********************************/
	
	/**
	 * Default constructor.
	 */
	GvvContextManager();
	
	/**
	 * Destructor.
	 */
	virtual ~GvvContextManager();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The current browsable
	 */
	GvViewerCore::GvvBrowsable* _currentBrowsable;
    
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvContextManager( const GvvContextManager& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvContextManager& operator=( const GvvContextManager& );

};

} // namespace GvViewerGui

/******************************************************************************
 ****************************** INLINE SECTION ********************************
 ******************************************************************************/

#include "GvvContextManager.inl"

#endif
