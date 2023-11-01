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

#ifndef _INSPECTOR_VIEW_H_
#define _INSPECTOR_VIEW_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QWidget>

// Project
#include "UI_GvvQInspectorView.h"

// GigaVoxels
#include <GvCore/Array3D.h>
#include <GvCore/Array3DGPULinear.h>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Project
class SampleCore;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class GvvCacheEditor
 *
 * @brief The GvvCacheEditor class provides ...
 *
 * ...
 */
class InspectorView : public QWidget, public Ui::GvvQInspectorView
{

	// Qt Macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * pParent ...
	 * pFlags ...
	 */
	InspectorView( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~InspectorView();

	/**
	 * Initialize this editor with the specified GigaVoxels pipeline
	 *
	 * @param pPipeline ...
	 */
	void initialize( SampleCore* pPipeline );

	/**
	 * Populates this editor with the specified GigaVoxels pipeline
	 *
	 * @param pPipeline ...
	 */
	void populate( SampleCore* pPipeline );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	GvCore::Array3D< uint >* _dataStructureChildArray;
	GvCore::Array3D< uint >* _dataStructureDataArray;

	GvCore::Array3DGPULinear< uint >* _nodeCacheTimeStampList;
	thrust::device_vector< uint >* _nodeCacheElementAddressList;
	uint _nodeCacheNbUnusedElements;

	GvCore::Array3DGPULinear< uint >* _brickCacheTimeStampList;
	thrust::device_vector< uint >* _brickCacheElementAddressList;
	uint _brickCacheNbUnusedElements;

	/******************************** METHODS *********************************/
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	InspectorView( const InspectorView& );

	/**
	 * Copy operator forbidden.
	 */
	InspectorView& operator=( const InspectorView& );

	/********************************* SLOTS **********************************/

private slots:

};

#endif
