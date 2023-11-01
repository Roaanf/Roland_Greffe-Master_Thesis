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

#ifndef _GVV_TRANSFER_FUNCTION_EDITOR_H_
#define _GVV_TRANSFER_FUNCTION_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvPipelineInterface.h"
#include "GvvContextListener.h"

// Qtfe
#include "Qtfe.h"

// Qt
#include <QObject>
#include <QWidget>

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
	class GvvPipelineInterface;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * ...
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvTransferFunctionEditor : public QObject, public GvvContextListener
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
	 * Constructor
	 */
	GvvTransferFunctionEditor( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvTransferFunctionEditor();

	/**
	 * ...
	 */
	void show();

	///**
	// * ...
	// *
	// * @param pPipeline ...
	// * @param pFlag ...
	// */
	//void connect( GvViewerCore::GvvPipelineInterface* pPipeline, bool pFlag );

	/**
	 * Get the transfer function.
	 *
	 * @return the transfer function
	 */
	Qtfe* getTransferFunction();

	/**
	 * Set the pipeline.
	 *
	 * @param pPipeline The pipeline
	 */
	void setPipeline( GvViewerCore::GvvPipelineInterface* pPipeline );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The cache editor
	 */
	Qtfe* _editor;

	/**
	 *
	 */
	GvViewerCore::GvvPipelineInterface* _pipeline;

	/******************************** METHODS *********************************/

	/**
	 * This slot is called when the current browsable is changed
	 */
	virtual void onCurrentBrowsableChanged();

	/********************************* SLOTS **********************************/

protected slots:

	/**
	 * Slot called when at least one canal changed
	 */
	void onFunctionChanged();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvTransferFunctionEditor( const GvvTransferFunctionEditor& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvTransferFunctionEditor& operator=( const GvvTransferFunctionEditor& );
	
};

} // namespace GvViewerGui

#endif
