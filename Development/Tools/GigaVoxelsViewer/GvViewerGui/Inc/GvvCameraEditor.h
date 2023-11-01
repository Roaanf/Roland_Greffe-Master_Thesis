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

#ifndef _GVV_CAMERA_EDITOR_H_
#define _GVV_CAMERA_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvvQCameraEditor.h"

// Qt
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
 * @class GvvCameraEditor
 *
 * @brief The GvvCameraEditor class provides IHM to manipulate the camera
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvCameraEditor : public QWidget, public Ui::GvvQCameraEditor
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
	 * Default constructor
	 */
	GvvCameraEditor( QWidget* pParent = 0, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvCameraEditor();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

private slots:

	/**
	 * Slot called when camera field of view value has changed
	 */
	void on__fieldOfViewDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when camera scene radius value has changed
	 */
	void on__sceneRadiusDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when camera z near coefficient value has changed
	 */
	void on__zNearCoefficientDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when camera z clipping coefficient value has changed
	 */
	void on__zClippingCoefficientDoubleSpinBox_valueChanged( double pValue );

};

} // namespace GvViewerGui

#endif
