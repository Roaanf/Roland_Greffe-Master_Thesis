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

#ifndef GVVMAINWINDOW_H
#define GVVMAINWINDOW_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvQMainWindow.h"

// Qt
#include <QMainWindow>

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
namespace GvViewerGui
{
	class Gvv3DWindow;
	class GvvCacheEditor;
	class GvvToolBar;
	class GvvAnalyzerToolBar;
	class GvvCaptureToolBar;
	class GvvPipelineBrowser;
	class GvvEditorWindow;
	class GvvDeviceBrowser;
	class GvvTransferFunctionEditor;
	class GvvGLSceneBrowser;
	class GvvGLSLSourceEditor;
	class GvvCUDASourceEditor;
	class GvvPlotView;
	class GvvCacheUsageView;
	class GvvTimeBudgetMonitoringEditor;
	class GvvCameraEditor;
}

//// Qt
//class QGroupBox;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvQMainWindow
 *
 * @brief The GvQMainWindow class provides ...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvMainWindow : public QMainWindow
{
	// Qt Macro
	Q_OBJECT

	/**
	 * ...
	 */
	friend class Gvv3DWindow;

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	///**
	// * The 3D window
	// */
	//QGroupBox* _3DViewGroupBox;
	
	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvMainWindow( QWidget *parent = 0, Qt::WindowFlags flags = 0 );

	/**
	 * Destructor.
	 */
	virtual ~GvvMainWindow();

	/**
	 * Initialize.
	 */
	void initialize();

	/**
	 * Get the 3D window.
	 *
	 * return the 3D window
	 */
	Gvv3DWindow* get3DWindow();

	/**
	 * Get the pipeline editor.
	 *
	 * return the pipeline editor
	 */
	//GvvCacheEditor* getPipelineEditor();
	//GvvPipelineEditor* getPipelineEditor();

	/**
	 * Get the pipeline browser.
	 *
	 * return the pipeline browser
	 */
	GvvPipelineBrowser* getPipelineBrowser();

	/**
	 * Get the pipeline editor.
	 *
	 * return the pipeline editor
	 */
	GvvEditorWindow* getEditorWindow();
	
	/**
	 * Get the transfer function editor.
	 *
	 * return the transfer function editor
	 */
	GvvTransferFunctionEditor* getTransferFunctionEditor();

	/**
	 * Get the programmable shader browser.
	 *
	 * return the programmable shader browser.
	 */
	GvvGLSLSourceEditor* getGLSLourceEditor();

	/**
	 * Get the scene browser.
	 *
	 * return the scene browser
	 */
	GvvCUDASourceEditor* getCUDASourceEditor();

	/**
	 * Get the cache plot viewer.
	 *
	 * return the cache plot viewer
	 */
	GvvPlotView* getCachePlotView();

	/**
	 * Get the cache usage view
	 *
	 * return the cache usage view
	 */
	GvvCacheUsageView* getCacheUsageView();

	/**
	 * Get the time budget monitoring view
	 *
	 * returnthe time budget monitoring view
	 */
	GvvTimeBudgetMonitoringEditor* getTimeBudgetMonitoringView();

	/**
	 * Get the Camera editor
	 *
	 * return the Camera editor
	 */
	GvvCameraEditor* getCameraEditor();
				
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The 3D window
	 */
	Gvv3DWindow* _3DWindow;

	/**
	 * The cache editor
	 */
	//GvvCacheEditor* _cacheEditor;
	//GvvPipelineEditor* _pipelineCacheEditor;
	
	/**
	 * The tool bar
	 */
	GvvToolBar* _toolBar;

	/**
	 * The analyzer tool bar
	 */
	GvvAnalyzerToolBar* _analyzerToolBar;

	/**
	 * The capture tool bar
	 */
	GvvCaptureToolBar* _captureToolBar;

	/**
	 * The pipeline browser
	 */
	GvvPipelineBrowser* _pipelineBrowser;

	/**
	 * The device browser
	 */
	GvvDeviceBrowser* _deviceBrowser;
	
	/**
	 * The transfer function editor
	 */
	GvvTransferFunctionEditor* _transferFunctionEditor;

	/**
	 * The editor window
	 */
	GvvEditorWindow* _editorWindow;

	/**
	 * The scene browser
	 */
	GvvGLSceneBrowser* _sceneBrowser;

	/**
	 * Shader source editor
	 */
	GvvGLSLSourceEditor* _GLSLSourceEditor;

	/**
	 * Shader source editor
	 */
	GvvCUDASourceEditor* _CUDASourceEditor;

	/**
	 * Cache plot viewer
	 */
	GvvPlotView* _cachePlotView;

	/**
	 * Cache usage view
	 */
	GvvCacheUsageView* _cacheUsageView;

	/**
	 * Time budget monitoring view
	 */
	GvvTimeBudgetMonitoringEditor* _timeBudgetMonitoringView;

	/**
	 * Camera editor
	 */
	GvvCameraEditor* _cameraEditor;
			
	/******************************** METHODS *********************************/
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * Ui designer class
	 */
	Ui::GvQMainWindow mUi;

	/**
	 * the opened filename
	 */
	QString mFilename;
	
	/******************************** METHODS *********************************/

private slots:

	/**
	 * Open file action
	 */
	void onActionOpenFile();

	/**
	 * Exit action
	 */
	void onActionExit();

	/**
	 * Edit preferences action
	 */
	void onActionEditPreferences();

	/**
	 * Display full screen action
	 */
	void onActionFullScreen();

	/**
	 * Display help action
	 */
	void onActionHelp();

	/**
	 * Open about dialog action
	 */
	void onActionAbout();

};

} // namespace GvViewerGui

#endif

