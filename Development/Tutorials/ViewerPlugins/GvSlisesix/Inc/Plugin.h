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

#ifndef _PLUGIN_H_
#define _PLUGIN_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include <GvvPluginInterface.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Viewer
namespace GvViewerCore
{
    class GvvPluginManager;
}

// Project
class SampleCore;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class CustomEditor
 *
 * @brief The CustomEditor class provides a custom editor to this GigaVoxels
 * pipeline effect.
 *
 * This editor has a static creator function used by the factory class "GvvEditorWindow"
 * to create the associated editor (@see GvvEditorWindow::registerEditorFactory())
 */
class Plugin : public GvViewerCore::GvvPluginInterface
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:
	
	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pManager the singleton Plugin Manager
	 */
	Plugin( GvViewerCore::GvvPluginManager& pManager );

	/**
     * Destructor
     */
    virtual ~Plugin();

	/**
     * Get the plugin name
	 *
	 * @return the plugin name
     */
    virtual const std::string& getName();

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
	
	/**
	 * Reference on the singleton Plugin Manager
	 */
	GvViewerCore::GvvPluginManager& _manager;

	/**
	 * Name
	 */
	std::string _name;

	/**
	 * Export name
	 */
	std::string _exportName;

	/**
	 * Reference on a GigaVoxels pipeline
	 */
	SampleCore* _pipeline;

	/******************************** METHODS *********************************/

	/**
	 * Initialize the plugin
	 */
	void initialize();

	/**
	 * Finalize the plugin
	 */
	void finalize();

};

#endif  // _PLUGIN_H_
