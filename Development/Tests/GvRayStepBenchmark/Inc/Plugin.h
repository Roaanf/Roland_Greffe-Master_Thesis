/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * BSD 3-Clause License:
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the organization nor the names  of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
