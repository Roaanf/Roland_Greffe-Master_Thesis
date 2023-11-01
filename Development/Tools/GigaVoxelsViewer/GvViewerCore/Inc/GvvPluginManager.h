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

#ifndef GVVPLUGINMANAGER_H
#define GVVPLUGINMANAGER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"

#include <string>
#include <vector>

#ifdef _WIN32
	#include <windows.h>
#else
#endif

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
	class GvvPluginInterface;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerCore
{

/**
 * GvPluginManager
 */
class GVVIEWERCORE_EXPORT GvvPluginManager
{

    /**************************************************************************
     ***************************** PUBLIC SECTION *****************************
     **************************************************************************/

public:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /******************************** METHODS *********************************/

    /**
     * getInstance()
     *
     * @return
     */
    static GvvPluginManager& get();

    /**
     * loadPlugins()
     *
     * @param pDir
     */
    void loadPlugins( const std::string& pDir );

	/**
     * getFilenames()
     *
     * @param pFilename
     */
    bool loadPlugin( const std::string& pFilename );

    /**
     * unloadAll()
     */
    void unloadAll();

    /**
     * getNumPlugins()
     *
     * @return
     */
    size_t getNbPlugins() const;

    /**
     * getPlugin()
     *
     * @param pIndex
     *
     * @return
     */
    GvvPluginInterface* getPlugin( int pIndex );

   /**************************************************************************
    **************************** PROTECTED SECTION ***************************
    **************************************************************************/

protected:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /******************************** METHODS *********************************/

    /**************************************************************************
     ***************************** PRIVATE SECTION ****************************
     **************************************************************************/

private:

    /******************************* INNER TYPES *******************************/

    /**
     * Structure PluginInfo
     */
    struct PluginInfo
    {
        /**
         * Pointer on the plugin
         */
        GvvPluginInterface* mPlugin;

		/**
         * Handle to library
         */
#ifdef _WIN32
		HINSTANCE mHandle;
#else
        void* mHandle;
#endif
    };

    /******************************* ATTRIBUTES *******************************/

    /**
     * The unique instance
     */
    static GvvPluginManager* msInstance;

    /**
     * List of plugins
     */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
    std::vector< PluginInfo > mPlugins;
#if defined _MSC_VER
#pragma warning( pop )
#endif

    /******************************** METHODS *********************************/

    /**
     * PluginManager()
     */
    GvvPluginManager();

    /**
     * getFilenames()
     *
     * @param pDir
     * @param pFilenames
     */
    void getFilenames( const std::string& pDir, std::vector< std::string >& pFilenames ) const;
	
};

} // namespace GvViewerCore

namespace GvViewerCore
{

	/**
	 * Definition of GVV_CREATE_PLUGIN as a function's pointer
	 */
	 typedef GvvPluginInterface* (* GVV_CREATE_PLUGIN)( GvvPluginManager& pManager );

} // namespace GvViewerCore

#endif
