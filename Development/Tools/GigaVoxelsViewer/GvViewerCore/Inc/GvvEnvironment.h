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

#ifndef _GVV_ENVIRONMENT_H_
#define _GVV_ENVIRONMENT_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"

// STL
#include <string>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerCore
{

/** 
 * @class GvvEnvironment
 *
 * @brief The GvvEnvironment class provides access to a GigaSpace Viewer project environment
 *
 * Environment deals with useful application directories (data, plugins, resources...),
 * files, etc...
 */
class GVVIEWERCORE_EXPORT GvvEnvironment
{

	/**************************************************************************
     ***************************** PUBLIC SECTION *****************************
     **************************************************************************/

public:

	/******************************* INNER TYPES ******************************/

	/**
	 * Specifies the different system directories
	 */
	enum GsSystemDir
	{
		eSettingsDir,
		ePluginsDir,
		eResourcesDir,
		eManualsDir,
		eLicensesDir,
		eToolsDir,
		eNbSystemDirs
	};

	/**
	 * Specifies the different system directories
	 */
	enum GsDataDir
	{
		e3DModelsDir,
		eShadersDir,
		eSkyBoxDir,
		eTerrainDir,
		eTransferFunctionsDir,
		eVideosDir,
		eVoxelsDir,
		eNbDataDirs
	};

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Specifies the names of the different system directories
	 */
	static const char* cSystemDirNames[ eNbSystemDirs ];

	/**
	 * Specifies the names of the different data directories
	 */
	static const char* cDataDirNames[ eNbDataDirs ];

	/**
	 * Data info
	 */
	static const char* _cDataElementName;
	static const char* _cDataPathAttributeName;

	/**
	 * User profile info
	 */
	static const char* _cUserProfileElementName;
	static const char* _cUserProfilePathAttributeName;

	/**
	 * Demo info
	 */
	static const char* _cDemoElementName;
	static const char* _cDemoPathAttributeName;

	/******************************** METHODS *********************************/
	
	/**
	 * Initializes the environment
	 *
	 * @param pApplicationFilePath the application file path
	 */
	static void initialize( const std::string pApplicationFilePath );

	/**
	 * Initialize the settings
	 *
	 * @return true if it succeeds
	 */
	static bool initializeSettings();

	/**
	 * Returns the application file path
	 *
	 * @return the application file path
	 */
	static const std::string& getApplicationFilePath();

	/**
	 * Returns the application directory
	 *
	 * @return the application directory
	 */
	static const std::string& getApplicationDirectory();

	/**
	 * Returns the data path
	 *
	 * @return the data path
	 */
	static const std::string& getDataPath();

	/**
	 * Returns the user profile path
	 *
	 * @return the user profile path
	 */
	static const std::string& getUserProfilePath();

	/**
	 * Returns the demo path
	 *
	 * @return the demo path
	 */
	static const std::string& getDemoPath();

	/**
	 * Returns the specified system directory
	 *
	 * @return the directory
	 */
	static std::string getSystemDir( GvvEnvironment::GsSystemDir pDirName );

	/**
	 * Returns the specified data directory
	 *
	 * @return the directory
	 */
	static std::string getDataDir( GvvEnvironment::GsDataDir pDirName );

    /**************************************************************************
	 **************************** PROTECTED SECTION ***************************
     **************************************************************************/

protected:

    /******************************** TYPEDEFS ********************************/

    /******************************* ATTRIBUTES *******************************/

	/**
	 * The application file path
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	static std::string _sApplicationFilePath;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * The application directory
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	static std::string _sApplicationDirectory;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * The data directory
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	static std::string _sDataDirectory;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * The user profile directory
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	static std::string _sUserProfileDirectory;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * The user profile directory
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	static std::string _sDemoDirectory;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/******************************** METHODS *********************************/

	/**
	 * Expand a directory name
	 *
	 * @param pDirectory the name to expand
	 */
	static std::string expand( const char* pDirectory );

	/**************************************************************************
     ***************************** PRIVATE SECTION ****************************
     **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

}

/******************************************************************************
 ****************************** INLINE SECTION ********************************
 ******************************************************************************/

//#include "GvvEnvironment.inl"

#endif
