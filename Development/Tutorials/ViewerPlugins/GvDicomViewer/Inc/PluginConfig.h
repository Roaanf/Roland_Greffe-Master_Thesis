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

/**
 * ...
 */
#ifndef GVDICOMVIEWER_CONFIG_H
#define GVDICOMVIEWER_CONFIG_H

//*** Plugin Library 

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/**
 * User type definitions
 */
//#define _DATA_HAS_NORMALS_

/******************************************************************************
 ************************** LIBRARY CONFIGURATION *****************************
 ******************************************************************************/

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVDICOMVIEWER_MAKELIB	// Create a static library.
#		define GVDICOMVIEWER_EXPORT
#		define GVDICOMVIEWER_TEMPLATE_EXPORT
#	elif defined GVDICOMVIEWER_USELIB	// Use a static library.
#		define GVDICOMVIEWER_EXPORT
#		define GVDICOMVIEWER_TEMPLATE_EXPORT
#	elif defined GVDICOMVIEWER_MAKEDLL	// Create a DLL library.
#		define GVDICOMVIEWER_EXPORT	__declspec(dllexport)
#		define GVDICOMVIEWER_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVDICOMVIEWER_EXPORT	__declspec(dllimport)
#		define GVDICOMVIEWER_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVDICOMVIEWER_MAKEDLL) || defined(GVDICOMVIEWER_MAKELIB)
#		define GVDICOMVIEWER_EXPORT
#		define GVDICOMVIEWER_TEMPLATE_EXPORT
#	else
#		define GVDICOMVIEWER_EXPORT
#		define GVDICOMVIEWER_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
