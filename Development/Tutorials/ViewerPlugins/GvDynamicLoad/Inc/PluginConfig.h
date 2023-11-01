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
#ifndef GVDYNAMICLOAD_CONFIG_H
#define GVDYNAMICLOAD_CONFIG_H

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
#	ifdef GVDYNAMICLOAD_MAKELIB	// Create a static library.
#		define GVDYNAMICLOAD_EXPORT
#		define GVDYNAMICLOAD_TEMPLATE_EXPORT
#	elif defined GVDYNAMICLOAD_USELIB	// Use a static library.
#		define GVDYNAMICLOAD_EXPORT
#		define GVDYNAMICLOAD_TEMPLATE_EXPORT
#	elif defined GVDYNAMICLOAD_MAKEDLL	// Create a DLL library.
#		define GVDYNAMICLOAD_EXPORT	__declspec(dllexport)
#		define GVDYNAMICLOAD_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVDYNAMICLOAD_EXPORT	__declspec(dllimport)
#		define GVDYNAMICLOAD_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVDYNAMICLOAD_MAKEDLL) || defined(GVDYNAMICLOAD_MAKELIB)
#		define GVDYNAMICLOAD_EXPORT
#		define GVDYNAMICLOAD_TEMPLATE_EXPORT
#	else
#		define GVDYNAMICLOAD_EXPORT
#		define GVDYNAMICLOAD_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
