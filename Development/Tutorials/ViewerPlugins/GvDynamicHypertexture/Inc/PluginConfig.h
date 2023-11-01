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

#ifndef _GV_DYNAMIC_HYPERTEXTURE_CONFIG_H_
#define _GV_DYNAMIC_HYPERTEXTURE_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/**
 * Plugin Library 
 */

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVDYNAMICHYPERTEXTURE_MAKELIB	// Create a static library.
#		define GVDYNAMICHYPERTEXTURE_EXPORT
#		define GVDYNAMICHYPERTEXTURE_TEMPLATE_EXPORT
#	elif defined GVDYNAMICHYPERTEXTURE_USELIB	// Use a static library.
#		define GVDYNAMICHYPERTEXTURE_EXPORT
#		define GVDYNAMICHYPERTEXTURE_TEMPLATE_EXPORT

#	elif defined GVDYNAMICHYPERTEXTURE_MAKEDLL	// Create a DLL library.
#		define GVDYNAMICHYPERTEXTURE_EXPORT	__declspec(dllexport)
#		define GVDYNAMICHYPERTEXTURE_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVDYNAMICHYPERTEXTURE_EXPORT	__declspec(dllimport)
#		define GVDYNAMICHYPERTEXTURE_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVDYNAMICHYPERTEXTURE_MAKEDLL) || defined(GVDYNAMICHYPERTEXTURE_MAKELIB)
#		define GVDYNAMICHYPERTEXTURE_EXPORT
#		define GVDYNAMICHYPERTEXTURE_TEMPLATE_EXPORT
#	else
#		define GVDYNAMICHYPERTEXTURE_EXPORT
#		define GVDYNAMICHYPERTEXTURE_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
