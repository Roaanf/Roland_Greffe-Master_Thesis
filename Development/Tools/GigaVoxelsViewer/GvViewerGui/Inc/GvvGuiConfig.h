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
 * @defgroup GvViewerGui
 */
#ifndef GVVIEWERGUICONFIG_H
#define GVVIEWERGUICONFIG_H

//*** GvViewerGui Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVVIEWERGUI_MAKELIB	// Create a static library.
#		define GVVIEWERGUI_EXPORT
#		define GVVIEWERGUI_TEMPLATE_EXPORT
#	elif defined GVVIEWERGUI_USELIB	// Use a static library.
#		define GVVIEWERGUI_EXPORT
#		define GVVIEWERGUI_TEMPLATE_EXPORT
#	elif defined GVVIEWERGUI_MAKEDLL	// Create a DLL library.
#		define GVVIEWERGUI_EXPORT	__declspec(dllexport)
#		define GVVIEWERGUI_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVVIEWERGUI_EXPORT	__declspec(dllimport)
#		define GVVIEWERGUI_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVVIEWERGUI_MAKEDLL) || defined(GVVIEWERGUI_MAKELIB)
#		define GVVIEWERGUI_EXPORT
#		define GVVIEWERGUI_TEMPLATE_EXPORT
#	else
#		define GVVIEWERGUI_EXPORT
#		define GVVIEWERGUI_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
