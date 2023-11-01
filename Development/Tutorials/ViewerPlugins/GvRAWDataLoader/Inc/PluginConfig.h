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
#ifndef GVRAWDATALOADER_CONFIG_H
#define GVRAWDATALOADER_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVRAWDATALOADER_MAKELIB	// Create a static library.
#		define GVRAWDATALOADER_EXPORT
#		define GVRAWDATALOADER_TEMPLATE_EXPORT
#	elif defined GVRAWDATALOADER_USELIB	// Use a static library.
#		define GVRAWDATALOADER_EXPORT
#		define GVRAWDATALOADER_TEMPLATE_EXPORT
#	elif defined GVRAWDATALOADER_MAKEDLL	// Create a DLL library.
#		define GVRAWDATALOADER_EXPORT	__declspec(dllexport)
#		define GVRAWDATALOADER_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVRAWDATALOADER_EXPORT	__declspec(dllimport)
#		define GVRAWDATALOADER_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVRAWDATALOADER_MAKEDLL) || defined(GVRAWDATALOADER_MAKELIB)
#		define GVRAWDATALOADER_EXPORT
#		define GVRAWDATALOADER_TEMPLATE_EXPORT
#	else
#		define GVRAWDATALOADER_EXPORT
#		define GVRAWDATALOADER_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
