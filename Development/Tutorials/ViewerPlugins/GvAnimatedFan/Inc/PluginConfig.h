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
#ifndef GVANIMATEDFAN_CONFIG_H
#define GVANIMATEDFAN_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVANIMATEDFAN_MAKELIB	// Create a static library.
#		define GVANIMATEDFAN_EXPORT
#		define GVANIMATEDFAN_TEMPLATE_EXPORT
#	elif defined GVANIMATEDFAN_USELIB	// Use a static library.
#		define GVANIMATEDFAN_EXPORT
#		define GVANIMATEDFAN_TEMPLATE_EXPORT
#	elif defined GVANIMATEDFAN_MAKEDLL	// Create a DLL library.
#		define GVANIMATEDFAN_EXPORT	__declspec(dllexport)
#		define GVANIMATEDFAN_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVANIMATEDFAN_EXPORT	__declspec(dllimport)
#		define GVANIMATEDFAN_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVANIMATEDFAN_MAKEDLL) || defined(GVANIMATEDFAN_MAKELIB)
#		define GVANIMATEDFAN_EXPORT
#		define GVANIMATEDFAN_TEMPLATE_EXPORT
#	else
#		define GVANIMATEDFAN_EXPORT
#		define GVANIMATEDFAN_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
