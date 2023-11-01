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
#ifndef GVENVIRONMENTMAPPING_CONFIG_H
#define GVENVIRONMENTMAPPING_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVENVIRONMENTMAPPING_MAKELIB	// Create a static library.
#		define GVENVIRONMENTMAPPING_EXPORT
#		define GVENVIRONMENTMAPPING_TEMPLATE_EXPORT
#	elif defined GVENVIRONMENTMAPPING_USELIB	// Use a static library.
#		define GVENVIRONMENTMAPPING_EXPORT
#		define GVENVIRONMENTMAPPING_TEMPLATE_EXPORT

#	elif defined GVENVIRONMENTMAPPING_MAKEDLL	// Create a DLL library.
#		define GVENVIRONMENTMAPPING_EXPORT	__declspec(dllexport)
#		define GVENVIRONMENTMAPPING_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVENVIRONMENTMAPPING_EXPORT	__declspec(dllimport)
#		define GVENVIRONMENTMAPPING_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVENVIRONMENTMAPPING_MAKEDLL) || defined(GVENVIRONMENTMAPPING_MAKELIB)
#		define GVENVIRONMENTMAPPING_EXPORT
#		define GVENVIRONMENTMAPPING_TEMPLATE_EXPORT
#	else
#		define GVENVIRONMENTMAPPING_EXPORT
#		define GVENVIRONMENTMAPPING_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
