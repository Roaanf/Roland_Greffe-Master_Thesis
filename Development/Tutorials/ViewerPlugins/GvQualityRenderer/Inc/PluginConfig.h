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
#ifndef GVSIMPLESPHERE_CONFIG_H
#define GVSIMPLESPHERE_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVSIMPLESPHERE_MAKELIB	// Create a static library.
#		define GVSIMPLESPHERE_EXPORT
#		define GVSIMPLESPHERE_TEMPLATE_EXPORT
#	elif defined GVSIMPLESPHERE_USELIB	// Use a static library.
#		define GVSIMPLESPHERE_EXPORT
#		define GVSIMPLESPHERE_TEMPLATE_EXPORT

#	elif defined GVSIMPLESPHERE_MAKEDLL	// Create a DLL library.
#		define GVSIMPLESPHERE_EXPORT	__declspec(dllexport)
#		define GVSIMPLESPHERE_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVSIMPLESPHERE_EXPORT	__declspec(dllimport)
#		define GVSIMPLESPHERE_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVSIMPLESPHERE_MAKEDLL) || defined(GVSIMPLESPHERE_MAKELIB)
#		define GVSIMPLESPHERE_EXPORT
#		define GVSIMPLESPHERE_TEMPLATE_EXPORT
#	else
#		define GVSIMPLESPHERE_EXPORT
#		define GVSIMPLESPHERE_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
