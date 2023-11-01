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
#ifndef GVSPHERERAYTRACING_CONFIG_H
#define GVSPHERERAYTRACING_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVSPHERERAYTRACING_MAKELIB	// Create a static library.
#		define GVSPHERERAYTRACING_EXPORT
#		define GVSPHERERAYTRACING_TEMPLATE_EXPORT
#	elif defined GVSPHERERAYTRACING_USELIB	// Use a static library.
#		define GVSPHERERAYTRACING_EXPORT
#		define GVSPHERERAYTRACING_TEMPLATE_EXPORT

#	elif defined GVSPHERERAYTRACING_MAKEDLL	// Create a DLL library.
#		define GVSPHERERAYTRACING_EXPORT	__declspec(dllexport)
#		define GVSPHERERAYTRACING_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVSPHERERAYTRACING_EXPORT	__declspec(dllimport)
#		define GVSPHERERAYTRACING_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVSPHERERAYTRACING_MAKEDLL) || defined(GVSPHERERAYTRACING_MAKELIB)
#		define GVSPHERERAYTRACING_EXPORT
#		define GVSPHERERAYTRACING_TEMPLATE_EXPORT
#	else
#		define GVSPHERERAYTRACING_EXPORT
#		define GVSPHERERAYTRACING_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
