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
#ifndef _GV_SIMPLES_PHERE_CPU_CONFIG_H_
#define _GV_SIMPLES_PHERE_CPU_CONFIG_H_

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVSIMPLESPHERECPU_MAKELIB	// Create a static library.
#		define GVSIMPLESPHERECPU_EXPORT
#		define GVSIMPLESPHERECPU_TEMPLATE_EXPORT
#	elif defined GVSIMPLESPHERECPU_USELIB	// Use a static library.
#		define GVSIMPLESPHERECPU_EXPORT
#		define GVSIMPLESPHERECPU_TEMPLATE_EXPORT

#	elif defined GVSIMPLESPHERECPU_MAKEDLL	// Create a DLL library.
#		define GVSIMPLESPHERECPU_EXPORT	__declspec(dllexport)
#		define GVSIMPLESPHERECPU_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVSIMPLESPHERECPU_EXPORT	__declspec(dllimport)
#		define GVSIMPLESPHERECPU_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVSIMPLESPHERECPU_MAKEDLL) || defined(GVSIMPLESPHERECPU_MAKELIB)
#		define GVSIMPLESPHERECPU_EXPORT
#		define GVSIMPLESPHERECPU_TEMPLATE_EXPORT
#	else
#		define GVSIMPLESPHERECPU_EXPORT
#		define GVSIMPLESPHERECPU_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
