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

#ifndef _GV_PRODUCTION_POLICIES_CONFIG_H_
#define _GV_PRODUCTION_POLICIES_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/**
 * Plugin Library 
 */

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVPRODUCTIONPOLICIES_MAKELIB	// Create a static library.
#		define GVPRODUCTIONPOLICIES_EXPORT
#		define GVPRODUCTIONPOLICIES_TEMPLATE_EXPORT
#	elif defined GVPRODUCTIONPOLICIES_USELIB	// Use a static library.
#		define GVPRODUCTIONPOLICIES_EXPORT
#		define GVPRODUCTIONPOLICIES_TEMPLATE_EXPORT

#	elif defined GVPRODUCTIONPOLICIES_MAKEDLL	// Create a DLL library.
#		define GVPRODUCTIONPOLICIES_EXPORT	__declspec(dllexport)
#		define GVPRODUCTIONPOLICIES_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVPRODUCTIONPOLICIES_EXPORT	__declspec(dllimport)
#		define GVPRODUCTIONPOLICIES_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVPRODUCTIONPOLICIES_MAKEDLL) || defined(GVPRODUCTIONPOLICIES_MAKELIB)
#		define GVPRODUCTIONPOLICIES_EXPORT
#		define GVPRODUCTIONPOLICIES_TEMPLATE_EXPORT
#	else
#		define GVPRODUCTIONPOLICIES_EXPORT
#		define GVPRODUCTIONPOLICIES_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
