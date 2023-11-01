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

#ifndef PLUGIN_H
#define PLUGIN_H

// GvViewer
#include <GvvPluginInterface.h>

namespace GvViewerCore
{
    class GvvPluginManager;
}

class SampleCore;

class Plugin : public GvViewerCore::GvvPluginInterface
{

  public:

    // Constructors and 
    Plugin( GvViewerCore::GvvPluginManager& pManager );

	/**
     * Destructor
     */
    virtual ~Plugin();

    virtual const std::string& getName();
	
  private:

	  GvViewerCore::GvvPluginManager& mManager;

      std::string mName;

      std::string mExportName;

	  SampleCore* _pipeline;

	  void initialize();

	  void finalize();

};

#endif  // PLUGIN_H
