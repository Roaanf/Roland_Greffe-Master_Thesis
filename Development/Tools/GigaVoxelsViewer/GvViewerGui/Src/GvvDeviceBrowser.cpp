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

#include "GvvDeviceBrowser.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowserItem.h"
#include "GvvBrowsable.h"
#include "GvvContextMenu.h"
#include "GvvDeviceInterface.h"

// Qt
#include <QContextMenuEvent>
#include <QTreeWidget>

// System
#include <cassert>

// GigaSpace
#include <GsCompute/GsDeviceManager.h>
#include <GsCompute/GsDevice.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

// GigaSpace
using namespace GsCompute;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Default constructor.
 ******************************************************************************/
GvvDeviceBrowser::GvvDeviceBrowser( QWidget* pParent ) 
:	GvvBrowser( pParent )
{
	//// TEST --------------------------------------------
	//GvvBrowsable* light = new GvvBrowsable();
	//GvvBrowserItem* lightItem = createItem( light );
	//addTopLevelItem( lightItem );
	//// TEST --------------------------------------------

	//// attention, peut-être déjà fait ailleurs...
	GsCompute::GsDeviceManager::get().initialize();
	
	for ( int i = 0; i < GsCompute::GsDeviceManager::get().getNbDevices(); i++ )
	{
		const GsCompute::GsDevice* device = GsCompute::GsDeviceManager::get().getDevice( i );

		GvvDeviceInterface* deviceInterface = new GvvDeviceInterface();

		GvvBrowserItem* deviceItem = createItem( deviceInterface );

		deviceItem->setText( 0, device->_name.c_str() );
		deviceItem->setToolTip( 0, QString( "Compute capability : " ) + QString::number( device->mProperties._computeCapabilityMajor ) + QString( "." ) + QString::number( device->mProperties._computeCapabilityMinor ) );

		addTopLevelItem( deviceItem );
	}
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvDeviceBrowser::~GvvDeviceBrowser()
{
}
