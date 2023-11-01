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

#include "GvvAboutDialog.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"

// Qt
#include <QMessageBox>

// STL
#include <string>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// STL
using namespace std;

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
GvvAboutDialog::GvvAboutDialog( QWidget* pParent ) 
:	QDialog( pParent )
{
	//** Set the name
	setAccessibleName( qApp->translate( "GvvAboutDialog", "About Dialog" ) );

	//** Initalizes the dialog
	setupUi( this );

	//** Populate the dialog
	//mAboutPixmap->setPixmap( GvvEnvironment::get().getSystemFilePath( GvvEnvironment::eAboutScreenFile ) );

	//mVersionLabel->setText( QString::number(GVV_VERSION_MAJOR) + "." +
	//					QString::number(GVV_VERSION_MINOR) );

	//** Populate the plug-ins list
	//** - empty the table
	//mPluginsList->clear();

	//** - layout the table
	//mPluginsList->verticalHeader()->hide();
	//mPluginsList->verticalHeader()->setDefaultSectionSize( 18 );

	//** - set table size
	//mPluginsList->setColumnCount( 1 );
	//mPluginsList->setRowCount( GvvPluginManager::get().getNbPlugins() );

	//** - set table column headers
	//QStringList lLabels;
	//lLabels.append( qApp->translate( "GvvAboutDialog", "Installed plugins" ) );
	//mPluginsList->setHorizontalHeaderLabels( lLabels );
	//mPluginsList->horizontalHeader()->setResizeMode ( QHeaderView::Stretch );

	////** - fill the table
	//for
	//	( int lInd = 0; lInd < GvvPluginManager::get().getNbPlugins(); lInd++ )
	//{
	//	const GvvPlugin* lPlugin = GvvPluginManager::get().getPlugin( lInd );

	//	//** plug-ins name
	//	QTableWidgetItem* lNameItem = new QTableWidgetItem( lPlugin->getName() );
	//	mPluginsList->setItem( lInd, 0, lNameItem );
	//	lNameItem->setFlags( Qt::ItemIsEnabled );
	//}
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvAboutDialog::~GvvAboutDialog()
{
}

/******************************************************************************
 * Slot called when Credits push button is released.
 ******************************************************************************/
void GvvAboutDialog::on__creditsPushButton_released()
{
	QMessageBox::information( this, tr( "Credits" ), tr( "Not yet implemented..." ) );
}

/******************************************************************************
 * Slot called when License push button is released.
 ******************************************************************************/
void GvvAboutDialog::on__licensePushButton_released()
{
	QMessageBox::information( this, tr( "License" ), tr( "Not yet implemented..." ) );
}
