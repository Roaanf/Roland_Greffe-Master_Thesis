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

#include "GvvOpenTransferFunctionEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "GvvContextManager.h"
#include "GvvPipelineInterface.h"
#include "GvvTransferFunctionEditor.h"

// Qtfe
#include "Qtfe.h"

// Qt
#include <QDir>
#include <QFile>
#include <QProcess>
#include <QDesktopServices>
#include <QUrl>
#include <QMessageBox>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/
/**
 * The unique name of the action
 */
const QString GvvOpenTransferFunctionEditor::cName = "openTransferFunctionEditor";

/**
 * The default text assigned to the action
 */
const char* GvvOpenTransferFunctionEditor::cDefaultText = QT_TR_NOOP( "Open Transfer Function Editor" );

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructs an action dependant of the applications project
 *
 * @param	pFileName	specifies the filename of the manual
 * @param	pText		specifies the descriptive text of this action
 * @param	pIconName	specifies the name of the icon for this action located in the icons application path
 *							Does nothing if the string is empty. A full file path can also be given.
 * @param	pIsToggled	specified if the action is toggled or not
 ******************************************************************************/
GvvOpenTransferFunctionEditor::GvvOpenTransferFunctionEditor( const QString& pFileName, 
										const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
,	GvvContextListener(  )
,	mFileName( pFileName )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvOpenTransferFunctionEditor", "Open transfer function editor" ) );
//	setShortcut( qApp->translate( "GvvOpenTransferFunctionEditor", "F1" ) );

	//  Disabled by default
	setDisabled( true );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvOpenTransferFunctionEditor::~GvvOpenTransferFunctionEditor()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvOpenTransferFunctionEditor::execute()
{
	if ( GvvApplication::get().getMainWindow()->getTransferFunctionEditor() != NULL )
	{
		GvvApplication::get().getMainWindow()->getTransferFunctionEditor()->show();
	}
}

/******************************************************************************
 * This slot is called when the current editable changed
 ******************************************************************************/
void GvvOpenTransferFunctionEditor::onCurrentBrowsableChanged()
{
	const GvvPipelineInterface* pipeline = dynamic_cast< const GvvPipelineInterface* >( GvvContextManager::get()->getCurrentBrowsable() );
	setEnabled(  pipeline != NULL );
}
