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

#include "GvvEditorWindow.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvEditor.h"
#include "GvvSectionEditor.h"
#include "GvvBrowsable.h"
#include "GvvContextManager.h"

// System
#include <cassert>

// Qt
#include <QToolBox>
#include <QIcon>
#include <QDir>

// GigaSpace
#include <GvUtils/GsEnvironment.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

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
GvvEditorWindow::GvvEditorWindow( QWidget* pParent )
:	QWidget( pParent )
,	GvvContextListener()
,	_editorFactories()
,	_currentEditor( NULL )
{
	assert( pParent != NULL );

	//** Name this widget
	setAccessibleName( qApp->translate( "GvvEditorWindow","Editor Window") );

	//** Setups the Ui
	setupUi( this );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvEditorWindow::~GvvEditorWindow()
{
	// TO DO
	// destroy ...

	if ( tabWidget != NULL )
	{
		tabWidget->clear();
	}

	delete _currentEditor;
}

/******************************************************************************
 * Views and edits the editable properties
 *
 * @param	pEditable	The editable to view (can be NULL pointer)
 ******************************************************************************/
void GvvEditorWindow::edit( GvvBrowsable* pBrowsable )
{
	//** If the editable is not null tries to create the corresponding editor
	if ( pBrowsable != NULL )
	{
		//* Setups the name
		QString windowTitle = accessibleName();
		windowTitle += " [" + QString( pBrowsable->getName() ) + "]" ;

		parentWidget()->setWindowTitle( windowTitle );
				
		//-----------------------------------------------------------------------------
		// TO DO : check if the problem comes from the int instead of the uint
		//-----------------------------------------------------------------------------
		int key = (int)( qHash( QString( pBrowsable->getTypeName() ) ) );
		GvvEditorFactories::iterator itFactories = _editorFactories.find( key );
		if ( itFactories != _editorFactories.end() )
		{
			// Tries to create the editor using factory
			GvvEditor* editor = itFactories.value()( this, pBrowsable );
			if ( editor != NULL )
			{
				// Removes the current editor
				if ( _currentEditor != NULL )
				{
					for ( unsigned int i = 0; i < _currentEditor->getNbSections(); i++ )
					{
						tabWidget->removeTab( i );
					}
					delete _currentEditor;
					_currentEditor = NULL;
				}
				
				QToolBox* toolBox = new QToolBox();
				tabWidget->addTab( toolBox, tr( "Common" ) );
				for ( unsigned int i = 0; i < editor->getNbSections(); i++ )
				{
					GvvSectionEditor* sectionEditor = editor->getSectionEditor( i );
					assert( sectionEditor != NULL );
					if ( sectionEditor != NULL )
					{
						// Set icon
						QString iconRepository = GvUtils::GsEnvironment::getSystemDir( GvUtils::GsEnvironment::eResourcesDir ).c_str();
						QString iconfilename = iconRepository;
						iconfilename += QDir::separator();
						iconfilename += QString( "Icons" );
						iconfilename += QDir::separator();
						iconfilename += QString( "Pipeline.png" );
						if ( i == 0 )
						{
							QIcon* icon = new QIcon( iconfilename );
							if ( icon != NULL )
							{	
								// ...
							}
							toolBox->insertItem( 0, sectionEditor, *icon, sectionEditor->getName() );
						}
						else if ( i == 1 )
						{
							QIcon* icon = new QIcon( iconfilename );
							if ( icon != NULL )
							{	
								// ...
							}
							toolBox->insertItem( 1, sectionEditor, *icon, sectionEditor->getName() );
						}
						else if ( i == 2 )
						{
							QIcon* icon = new QIcon( iconfilename );
							if ( icon != NULL )
							{	
								// ...
							}
							toolBox->insertItem( 2, sectionEditor, *icon, sectionEditor->getName() );
						}
						else
						{
							tabWidget->addTab( sectionEditor, sectionEditor->getName() );
						}
					}
				}
				
				// Sets the created editor as the current
				_currentEditor = editor;

				// Populate editor
				editor->populate( pBrowsable );
			}
		}
	}
	else
	{
		clear();
	}

    //** If the editable is null or no editor corresponds, clear the editor
	parentWidget()->setWindowTitle( accessibleName() );	
}

/******************************************************************************
 * Clears this editor
 ******************************************************************************/
void GvvEditorWindow::clear()
{
	//* Clears the current editor
	if ( _currentEditor != NULL )
	{
		// Removes all the pages, but does not delete them.
		// Calling this function is equivalent to calling removeTab() until the tab widget is empty.
		tabWidget->clear();

		delete _currentEditor;
		_currentEditor = NULL;
	}
}

/******************************************************************************
 * Registers the specified editor builder
 *
 * @param pBuilder the editor builder to be registered
 ******************************************************************************/
void GvvEditorWindow::registerEditorFactory( const QString& pEditableType, GvvEditorFactory* pEditorFactory )
{
	//** Computes the key
	//-----------------------------------------------------------------------------
	// TO DO : check if the problem comes from the int instead of the uint
	//-----------------------------------------------------------------------------
	int key = (int)( qHash( pEditableType ) );

	//** Retrieves the menu of the given name
	GvvEditorFactories::iterator itFactories = _editorFactories.find( key );
	if ( itFactories != _editorFactories.end() )
	{
		// Error
		// TO DO
		// Handle this...
		// ...
		assert( false );
	}

	_editorFactories.insert( key, pEditorFactory );
}

/******************************************************************************
 * Unregisters the specified editor builder
 *
 * @param pBuilder the editor builder to be unregistered
 ******************************************************************************/
void GvvEditorWindow::unregisterEditorFactory( const QString& pEditableType )
{
	//** Computes the key
	//-----------------------------------------------------------------------------
	// TO DO : check if the problem comes from the int instead of the uint
	//-----------------------------------------------------------------------------
	int key = (int)( qHash( pEditableType ) );

	//** Retrieves the menu of the given name
	GvvEditorFactories::iterator itFactories = _editorFactories.find( key );
	if ( itFactories != _editorFactories.end() )
	{
		_editorFactories.remove( key );
	}
	else
	{
		// Just for test...
		assert( false );
	}	
}

/******************************************************************************
 * This slot is called when the current editable changed
 ******************************************************************************/
void GvvEditorWindow::onCurrentBrowsableChanged()
{
	edit( GvvContextManager::get()->editCurrentBrowsable() );
}
