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

#ifndef GVVEDITORWINDOW_H
#define GVVEDITORWINDOW_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvvQEditorWidget.h"
#include "GvvContextListener.h"

// Qt
#include <QWidget>
#include <QHash>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GvViewer
namespace GvViewerCore
{
	class GvvBrowsable;
}
namespace GvViewerGui
{
	class GvvEditor;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{
	/**
	 * Factory method used to create editors.
	 * Each editor need to provide a static method exactly as this function pointer.
	 */
	typedef GvvEditor* (GvvEditorFactory)( QWidget*, GvViewerCore::GvvBrowsable* );
}

namespace GvViewerGui
{

/**
 * ...
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvEditorWindow : public QWidget, public Ui::GvvQEditorWidget, public GvvContextListener
{
	// Qt macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Default constructor.
	 */
	GvvEditorWindow( QWidget* pParent = NULL );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvEditorWindow();

	/**
	 * Edits the specified editable entity
	 *
	 * @param	pEditable	the entity to be edited
	 */
	void edit( GvViewerCore::GvvBrowsable* pBrowsable );

	/**
	 * Clears this editor
	 */
	void clear();

	/**
	 * Registers the specified editor builder
	 *
	 * @param pBuilder the editor builder to be registered
	 */
	void registerEditorFactory( const QString& pEditableType, GvvEditorFactory* pEditorFactory );

	/**
	 * Unregisters the specified editor builder
	 *
	 * @param pBuilder the editor builder to be unregistered
	 */
	void unregisterEditorFactory( const QString& pEditableType );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/
	
	/**
	 * Type definition of factory methods used to create editors
	 */
	typedef QHash< unsigned int, GvvEditorFactory* > GvvEditorFactories;
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The factory methods used to create editors
	 */
	GvvEditorFactories _editorFactories;

	/******************************** METHODS *********************************/

	/**
	 * This slot is called when the current browsable is changed
	 */
	virtual void onCurrentBrowsableChanged();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/	

	/**
	 * The current editor
	 */ 
	GvvEditor* _currentEditor;

	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvEditorWindow( const GvvEditorWindow& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvEditorWindow& operator=( const GvvEditorWindow& );
	
};

} // namespace GvViewerGui

#endif
