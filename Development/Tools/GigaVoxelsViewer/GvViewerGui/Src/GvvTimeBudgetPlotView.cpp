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

#include "GvvTimeBudgetPlotView.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QPainter>
#include <QPicture>
#include <QFile>
#include <QFont>

// Qwt
#define GS_QWT_6_0
#include <qwt_plot_picker.h>
#include <qwt_plot_layout.h>
#include <qwt_dyngrid_layout.h>
#include <qwt_painter.h>
#include <qwt_plot_grid.h>
#include <qwt_plot_panner.h>
#include <qwt_plot_zoomer.h>
#include <qwt_symbol.h>
#include <qwt_legend.h>
#ifdef GS_QWT_6_0
#include <qwt_legend_item.h>
#endif
#include <qwt_plot_curve.h>
#include <qwt_scale_engine.h>
#include <qwt_scale_widget.h>
#include <qwt_text_label.h>
#include <qwt_plot_marker.h>

// System
#include <cstring>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
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
 * Default constructor
 ******************************************************************************/
GvvTimeBudgetPlotView::GvvTimeBudgetPlotView( QWidget* pParent, const char* pName ) 
:	QwtPlot( QwtText( pName ), pParent )
,	_frameTimeCurve( NULL )
,	_frameTimeMarker( NULL )
,	_timeBudgetCurve( NULL )
,	_timeBudgetMarker( NULL )
{
	//** Setups background color
	setCanvasBackground( QColor( "white" ) );

	//setFooter( QString( "Evolution of frame duration rate along time" ) );

	//** Setups the panner
    QwtPlotPanner* panner = new QwtPlotPanner( canvas() );
    panner->setMouseButton( Qt::MidButton );
    panner->setEnabled( true );

	//** Setups the grid
	QwtPlotGrid* grid = new QwtPlotGrid;
    grid->attach( this );
	grid->enableX( true );
	grid->enableY( true );
#ifdef GS_QWT_6_0
	grid->setMajPen( QPen( QBrush( QColor( Qt::black ) ), 1.f, Qt::DotLine ) );
#else
	grid->setMajorPen( QPen( QBrush( QColor( Qt::black ) ), 1.f, Qt::DotLine ) );
#endif
	grid->enableXMin( true );
	grid->enableYMin( false );
#ifdef GS_QWT_6_0
	grid->setMinPen( QPen( QBrush( QColor( Qt::black ) ), 1.f, Qt::DotLine ) );
#else
	grid->setMinorPen( QPen( QBrush( QColor( Qt::black ) ), 1.f, Qt::DotLine ) );
#endif

	//** Setups the zoomer
	QwtPlotZoomer* zoommer = new QwtPlotZoomer( QwtPlot::xBottom, QwtPlot::yLeft, canvas() );
	zoommer->setMaxStackDepth( 4 );
	//zoommer->setSelectionFlags( QwtPicker::DragSelection | QwtPicker::CornerToCorner );
	zoommer->setTrackerMode( QwtPicker::AlwaysOff );
	zoommer->setMousePattern( QwtEventPattern::MouseSelect2, Qt::RightButton, Qt::ControlModifier );
	zoommer->setMousePattern( QwtEventPattern::MouseSelect3, Qt::RightButton );
	zoommer->setRubberBand( QwtPicker::RectRubberBand );
	zoommer->setRubberBandPen( QColor( Qt::green ) );   
	zoommer->setEnabled( true );
	zoommer->zoom( 0 );

	QwtPlotPicker* picker = new QwtPlotPicker( QwtPlot::xBottom, QwtPlot::yLeft,
								QwtPlotPicker::CrossRubberBand, 
								QwtPicker::AlwaysOn, 
								canvas() );
	if ( picker != NULL )
	{
		picker->setRubberBandPen( QColor( Qt::green ) );
		picker->setRubberBand( QwtPicker::CrossRubberBand );
		picker->setTrackerPen( QColor( Qt::black ) );
	}

	_frameTimeMarker = new QwtPlotMarker();
	_frameTimeMarker->attach( this );
	QwtSymbol* nodeSymbol = new QwtSymbol( QwtSymbol::Diamond, QBrush( QColor( Qt::red ) ), QPen(), QSize( 15, 15 ) );
	_frameTimeMarker->setSymbol( nodeSymbol );

	/*_timeBudgetMarker = new QwtPlotMarker();
	_timeBudgetMarker->attach( this );
	QwtSymbol* brickSymbol = new QwtSymbol( QwtSymbol::Ellipse, QBrush( QColor( Qt::green ) ), QPen(), QSize( 15, 15 ) );
	_timeBudgetMarker->setSymbol( brickSymbol );*/

	//** Setups the legend
	QwtLegend* legend = new QwtLegend();
#ifdef GS_QWT_6_0
	legend->setItemMode( QwtLegend::CheckableItem );
#else
	legend->setDefaultItemMode( QwtLegendData::Checkable );
#endif
	insertLegend( legend, QwtPlot::RightLegend );

	//** Sets auto replot
	setAutoReplot( true );

	//** Disables the autodelete mode
	setAutoDelete( true );

	_frameTimeCurve = new QwtPlotCurve( "Frame Time" );
	_frameTimeCurve->attach( this );
	_frameTimeCurve->setRenderHint( QwtPlotItem::RenderAntialiased );
	_frameTimeCurve->setPen( QPen( QColor( Qt::red ), 1.0f, Qt::SolidLine ) );

	_timeBudgetCurve = new QwtPlotCurve( "Requested Time Budget" );
	_timeBudgetCurve->attach( this );
	_timeBudgetCurve->setRenderHint( QwtPlotItem::RenderAntialiased );
	_timeBudgetCurve->setPen( QPen( QColor( Qt::green ), 1.0f, Qt::SolidLine ) );

	//_unusedNodesCurve = new QwtPlotCurve( "Unused Nodes" );
	//_unusedNodesCurve->attach( this );
	//_unusedNodesCurve->setRenderHint( QwtPlotItem::RenderAntialiased );
	//_unusedNodesCurve->setPen( QPen( QColor( Qt::blue ), 1.0f, Qt::DotLine ) );

	//_unusedBricksCurve = new QwtPlotCurve( "Unused Bricks" );
	//_unusedBricksCurve->attach( this );
	//_unusedBricksCurve->setRenderHint( QwtPlotItem::RenderAntialiased );
	//_unusedBricksCurve->setPen( QPen( QColor( Qt::yellow ), 1.0f, Qt::DotLine ) );

	for ( unsigned int i = 0; i < 500 ; i++ )
	{
		_xFrameTime[ i ] = i;
		_yFrameTime[ i ] = 0;

		_xTimeBudget[ i ] = i;
		_yTimeBudget[ i ] = ( 1.f / 60.f ) * 1000.f;
	}
	_frameTimeCurve->setRawSamples( _xFrameTime, _yFrameTime, 500 );
	_timeBudgetCurve->setRawSamples( _xTimeBudget, _yTimeBudget, 500 );

	setAxisScale( QwtPlot::yLeft, 0.f, 50.f );

	setAxisTitle( QwtPlot::yLeft, tr( "Frame duration (ms)" ) );
	setAxisTitle( QwtPlot::xBottom, tr( "time" ) );
}

/******************************************************************************
 * Default destructor
 ******************************************************************************/
GvvTimeBudgetPlotView::~GvvTimeBudgetPlotView()
{
}

/******************************************************************************
 * Draw the specified curve
 *
 * @param pCurve specifies the curve to be drawn
 ******************************************************************************/
void GvvTimeBudgetPlotView::onCurveChanged( unsigned int pFrame, float pFrameTime )
{
	if ( ( pFrame % 500 ) == 0 )
	{
		memset( _yFrameTime, 0, 500 *sizeof( double ) );
	//	memset( _yTimeBudget, 0, 500 *sizeof( double ) );
	}

	const unsigned int indexFrame = pFrame % 500;

	_yFrameTime[ indexFrame ] = pFrameTime;
	//_yTimeBudget[ indexFrame ] = ( 1.f / 60.f ) * 1000.f;
	
	_frameTimeMarker->setXValue( indexFrame );
	_frameTimeMarker->setYValue( pFrameTime );
	/*_timeBudgetMarker->setXValue( indexFrame );
	_timeBudgetMarker->setYValue( pBrickValue );*/

	replot();
}

/******************************************************************************
 * Set the user requested time budget
 *
 * @param pValue the user requested time budget
 ******************************************************************************/
void GvvTimeBudgetPlotView::setTimeBudget( unsigned int pValue )
{
	for ( unsigned int i = 0; i < 500 ; i++ )
	{
		_yTimeBudget[ i ] = ( 1.f / static_cast< float >( pValue ) ) * 1000.f;
	}
}
