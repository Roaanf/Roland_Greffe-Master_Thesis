/****************************************************************************
** Meta object code from reading C++ file 'GvvBrowser.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvBrowser.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvBrowser.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvViewerGui__GvvBrowser[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      39,   25,   24,   24, 0x09,
      75,   25,   24,   24, 0x09,
     130,  111,   24,   24, 0x09,

       0        // eod
};

static const char qt_meta_stringdata_GvViewerGui__GvvBrowser[] = {
    "GvViewerGui::GvvBrowser\0\0pItem,pColumn\0"
    "onItemClicked(QTreeWidgetItem*,int)\0"
    "onItemChanged(QTreeWidgetItem*,int)\0"
    "pCurrent,pPrevious\0"
    "onCurrentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)\0"
};

void GvViewerGui::GvvBrowser::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvBrowser *_t = static_cast<GvvBrowser *>(_o);
        switch (_id) {
        case 0: _t->onItemClicked((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 1: _t->onItemChanged((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 2: _t->onCurrentItemChanged((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< QTreeWidgetItem*(*)>(_a[2]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData GvViewerGui::GvvBrowser::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvViewerGui::GvvBrowser::staticMetaObject = {
    { &QTreeWidget::staticMetaObject, qt_meta_stringdata_GvViewerGui__GvvBrowser,
      qt_meta_data_GvViewerGui__GvvBrowser, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvViewerGui::GvvBrowser::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvViewerGui::GvvBrowser::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvViewerGui::GvvBrowser::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvViewerGui__GvvBrowser))
        return static_cast<void*>(const_cast< GvvBrowser*>(this));
    return QTreeWidget::qt_metacast(_clname);
}

int GvViewerGui::GvvBrowser::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QTreeWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
