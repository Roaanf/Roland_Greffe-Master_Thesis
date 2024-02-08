/****************************************************************************
** Meta object code from reading C++ file 'GvvMainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvMainWindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvMainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvViewerGui__GvvMainWindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      28,   27,   27,   27, 0x08,
      47,   27,   27,   27, 0x08,
      62,   27,   27,   27, 0x08,
      88,   27,   27,   27, 0x08,
     109,   27,   27,   27, 0x08,
     124,   27,   27,   27, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_GvViewerGui__GvvMainWindow[] = {
    "GvViewerGui::GvvMainWindow\0\0"
    "onActionOpenFile()\0onActionExit()\0"
    "onActionEditPreferences()\0"
    "onActionFullScreen()\0onActionHelp()\0"
    "onActionAbout()\0"
};

void GvViewerGui::GvvMainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvMainWindow *_t = static_cast<GvvMainWindow *>(_o);
        switch (_id) {
        case 0: _t->onActionOpenFile(); break;
        case 1: _t->onActionExit(); break;
        case 2: _t->onActionEditPreferences(); break;
        case 3: _t->onActionFullScreen(); break;
        case 4: _t->onActionHelp(); break;
        case 5: _t->onActionAbout(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData GvViewerGui::GvvMainWindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvViewerGui::GvvMainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_GvViewerGui__GvvMainWindow,
      qt_meta_data_GvViewerGui__GvvMainWindow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvViewerGui::GvvMainWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvViewerGui::GvvMainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvViewerGui::GvvMainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvViewerGui__GvvMainWindow))
        return static_cast<void*>(const_cast< GvvMainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int GvViewerGui::GvvMainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
