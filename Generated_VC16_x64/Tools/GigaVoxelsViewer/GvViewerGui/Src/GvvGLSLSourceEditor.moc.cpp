/****************************************************************************
** Meta object code from reading C++ file 'GvvGLSLSourceEditor.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvGLSLSourceEditor.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvGLSLSourceEditor.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvViewerGui__GvvGLSLSourceEditor[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      34,   33,   33,   33, 0x08,
      45,   33,   33,   33, 0x08,
      62,   55,   33,   33, 0x08,
      95,   33,   33,   33, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_GvViewerGui__GvvGLSLSourceEditor[] = {
    "GvViewerGui::GvvGLSLSourceEditor\0\0"
    "onReload()\0onApply()\0pIndex\0"
    "on_tabWidget_currentChanged(int)\0"
    "on__applyButton_released()\0"
};

void GvViewerGui::GvvGLSLSourceEditor::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvGLSLSourceEditor *_t = static_cast<GvvGLSLSourceEditor *>(_o);
        switch (_id) {
        case 0: _t->onReload(); break;
        case 1: _t->onApply(); break;
        case 2: _t->on_tabWidget_currentChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->on__applyButton_released(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData GvViewerGui::GvvGLSLSourceEditor::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvViewerGui::GvvGLSLSourceEditor::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_GvViewerGui__GvvGLSLSourceEditor,
      qt_meta_data_GvViewerGui__GvvGLSLSourceEditor, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvViewerGui::GvvGLSLSourceEditor::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvViewerGui::GvvGLSLSourceEditor::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvViewerGui::GvvGLSLSourceEditor::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvViewerGui__GvvGLSLSourceEditor))
        return static_cast<void*>(const_cast< GvvGLSLSourceEditor*>(this));
    return QWidget::qt_metacast(_clname);
}

int GvViewerGui::GvvGLSLSourceEditor::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
