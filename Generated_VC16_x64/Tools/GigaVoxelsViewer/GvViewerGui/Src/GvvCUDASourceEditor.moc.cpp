/****************************************************************************
** Meta object code from reading C++ file 'GvvCUDASourceEditor.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvCUDASourceEditor.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvCUDASourceEditor.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvViewerGui__GvvCUDASourceEditor[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      34,   33,   33,   33, 0x08,
      46,   33,   33,   33, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_GvViewerGui__GvvCUDASourceEditor[] = {
    "GvViewerGui::GvvCUDASourceEditor\0\0"
    "onCompile()\0onApply()\0"
};

void GvViewerGui::GvvCUDASourceEditor::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvCUDASourceEditor *_t = static_cast<GvvCUDASourceEditor *>(_o);
        switch (_id) {
        case 0: _t->onCompile(); break;
        case 1: _t->onApply(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData GvViewerGui::GvvCUDASourceEditor::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvViewerGui::GvvCUDASourceEditor::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_GvViewerGui__GvvCUDASourceEditor,
      qt_meta_data_GvViewerGui__GvvCUDASourceEditor, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvViewerGui::GvvCUDASourceEditor::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvViewerGui::GvvCUDASourceEditor::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvViewerGui::GvvCUDASourceEditor::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvViewerGui__GvvCUDASourceEditor))
        return static_cast<void*>(const_cast< GvvCUDASourceEditor*>(this));
    return QWidget::qt_metacast(_clname);
}

int GvViewerGui::GvvCUDASourceEditor::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
