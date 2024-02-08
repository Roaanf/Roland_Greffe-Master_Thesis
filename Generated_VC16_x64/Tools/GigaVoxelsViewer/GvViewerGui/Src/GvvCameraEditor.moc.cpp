/****************************************************************************
** Meta object code from reading C++ file 'GvvCameraEditor.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvCameraEditor.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvCameraEditor.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvViewerGui__GvvCameraEditor[] = {

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
      37,   30,   29,   29, 0x08,
      87,   30,   29,   29, 0x08,
     137,   30,   29,   29, 0x08,
     192,   30,   29,   29, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_GvViewerGui__GvvCameraEditor[] = {
    "GvViewerGui::GvvCameraEditor\0\0pValue\0"
    "on__fieldOfViewDoubleSpinBox_valueChanged(double)\0"
    "on__sceneRadiusDoubleSpinBox_valueChanged(double)\0"
    "on__zNearCoefficientDoubleSpinBox_valueChanged(double)\0"
    "on__zClippingCoefficientDoubleSpinBox_valueChanged(double)\0"
};

void GvViewerGui::GvvCameraEditor::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvCameraEditor *_t = static_cast<GvvCameraEditor *>(_o);
        switch (_id) {
        case 0: _t->on__fieldOfViewDoubleSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->on__sceneRadiusDoubleSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->on__zNearCoefficientDoubleSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->on__zClippingCoefficientDoubleSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData GvViewerGui::GvvCameraEditor::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvViewerGui::GvvCameraEditor::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_GvViewerGui__GvvCameraEditor,
      qt_meta_data_GvViewerGui__GvvCameraEditor, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvViewerGui::GvvCameraEditor::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvViewerGui::GvvCameraEditor::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvViewerGui::GvvCameraEditor::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvViewerGui__GvvCameraEditor))
        return static_cast<void*>(const_cast< GvvCameraEditor*>(this));
    if (!strcmp(_clname, "Ui::GvvQCameraEditor"))
        return static_cast< Ui::GvvQCameraEditor*>(const_cast< GvvCameraEditor*>(this));
    return QWidget::qt_metacast(_clname);
}

int GvViewerGui::GvvCameraEditor::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
