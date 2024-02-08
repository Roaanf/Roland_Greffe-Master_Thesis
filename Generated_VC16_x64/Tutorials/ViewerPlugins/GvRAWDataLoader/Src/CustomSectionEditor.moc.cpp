/****************************************************************************
** Meta object code from reading C++ file 'CustomSectionEditor.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tutorials/ViewerPlugins/GvRAWDataLoader/Inc/CustomSectionEditor.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'CustomSectionEditor.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_CustomSectionEditor[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      27,   21,   20,   20, 0x08,
      86,   21,   20,   20, 0x08,
     146,   21,   20,   20, 0x08,
     203,   21,   20,   20, 0x08,
     261,   21,   20,   20, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_CustomSectionEditor[] = {
    "CustomSectionEditor\0\0value\0"
    "on__producerThresholdDoubleSpinBoxLow_valueChanged(double)\0"
    "on__producerThresholdDoubleSpinBoxHigh_valueChanged(double)\0"
    "on__shaderThresholdDoubleSpinBoxLow_valueChanged(double)\0"
    "on__shaderThresholdDoubleSpinBoxHigh_valueChanged(double)\0"
    "on__shaderFullOpacityDistanceDoubleSpinBox_valueChanged(double)\0"
};

void CustomSectionEditor::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        CustomSectionEditor *_t = static_cast<CustomSectionEditor *>(_o);
        switch (_id) {
        case 0: _t->on__producerThresholdDoubleSpinBoxLow_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->on__producerThresholdDoubleSpinBoxHigh_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->on__shaderThresholdDoubleSpinBoxLow_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->on__shaderThresholdDoubleSpinBoxHigh_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->on__shaderFullOpacityDistanceDoubleSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData CustomSectionEditor::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CustomSectionEditor::staticMetaObject = {
    { &GvViewerGui::GvvSectionEditor::staticMetaObject, qt_meta_stringdata_CustomSectionEditor,
      qt_meta_data_CustomSectionEditor, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CustomSectionEditor::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CustomSectionEditor::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CustomSectionEditor::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CustomSectionEditor))
        return static_cast<void*>(const_cast< CustomSectionEditor*>(this));
    if (!strcmp(_clname, "Ui::GvtQCustomEditor"))
        return static_cast< Ui::GvtQCustomEditor*>(const_cast< CustomSectionEditor*>(this));
    typedef GvViewerGui::GvvSectionEditor QMocSuperClass;
    return QMocSuperClass::qt_metacast(_clname);
}

int CustomSectionEditor::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    typedef GvViewerGui::GvvSectionEditor QMocSuperClass;
    _id = QMocSuperClass::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
