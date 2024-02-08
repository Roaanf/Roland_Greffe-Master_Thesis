/****************************************************************************
** Meta object code from reading C++ file 'GvvTransformationEditor.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvTransformationEditor.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvTransformationEditor.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvViewerGui__GvvTransformationEditor[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      45,   38,   37,   37, 0x08,
      90,   38,   37,   37, 0x08,
     135,   38,   37,   37, 0x08,
     180,   38,   37,   37, 0x08,
     222,   38,   37,   37, 0x08,
     264,   38,   37,   37, 0x08,
     306,   38,   37,   37, 0x08,
     352,   38,   37,   37, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_GvViewerGui__GvvTransformationEditor[] = {
    "GvViewerGui::GvvTransformationEditor\0"
    "\0pValue\0on__xTranslationSpinBox_valueChanged(double)\0"
    "on__yTranslationSpinBox_valueChanged(double)\0"
    "on__zTranslationSpinBox_valueChanged(double)\0"
    "on__xRotationSpinBox_valueChanged(double)\0"
    "on__yRotationSpinBox_valueChanged(double)\0"
    "on__zRotationSpinBox_valueChanged(double)\0"
    "on__angleRotationSpinBox_valueChanged(double)\0"
    "on__uniformScaleSpinBox_valueChanged(double)\0"
};

void GvViewerGui::GvvTransformationEditor::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvTransformationEditor *_t = static_cast<GvvTransformationEditor *>(_o);
        switch (_id) {
        case 0: _t->on__xTranslationSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->on__yTranslationSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->on__zTranslationSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->on__xRotationSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->on__yRotationSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->on__zRotationSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: _t->on__angleRotationSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: _t->on__uniformScaleSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData GvViewerGui::GvvTransformationEditor::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvViewerGui::GvvTransformationEditor::staticMetaObject = {
    { &GvvSectionEditor::staticMetaObject, qt_meta_stringdata_GvViewerGui__GvvTransformationEditor,
      qt_meta_data_GvViewerGui__GvvTransformationEditor, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvViewerGui::GvvTransformationEditor::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvViewerGui::GvvTransformationEditor::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvViewerGui::GvvTransformationEditor::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvViewerGui__GvvTransformationEditor))
        return static_cast<void*>(const_cast< GvvTransformationEditor*>(this));
    if (!strcmp(_clname, "Ui::GvvQTransformationEditor"))
        return static_cast< Ui::GvvQTransformationEditor*>(const_cast< GvvTransformationEditor*>(this));
    return GvvSectionEditor::qt_metacast(_clname);
}

int GvViewerGui::GvvTransformationEditor::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = GvvSectionEditor::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
