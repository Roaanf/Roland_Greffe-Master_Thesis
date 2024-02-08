/****************************************************************************
** Meta object code from reading C++ file 'GvvRendererEditor.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvRendererEditor.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvRendererEditor.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvViewerGui__GvvRendererEditor[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      34,   32,   31,   31, 0x08,
      81,   72,   31,   31, 0x08,
     121,   72,   31,   31, 0x08,
     167,   72,   31,   31, 0x08,
     222,  215,   31,   31, 0x08,
     271,  215,   31,   31, 0x08,
     336,  321,   31,   31, 0x08,
     361,   72,   31,   31, 0x08,
     408,  215,   31,   31, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_GvViewerGui__GvvRendererEditor[] = {
    "GvViewerGui::GvvRendererEditor\0\0i\0"
    "on__maxDepthSpinBox_valueChanged(int)\0"
    "pChecked\0on__dynamicUpdateCheckBox_toggled(bool)\0"
    "on__priorityOnBricksRadioButton_toggled(bool)\0"
    "on__viewportOffscreenSizeGroupBox_toggled(bool)\0"
    "pValue\0on__graphicsBufferWidthSpinBox_valueChanged(int)\0"
    "on__graphicsBufferHeightSpinBox_valueChanged(int)\0"
    "pWidth,pHeight\0onViewerResized(int,int)\0"
    "on__timeBudgetParametersGroupBox_toggled(bool)\0"
    "on__timeBudgetSpinBox_valueChanged(int)\0"
};

void GvViewerGui::GvvRendererEditor::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvRendererEditor *_t = static_cast<GvvRendererEditor *>(_o);
        switch (_id) {
        case 0: _t->on__maxDepthSpinBox_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->on__dynamicUpdateCheckBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: _t->on__priorityOnBricksRadioButton_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: _t->on__viewportOffscreenSizeGroupBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 4: _t->on__graphicsBufferWidthSpinBox_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->on__graphicsBufferHeightSpinBox_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->onViewerResized((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 7: _t->on__timeBudgetParametersGroupBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: _t->on__timeBudgetSpinBox_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData GvViewerGui::GvvRendererEditor::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvViewerGui::GvvRendererEditor::staticMetaObject = {
    { &GvvSectionEditor::staticMetaObject, qt_meta_stringdata_GvViewerGui__GvvRendererEditor,
      qt_meta_data_GvViewerGui__GvvRendererEditor, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvViewerGui::GvvRendererEditor::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvViewerGui::GvvRendererEditor::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvViewerGui::GvvRendererEditor::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvViewerGui__GvvRendererEditor))
        return static_cast<void*>(const_cast< GvvRendererEditor*>(this));
    if (!strcmp(_clname, "Ui::GvvQRendererEditor"))
        return static_cast< Ui::GvvQRendererEditor*>(const_cast< GvvRendererEditor*>(this));
    return GvvSectionEditor::qt_metacast(_clname);
}

int GvViewerGui::GvvRendererEditor::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = GvvSectionEditor::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
