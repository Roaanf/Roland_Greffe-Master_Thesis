/****************************************************************************
** Meta object code from reading C++ file 'GvvCacheEditor.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvCacheEditor.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvCacheEditor.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvViewerGui__GvvCacheEditor[] = {

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
      38,   29,   28,   28, 0x08,
     104,   29,   28,   28, 0x08,
     157,  155,   28,   28, 0x08,
     201,  155,   28,   28, 0x08,
     238,   29,   28,   28, 0x08,
     281,  274,   28,   28, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_GvViewerGui__GvvCacheEditor[] = {
    "GvViewerGui::GvvCacheEditor\0\0pChecked\0"
    "on__preventReplacingUsedElementsCachePolicyCheckBox_toggled(bool)\0"
    "on__smoothLoadingCachePolicyGroupBox_toggled(bool)\0"
    "i\0on__nbSubdivisionsSpinBox_valueChanged(int)\0"
    "on__nbLoadsSpinBox_valueChanged(int)\0"
    "on__timeLimitGroupBox_toggled(bool)\0"
    "pValue\0on__timeLimitDoubleSpinBox_valueChanged(double)\0"
};

void GvViewerGui::GvvCacheEditor::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvCacheEditor *_t = static_cast<GvvCacheEditor *>(_o);
        switch (_id) {
        case 0: _t->on__preventReplacingUsedElementsCachePolicyCheckBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->on__smoothLoadingCachePolicyGroupBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: _t->on__nbSubdivisionsSpinBox_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->on__nbLoadsSpinBox_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->on__timeLimitGroupBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->on__timeLimitDoubleSpinBox_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData GvViewerGui::GvvCacheEditor::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvViewerGui::GvvCacheEditor::staticMetaObject = {
    { &GvvSectionEditor::staticMetaObject, qt_meta_stringdata_GvViewerGui__GvvCacheEditor,
      qt_meta_data_GvViewerGui__GvvCacheEditor, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvViewerGui::GvvCacheEditor::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvViewerGui::GvvCacheEditor::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvViewerGui::GvvCacheEditor::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvViewerGui__GvvCacheEditor))
        return static_cast<void*>(const_cast< GvvCacheEditor*>(this));
    if (!strcmp(_clname, "Ui::GvvQCacheEditor"))
        return static_cast< Ui::GvvQCacheEditor*>(const_cast< GvvCacheEditor*>(this));
    return GvvSectionEditor::qt_metacast(_clname);
}

int GvViewerGui::GvvCacheEditor::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = GvvSectionEditor::qt_metacall(_c, _id, _a);
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
