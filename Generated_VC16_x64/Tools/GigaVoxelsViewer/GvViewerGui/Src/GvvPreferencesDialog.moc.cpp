/****************************************************************************
** Meta object code from reading C++ file 'GvvPreferencesDialog.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvPreferencesDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvPreferencesDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvvPreferencesDialog[] = {

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
      22,   21,   21,   21, 0x09,
      71,   21,   21,   21, 0x09,
     122,   21,   21,   21, 0x09,
     176,   21,   21,   21, 0x09,
     228,   21,   21,   21, 0x09,
     287,  278,   21,   21, 0x09,
     334,  278,   21,   21, 0x09,
     384,  278,   21,   21, 0x09,
     432,  278,   21,   21, 0x09,

       0        // eod
};

static const char qt_meta_stringdata_GvvPreferencesDialog[] = {
    "GvvPreferencesDialog\0\0"
    "on__3DWindowBackgroundColorToolButton_released()\0"
    "on__nodeHasBrickTerminalColorToolButton_released()\0"
    "on__nodeHasBrickNotTerminalColorToolButton_released()\0"
    "on__nodeIsBrickNotInCacheColorToolButton_released()\0"
    "on__nodeEmptyOrConstantColorToolButton_released()\0"
    "pChecked\0on__nodeHasBrickTerminalCheckBox_toggled(bool)\0"
    "on__nodeHasBrickNotTerminalCheckBox_toggled(bool)\0"
    "on__nodeIsBrickNotInCacheCheckBox_toggled(bool)\0"
    "on__nodeEmptyOrConstantCheckBox_toggled(bool)\0"
};

void GvvPreferencesDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvPreferencesDialog *_t = static_cast<GvvPreferencesDialog *>(_o);
        switch (_id) {
        case 0: _t->on__3DWindowBackgroundColorToolButton_released(); break;
        case 1: _t->on__nodeHasBrickTerminalColorToolButton_released(); break;
        case 2: _t->on__nodeHasBrickNotTerminalColorToolButton_released(); break;
        case 3: _t->on__nodeIsBrickNotInCacheColorToolButton_released(); break;
        case 4: _t->on__nodeEmptyOrConstantColorToolButton_released(); break;
        case 5: _t->on__nodeHasBrickTerminalCheckBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->on__nodeHasBrickNotTerminalCheckBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->on__nodeIsBrickNotInCacheCheckBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: _t->on__nodeEmptyOrConstantCheckBox_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData GvvPreferencesDialog::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvvPreferencesDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_GvvPreferencesDialog,
      qt_meta_data_GvvPreferencesDialog, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvvPreferencesDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvvPreferencesDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvvPreferencesDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvvPreferencesDialog))
        return static_cast<void*>(const_cast< GvvPreferencesDialog*>(this));
    if (!strcmp(_clname, "Ui::GvvQPreferencesDialog"))
        return static_cast< Ui::GvvQPreferencesDialog*>(const_cast< GvvPreferencesDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int GvvPreferencesDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
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
