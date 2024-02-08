/****************************************************************************
** Meta object code from reading C++ file 'GvvAboutDialog.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/GvvAboutDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GvvAboutDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GvvAboutDialog[] = {

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
      16,   15,   15,   15, 0x09,
      49,   15,   15,   15, 0x09,

       0        // eod
};

static const char qt_meta_stringdata_GvvAboutDialog[] = {
    "GvvAboutDialog\0\0on__creditsPushButton_released()\0"
    "on__licensePushButton_released()\0"
};

void GvvAboutDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GvvAboutDialog *_t = static_cast<GvvAboutDialog *>(_o);
        switch (_id) {
        case 0: _t->on__creditsPushButton_released(); break;
        case 1: _t->on__licensePushButton_released(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData GvvAboutDialog::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GvvAboutDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_GvvAboutDialog,
      qt_meta_data_GvvAboutDialog, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GvvAboutDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GvvAboutDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GvvAboutDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GvvAboutDialog))
        return static_cast<void*>(const_cast< GvvAboutDialog*>(this));
    if (!strcmp(_clname, "Ui::GvvQAboutDialog"))
        return static_cast< Ui::GvvQAboutDialog*>(const_cast< GvvAboutDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int GvvAboutDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
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
