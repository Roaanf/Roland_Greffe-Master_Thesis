/****************************************************************************
** Meta object code from reading C++ file 'Qtfe.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/Qtfe.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Qtfe.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_Qtfe[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
       6,    5,    5,    5, 0x05,

 // slots: signature, parameters, type, tag, flags
      24,    5,    5,    5, 0x09,
      46,    5,    5,    5, 0x09,
      70,    5,    5,    5, 0x09,
      92,    5,    5,    5, 0x09,
     114,    5,    5,    5, 0x09,
     133,    5,    5,    5, 0x09,

       0        // eod
};

static const char qt_meta_stringdata_Qtfe[] = {
    "Qtfe\0\0functionChanged()\0onSaveButtonClicked()\0"
    "onSaveAsButtonClicked()\0onLoadButtonClicked()\0"
    "onQuitButtonClicked()\0onChannelChanged()\0"
    "onOutputBindingChanged()\0"
};

void Qtfe::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        Qtfe *_t = static_cast<Qtfe *>(_o);
        switch (_id) {
        case 0: _t->functionChanged(); break;
        case 1: _t->onSaveButtonClicked(); break;
        case 2: _t->onSaveAsButtonClicked(); break;
        case 3: _t->onLoadButtonClicked(); break;
        case 4: _t->onQuitButtonClicked(); break;
        case 5: _t->onChannelChanged(); break;
        case 6: _t->onOutputBindingChanged(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData Qtfe::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject Qtfe::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_Qtfe,
      qt_meta_data_Qtfe, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Qtfe::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Qtfe::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Qtfe::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Qtfe))
        return static_cast<void*>(const_cast< Qtfe*>(this));
    if (!strcmp(_clname, "Ui::GvQTransferFunctionEditor"))
        return static_cast< Ui::GvQTransferFunctionEditor*>(const_cast< Qtfe*>(this));
    return QWidget::qt_metacast(_clname);
}

int Qtfe::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void Qtfe::functionChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
