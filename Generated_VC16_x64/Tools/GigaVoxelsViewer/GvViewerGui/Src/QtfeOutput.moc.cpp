/****************************************************************************
** Meta object code from reading C++ file 'QtfeOutput.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Development/Tools/GigaVoxelsViewer/GvViewerGui/Inc/QtfeOutput.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QtfeOutput.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QtfeOutput[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      12,   11,   11,   11, 0x05,

 // slots: signature, parameters, type, tag, flags
      49,   35,   11,   11, 0x08,
      69,   35,   11,   11, 0x08,
      89,   35,   11,   11, 0x08,
     109,   35,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_QtfeOutput[] = {
    "QtfeOutput\0\0outputBindingChanged()\0"
    "pChannelIndex\0bindChannelToR(int)\0"
    "bindChannelToG(int)\0bindChannelToB(int)\0"
    "bindChannelToA(int)\0"
};

void QtfeOutput::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        QtfeOutput *_t = static_cast<QtfeOutput *>(_o);
        switch (_id) {
        case 0: _t->outputBindingChanged(); break;
        case 1: _t->bindChannelToR((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->bindChannelToG((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->bindChannelToB((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->bindChannelToA((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData QtfeOutput::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject QtfeOutput::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_QtfeOutput,
      qt_meta_data_QtfeOutput, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QtfeOutput::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QtfeOutput::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QtfeOutput::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QtfeOutput))
        return static_cast<void*>(const_cast< QtfeOutput*>(this));
    return QWidget::qt_metacast(_clname);
}

int QtfeOutput::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void QtfeOutput::outputBindingChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
