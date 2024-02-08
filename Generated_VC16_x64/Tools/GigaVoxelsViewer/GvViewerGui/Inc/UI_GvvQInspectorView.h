/********************************************************************************
** Form generated from reading UI file 'GvvQInspectorView.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQINSPECTORVIEW_H
#define UI_GVVQINSPECTORVIEW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QSpacerItem>
#include <QtGui/QTableWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQInspectorView
{
public:
    QVBoxLayout *verticalLayout_5;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QTableWidget *_nodePoolTableWidget;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_2;
    QTableWidget *_dataPoolTableWidget;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout_3;
    QTableWidget *_nodeCacheManagerTableWidget;
    QGroupBox *groupBox_4;
    QVBoxLayout *verticalLayout_4;
    QTableWidget *_dataCacheManagerTableWidget;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *GvvQInspectorView)
    {
        if (GvvQInspectorView->objectName().isEmpty())
            GvvQInspectorView->setObjectName(QString::fromUtf8("GvvQInspectorView"));
        GvvQInspectorView->resize(952, 555);
        verticalLayout_5 = new QVBoxLayout(GvvQInspectorView);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        groupBox = new QGroupBox(GvvQInspectorView);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout = new QVBoxLayout(groupBox);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        _nodePoolTableWidget = new QTableWidget(groupBox);
        if (_nodePoolTableWidget->rowCount() < 2)
            _nodePoolTableWidget->setRowCount(2);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        _nodePoolTableWidget->setVerticalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        _nodePoolTableWidget->setVerticalHeaderItem(1, __qtablewidgetitem1);
        _nodePoolTableWidget->setObjectName(QString::fromUtf8("_nodePoolTableWidget"));
        _nodePoolTableWidget->setAlternatingRowColors(true);

        verticalLayout->addWidget(_nodePoolTableWidget);


        verticalLayout_5->addWidget(groupBox);

        groupBox_2 = new QGroupBox(GvvQInspectorView);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        verticalLayout_2 = new QVBoxLayout(groupBox_2);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        _dataPoolTableWidget = new QTableWidget(groupBox_2);
        if (_dataPoolTableWidget->rowCount() < 1)
            _dataPoolTableWidget->setRowCount(1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        _dataPoolTableWidget->setVerticalHeaderItem(0, __qtablewidgetitem2);
        _dataPoolTableWidget->setObjectName(QString::fromUtf8("_dataPoolTableWidget"));
        _dataPoolTableWidget->setAlternatingRowColors(true);

        verticalLayout_2->addWidget(_dataPoolTableWidget);


        verticalLayout_5->addWidget(groupBox_2);

        groupBox_3 = new QGroupBox(GvvQInspectorView);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        verticalLayout_3 = new QVBoxLayout(groupBox_3);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        _nodeCacheManagerTableWidget = new QTableWidget(groupBox_3);
        if (_nodeCacheManagerTableWidget->rowCount() < 2)
            _nodeCacheManagerTableWidget->setRowCount(2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        _nodeCacheManagerTableWidget->setVerticalHeaderItem(0, __qtablewidgetitem3);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        _nodeCacheManagerTableWidget->setVerticalHeaderItem(1, __qtablewidgetitem4);
        _nodeCacheManagerTableWidget->setObjectName(QString::fromUtf8("_nodeCacheManagerTableWidget"));
        _nodeCacheManagerTableWidget->setAlternatingRowColors(true);

        verticalLayout_3->addWidget(_nodeCacheManagerTableWidget);


        verticalLayout_5->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(GvvQInspectorView);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        verticalLayout_4 = new QVBoxLayout(groupBox_4);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        _dataCacheManagerTableWidget = new QTableWidget(groupBox_4);
        if (_dataCacheManagerTableWidget->rowCount() < 2)
            _dataCacheManagerTableWidget->setRowCount(2);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        _dataCacheManagerTableWidget->setVerticalHeaderItem(0, __qtablewidgetitem5);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        _dataCacheManagerTableWidget->setVerticalHeaderItem(1, __qtablewidgetitem6);
        _dataCacheManagerTableWidget->setObjectName(QString::fromUtf8("_dataCacheManagerTableWidget"));
        _dataCacheManagerTableWidget->setAlternatingRowColors(true);

        verticalLayout_4->addWidget(_dataCacheManagerTableWidget);


        verticalLayout_5->addWidget(groupBox_4);

        verticalSpacer = new QSpacerItem(20, 2, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer);


        retranslateUi(GvvQInspectorView);

        QMetaObject::connectSlotsByName(GvvQInspectorView);
    } // setupUi

    void retranslateUi(QWidget *GvvQInspectorView)
    {
        GvvQInspectorView->setWindowTitle(QApplication::translate("GvvQInspectorView", "Form", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvvQInspectorView", "Node Pool", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem = _nodePoolTableWidget->verticalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("GvvQInspectorView", "child array", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = _nodePoolTableWidget->verticalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("GvvQInspectorView", "data array", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("GvvQInspectorView", "Data Pool", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = _dataPoolTableWidget->verticalHeaderItem(0);
        ___qtablewidgetitem2->setText(QApplication::translate("GvvQInspectorView", "channel 0", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("GvvQInspectorView", "Node Cache Manager", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem3 = _nodeCacheManagerTableWidget->verticalHeaderItem(0);
        ___qtablewidgetitem3->setText(QApplication::translate("GvvQInspectorView", "time stamps", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem4 = _nodeCacheManagerTableWidget->verticalHeaderItem(1);
        ___qtablewidgetitem4->setText(QApplication::translate("GvvQInspectorView", "element addresses", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("GvvQInspectorView", "Data Cache Manager", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem5 = _dataCacheManagerTableWidget->verticalHeaderItem(0);
        ___qtablewidgetitem5->setText(QApplication::translate("GvvQInspectorView", "time stamps", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem6 = _dataCacheManagerTableWidget->verticalHeaderItem(1);
        ___qtablewidgetitem6->setText(QApplication::translate("GvvQInspectorView", "element addresses", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQInspectorView: public Ui_GvvQInspectorView {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQINSPECTORVIEW_H
