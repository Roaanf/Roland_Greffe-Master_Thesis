/********************************************************************************
** Form generated from reading UI file 'GvvQCacheUsageWidget.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQCACHEUSAGEWIDGET_H
#define UI_GVVQCACHEUSAGEWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQCacheUsageWidget
{
public:
    QVBoxLayout *verticalLayout_3;
    QGroupBox *_cacheUsageGroupBox;
    QVBoxLayout *verticalLayout_2;
    QGroupBox *_nodeCacheGroupBox;
    QGridLayout *gridLayout;
    QLabel *label_9;
    QLineEdit *_nodeCacheMemoryLineEdit;
    QLabel *label_12;
    QSpacerItem *horizontalSpacer_2;
    QLabel *label_5;
    QLineEdit *_nodeCacheCapacityLineEdit;
    QLabel *label_11;
    QSpacerItem *horizontalSpacer;
    QLabel *label_3;
    QLineEdit *_nodeCacheFillingRatioLineEdit;
    QLabel *label;
    QLineEdit *_nodeCacheNbElementsLineEdit;
    QGroupBox *_dataCacheGroupBox;
    QGridLayout *gridLayout_3;
    QLabel *label_15;
    QLineEdit *_dataCacheMemoryLineEdit;
    QLabel *label_13;
    QSpacerItem *horizontalSpacer_5;
    QLabel *label_6;
    QLineEdit *_dataCacheCapacityLineEdit;
    QLabel *label_7;
    QSpacerItem *horizontalSpacer_3;
    QLabel *label_4;
    QLineEdit *_dataCacheFillingRatioLineEdit;
    QLabel *label_2;
    QLineEdit *_dataCacheNbElementsLineEdit;
    QGroupBox *_treeMonitoringGroupBox;
    QVBoxLayout *verticalLayout;
    QGroupBox *_treeNodeGroupBox;
    QGridLayout *gridLayout_2;
    QLabel *label_8;
    QLineEdit *_treeEmptyNodeRatioLineEdit;
    QLabel *label_10;
    QLineEdit *_treeNbEmptyNodeLineEdit;
    QGroupBox *_treeBrickGroupBox;
    QGridLayout *gridLayout_4;
    QLabel *label_14;
    QLineEdit *_treeBrickSparsenessRatioLineEdit;
    QSpacerItem *horizontalSpacer_4;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *GvvQCacheUsageWidget)
    {
        if (GvvQCacheUsageWidget->objectName().isEmpty())
            GvvQCacheUsageWidget->setObjectName(QString::fromUtf8("GvvQCacheUsageWidget"));
        GvvQCacheUsageWidget->resize(297, 428);
        verticalLayout_3 = new QVBoxLayout(GvvQCacheUsageWidget);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        _cacheUsageGroupBox = new QGroupBox(GvvQCacheUsageWidget);
        _cacheUsageGroupBox->setObjectName(QString::fromUtf8("_cacheUsageGroupBox"));
        verticalLayout_2 = new QVBoxLayout(_cacheUsageGroupBox);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        _nodeCacheGroupBox = new QGroupBox(_cacheUsageGroupBox);
        _nodeCacheGroupBox->setObjectName(QString::fromUtf8("_nodeCacheGroupBox"));
        gridLayout = new QGridLayout(_nodeCacheGroupBox);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label_9 = new QLabel(_nodeCacheGroupBox);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout->addWidget(label_9, 0, 0, 1, 1);

        _nodeCacheMemoryLineEdit = new QLineEdit(_nodeCacheGroupBox);
        _nodeCacheMemoryLineEdit->setObjectName(QString::fromUtf8("_nodeCacheMemoryLineEdit"));
        _nodeCacheMemoryLineEdit->setLayoutDirection(Qt::LeftToRight);
        _nodeCacheMemoryLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _nodeCacheMemoryLineEdit->setReadOnly(true);

        gridLayout->addWidget(_nodeCacheMemoryLineEdit, 0, 1, 1, 1);

        label_12 = new QLabel(_nodeCacheGroupBox);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        gridLayout->addWidget(label_12, 0, 2, 1, 2);

        horizontalSpacer_2 = new QSpacerItem(35, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_2, 0, 4, 1, 1);

        label_5 = new QLabel(_nodeCacheGroupBox);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 1, 0, 1, 1);

        _nodeCacheCapacityLineEdit = new QLineEdit(_nodeCacheGroupBox);
        _nodeCacheCapacityLineEdit->setObjectName(QString::fromUtf8("_nodeCacheCapacityLineEdit"));
        _nodeCacheCapacityLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _nodeCacheCapacityLineEdit->setReadOnly(true);

        gridLayout->addWidget(_nodeCacheCapacityLineEdit, 1, 1, 1, 1);

        label_11 = new QLabel(_nodeCacheGroupBox);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout->addWidget(label_11, 1, 2, 1, 2);

        horizontalSpacer = new QSpacerItem(44, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 1, 4, 1, 1);

        label_3 = new QLabel(_nodeCacheGroupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        _nodeCacheFillingRatioLineEdit = new QLineEdit(_nodeCacheGroupBox);
        _nodeCacheFillingRatioLineEdit->setObjectName(QString::fromUtf8("_nodeCacheFillingRatioLineEdit"));
        _nodeCacheFillingRatioLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(_nodeCacheFillingRatioLineEdit, 2, 1, 1, 1);

        label = new QLabel(_nodeCacheGroupBox);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 2, 2, 1, 1);

        _nodeCacheNbElementsLineEdit = new QLineEdit(_nodeCacheGroupBox);
        _nodeCacheNbElementsLineEdit->setObjectName(QString::fromUtf8("_nodeCacheNbElementsLineEdit"));
        _nodeCacheNbElementsLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _nodeCacheNbElementsLineEdit->setReadOnly(true);

        gridLayout->addWidget(_nodeCacheNbElementsLineEdit, 2, 3, 1, 2);


        verticalLayout_2->addWidget(_nodeCacheGroupBox);

        _dataCacheGroupBox = new QGroupBox(_cacheUsageGroupBox);
        _dataCacheGroupBox->setObjectName(QString::fromUtf8("_dataCacheGroupBox"));
        gridLayout_3 = new QGridLayout(_dataCacheGroupBox);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        label_15 = new QLabel(_dataCacheGroupBox);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        gridLayout_3->addWidget(label_15, 0, 0, 1, 1);

        _dataCacheMemoryLineEdit = new QLineEdit(_dataCacheGroupBox);
        _dataCacheMemoryLineEdit->setObjectName(QString::fromUtf8("_dataCacheMemoryLineEdit"));
        _dataCacheMemoryLineEdit->setLayoutDirection(Qt::LeftToRight);
        _dataCacheMemoryLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _dataCacheMemoryLineEdit->setReadOnly(true);

        gridLayout_3->addWidget(_dataCacheMemoryLineEdit, 0, 1, 1, 1);

        label_13 = new QLabel(_dataCacheGroupBox);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        gridLayout_3->addWidget(label_13, 0, 2, 1, 2);

        horizontalSpacer_5 = new QSpacerItem(37, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_3->addItem(horizontalSpacer_5, 0, 4, 1, 1);

        label_6 = new QLabel(_dataCacheGroupBox);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout_3->addWidget(label_6, 1, 0, 1, 1);

        _dataCacheCapacityLineEdit = new QLineEdit(_dataCacheGroupBox);
        _dataCacheCapacityLineEdit->setObjectName(QString::fromUtf8("_dataCacheCapacityLineEdit"));
        _dataCacheCapacityLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _dataCacheCapacityLineEdit->setReadOnly(true);

        gridLayout_3->addWidget(_dataCacheCapacityLineEdit, 1, 1, 1, 1);

        label_7 = new QLabel(_dataCacheGroupBox);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout_3->addWidget(label_7, 1, 2, 1, 2);

        horizontalSpacer_3 = new QSpacerItem(89, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_3->addItem(horizontalSpacer_3, 1, 4, 1, 1);

        label_4 = new QLabel(_dataCacheGroupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_3->addWidget(label_4, 2, 0, 1, 1);

        _dataCacheFillingRatioLineEdit = new QLineEdit(_dataCacheGroupBox);
        _dataCacheFillingRatioLineEdit->setObjectName(QString::fromUtf8("_dataCacheFillingRatioLineEdit"));
        _dataCacheFillingRatioLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(_dataCacheFillingRatioLineEdit, 2, 1, 1, 1);

        label_2 = new QLabel(_dataCacheGroupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout_3->addWidget(label_2, 2, 2, 1, 1);

        _dataCacheNbElementsLineEdit = new QLineEdit(_dataCacheGroupBox);
        _dataCacheNbElementsLineEdit->setObjectName(QString::fromUtf8("_dataCacheNbElementsLineEdit"));
        _dataCacheNbElementsLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _dataCacheNbElementsLineEdit->setReadOnly(true);

        gridLayout_3->addWidget(_dataCacheNbElementsLineEdit, 2, 3, 1, 2);


        verticalLayout_2->addWidget(_dataCacheGroupBox);


        verticalLayout_3->addWidget(_cacheUsageGroupBox);

        _treeMonitoringGroupBox = new QGroupBox(GvvQCacheUsageWidget);
        _treeMonitoringGroupBox->setObjectName(QString::fromUtf8("_treeMonitoringGroupBox"));
        _treeMonitoringGroupBox->setCheckable(true);
        _treeMonitoringGroupBox->setChecked(false);
        verticalLayout = new QVBoxLayout(_treeMonitoringGroupBox);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        _treeNodeGroupBox = new QGroupBox(_treeMonitoringGroupBox);
        _treeNodeGroupBox->setObjectName(QString::fromUtf8("_treeNodeGroupBox"));
        gridLayout_2 = new QGridLayout(_treeNodeGroupBox);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        label_8 = new QLabel(_treeNodeGroupBox);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout_2->addWidget(label_8, 0, 0, 1, 1);

        _treeEmptyNodeRatioLineEdit = new QLineEdit(_treeNodeGroupBox);
        _treeEmptyNodeRatioLineEdit->setObjectName(QString::fromUtf8("_treeEmptyNodeRatioLineEdit"));
        _treeEmptyNodeRatioLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(_treeEmptyNodeRatioLineEdit, 0, 1, 1, 1);

        label_10 = new QLabel(_treeNodeGroupBox);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        gridLayout_2->addWidget(label_10, 0, 2, 1, 1);

        _treeNbEmptyNodeLineEdit = new QLineEdit(_treeNodeGroupBox);
        _treeNbEmptyNodeLineEdit->setObjectName(QString::fromUtf8("_treeNbEmptyNodeLineEdit"));
        _treeNbEmptyNodeLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _treeNbEmptyNodeLineEdit->setReadOnly(true);

        gridLayout_2->addWidget(_treeNbEmptyNodeLineEdit, 0, 3, 1, 1);


        verticalLayout->addWidget(_treeNodeGroupBox);

        _treeBrickGroupBox = new QGroupBox(_treeMonitoringGroupBox);
        _treeBrickGroupBox->setObjectName(QString::fromUtf8("_treeBrickGroupBox"));
        gridLayout_4 = new QGridLayout(_treeBrickGroupBox);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        label_14 = new QLabel(_treeBrickGroupBox);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        gridLayout_4->addWidget(label_14, 0, 0, 1, 1);

        _treeBrickSparsenessRatioLineEdit = new QLineEdit(_treeBrickGroupBox);
        _treeBrickSparsenessRatioLineEdit->setObjectName(QString::fromUtf8("_treeBrickSparsenessRatioLineEdit"));
        _treeBrickSparsenessRatioLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_4->addWidget(_treeBrickSparsenessRatioLineEdit, 0, 1, 1, 1);

        horizontalSpacer_4 = new QSpacerItem(76, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_4->addItem(horizontalSpacer_4, 0, 2, 1, 1);


        verticalLayout->addWidget(_treeBrickGroupBox);


        verticalLayout_3->addWidget(_treeMonitoringGroupBox);

        verticalSpacer = new QSpacerItem(20, 43, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);


        retranslateUi(GvvQCacheUsageWidget);

        QMetaObject::connectSlotsByName(GvvQCacheUsageWidget);
    } // setupUi

    void retranslateUi(QWidget *GvvQCacheUsageWidget)
    {
        GvvQCacheUsageWidget->setWindowTitle(QApplication::translate("GvvQCacheUsageWidget", "Form", 0, QApplication::UnicodeUTF8));
        _cacheUsageGroupBox->setTitle(QApplication::translate("GvvQCacheUsageWidget", "Real-Time Cache System Monitoring", 0, QApplication::UnicodeUTF8));
        _nodeCacheGroupBox->setTitle(QApplication::translate("GvvQCacheUsageWidget", "Node Cache", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("GvvQCacheUsageWidget", "Memory", 0, QApplication::UnicodeUTF8));
        _nodeCacheMemoryLineEdit->setText(QString());
        label_12->setText(QApplication::translate("GvvQCacheUsageWidget", "Mo", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("GvvQCacheUsageWidget", "Capacity", 0, QApplication::UnicodeUTF8));
        _nodeCacheCapacityLineEdit->setText(QString());
        label_11->setText(QApplication::translate("GvvQCacheUsageWidget", "nodes", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("GvvQCacheUsageWidget", "Filling Ratio (%)", 0, QApplication::UnicodeUTF8));
        _nodeCacheFillingRatioLineEdit->setText(QString());
        label->setText(QApplication::translate("GvvQCacheUsageWidget", "Nb", 0, QApplication::UnicodeUTF8));
        _nodeCacheNbElementsLineEdit->setText(QString());
        _dataCacheGroupBox->setTitle(QApplication::translate("GvvQCacheUsageWidget", "Data Cache", 0, QApplication::UnicodeUTF8));
        label_15->setText(QApplication::translate("GvvQCacheUsageWidget", "Memory", 0, QApplication::UnicodeUTF8));
        _dataCacheMemoryLineEdit->setText(QString());
        label_13->setText(QApplication::translate("GvvQCacheUsageWidget", "Mo", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("GvvQCacheUsageWidget", "Capacity", 0, QApplication::UnicodeUTF8));
        _dataCacheCapacityLineEdit->setText(QString());
        label_7->setText(QApplication::translate("GvvQCacheUsageWidget", "bricks", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("GvvQCacheUsageWidget", "Filling Ratio (%)", 0, QApplication::UnicodeUTF8));
        _dataCacheFillingRatioLineEdit->setText(QString());
        label_2->setText(QApplication::translate("GvvQCacheUsageWidget", "Nb", 0, QApplication::UnicodeUTF8));
        _dataCacheNbElementsLineEdit->setText(QString());
        _treeMonitoringGroupBox->setTitle(QApplication::translate("GvvQCacheUsageWidget", "Real-Time Tree Data Structure Monitoring", 0, QApplication::UnicodeUTF8));
        _treeNodeGroupBox->setTitle(QApplication::translate("GvvQCacheUsageWidget", "Nodes", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("GvvQCacheUsageWidget", "Empty nodes (%)", 0, QApplication::UnicodeUTF8));
        _treeEmptyNodeRatioLineEdit->setText(QString());
        label_10->setText(QApplication::translate("GvvQCacheUsageWidget", "Nb", 0, QApplication::UnicodeUTF8));
        _treeBrickGroupBox->setTitle(QApplication::translate("GvvQCacheUsageWidget", "Bricks", 0, QApplication::UnicodeUTF8));
        label_14->setText(QApplication::translate("GvvQCacheUsageWidget", "Empty voxels (%)", 0, QApplication::UnicodeUTF8));
        _treeBrickSparsenessRatioLineEdit->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class GvvQCacheUsageWidget: public Ui_GvvQCacheUsageWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQCACHEUSAGEWIDGET_H
