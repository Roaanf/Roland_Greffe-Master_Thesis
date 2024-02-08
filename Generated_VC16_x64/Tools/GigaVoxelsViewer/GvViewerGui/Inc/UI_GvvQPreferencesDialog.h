/********************************************************************************
** Form generated from reading UI file 'GvvQPreferencesDialog.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQPREFERENCESDIALOG_H
#define UI_GVVQPREFERENCESDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QSpacerItem>
#include <QtGui/QTabWidget>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQPreferencesDialog
{
public:
    QVBoxLayout *verticalLayout_2;
    QLabel *label;
    QTabWidget *mSettingsTabs;
    QWidget *mDisplayTab;
    QVBoxLayout *verticalLayout;
    QGroupBox *mUnitsFormatGroup;
    QHBoxLayout *horizontalLayout;
    QLabel *labelPositionFormatCombo;
    QSpacerItem *horizontalSpacer;
    QToolButton *_3DWindowBackgroundColorToolButton;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QCheckBox *_nodeHasBrickTerminalCheckBox;
    QSpacerItem *horizontalSpacer_2;
    QToolButton *_nodeHasBrickTerminalColorToolButton;
    QCheckBox *_nodeHasBrickNotTerminalCheckBox;
    QSpacerItem *horizontalSpacer_3;
    QToolButton *_nodeHasBrickNotTerminalColorToolButton;
    QCheckBox *_nodeIsBrickNotInCacheCheckBox;
    QSpacerItem *horizontalSpacer_4;
    QToolButton *_nodeIsBrickNotInCacheColorToolButton;
    QCheckBox *_nodeEmptyOrConstantCheckBox;
    QSpacerItem *horizontalSpacer_5;
    QToolButton *_nodeEmptyOrConstantColorToolButton;
    QSpacerItem *spacerItem;
    QDialogButtonBox *mDialogButtons;

    void setupUi(QDialog *GvvQPreferencesDialog)
    {
        if (GvvQPreferencesDialog->objectName().isEmpty())
            GvvQPreferencesDialog->setObjectName(QString::fromUtf8("GvvQPreferencesDialog"));
        GvvQPreferencesDialog->resize(326, 439);
        verticalLayout_2 = new QVBoxLayout(GvvQPreferencesDialog);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        label = new QLabel(GvvQPreferencesDialog);
        label->setObjectName(QString::fromUtf8("label"));
        label->setPixmap(QPixmap(QString::fromUtf8(":/icons/Icons/GigaVoxelsLogo_div2.png")));
        label->setScaledContents(true);

        verticalLayout_2->addWidget(label);

        mSettingsTabs = new QTabWidget(GvvQPreferencesDialog);
        mSettingsTabs->setObjectName(QString::fromUtf8("mSettingsTabs"));
        mSettingsTabs->setAutoFillBackground(true);
        mDisplayTab = new QWidget();
        mDisplayTab->setObjectName(QString::fromUtf8("mDisplayTab"));
        verticalLayout = new QVBoxLayout(mDisplayTab);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        mUnitsFormatGroup = new QGroupBox(mDisplayTab);
        mUnitsFormatGroup->setObjectName(QString::fromUtf8("mUnitsFormatGroup"));
        horizontalLayout = new QHBoxLayout(mUnitsFormatGroup);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        labelPositionFormatCombo = new QLabel(mUnitsFormatGroup);
        labelPositionFormatCombo->setObjectName(QString::fromUtf8("labelPositionFormatCombo"));

        horizontalLayout->addWidget(labelPositionFormatCombo);

        horizontalSpacer = new QSpacerItem(136, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        _3DWindowBackgroundColorToolButton = new QToolButton(mUnitsFormatGroup);
        _3DWindowBackgroundColorToolButton->setObjectName(QString::fromUtf8("_3DWindowBackgroundColorToolButton"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/Icons/Colors.png"), QSize(), QIcon::Normal, QIcon::Off);
        _3DWindowBackgroundColorToolButton->setIcon(icon);

        horizontalLayout->addWidget(_3DWindowBackgroundColorToolButton);


        verticalLayout->addWidget(mUnitsFormatGroup);

        groupBox = new QGroupBox(mDisplayTab);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        _nodeHasBrickTerminalCheckBox = new QCheckBox(groupBox);
        _nodeHasBrickTerminalCheckBox->setObjectName(QString::fromUtf8("_nodeHasBrickTerminalCheckBox"));

        gridLayout->addWidget(_nodeHasBrickTerminalCheckBox, 0, 0, 1, 2);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_2, 0, 3, 1, 1);

        _nodeHasBrickTerminalColorToolButton = new QToolButton(groupBox);
        _nodeHasBrickTerminalColorToolButton->setObjectName(QString::fromUtf8("_nodeHasBrickTerminalColorToolButton"));
        _nodeHasBrickTerminalColorToolButton->setIcon(icon);

        gridLayout->addWidget(_nodeHasBrickTerminalColorToolButton, 0, 4, 1, 1);

        _nodeHasBrickNotTerminalCheckBox = new QCheckBox(groupBox);
        _nodeHasBrickNotTerminalCheckBox->setObjectName(QString::fromUtf8("_nodeHasBrickNotTerminalCheckBox"));

        gridLayout->addWidget(_nodeHasBrickNotTerminalCheckBox, 1, 0, 1, 3);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_3, 1, 3, 1, 1);

        _nodeHasBrickNotTerminalColorToolButton = new QToolButton(groupBox);
        _nodeHasBrickNotTerminalColorToolButton->setObjectName(QString::fromUtf8("_nodeHasBrickNotTerminalColorToolButton"));
        _nodeHasBrickNotTerminalColorToolButton->setIcon(icon);

        gridLayout->addWidget(_nodeHasBrickNotTerminalColorToolButton, 1, 4, 1, 1);

        _nodeIsBrickNotInCacheCheckBox = new QCheckBox(groupBox);
        _nodeIsBrickNotInCacheCheckBox->setObjectName(QString::fromUtf8("_nodeIsBrickNotInCacheCheckBox"));

        gridLayout->addWidget(_nodeIsBrickNotInCacheCheckBox, 2, 0, 1, 2);

        horizontalSpacer_4 = new QSpacerItem(59, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_4, 2, 2, 1, 2);

        _nodeIsBrickNotInCacheColorToolButton = new QToolButton(groupBox);
        _nodeIsBrickNotInCacheColorToolButton->setObjectName(QString::fromUtf8("_nodeIsBrickNotInCacheColorToolButton"));
        _nodeIsBrickNotInCacheColorToolButton->setIcon(icon);

        gridLayout->addWidget(_nodeIsBrickNotInCacheColorToolButton, 2, 4, 1, 1);

        _nodeEmptyOrConstantCheckBox = new QCheckBox(groupBox);
        _nodeEmptyOrConstantCheckBox->setObjectName(QString::fromUtf8("_nodeEmptyOrConstantCheckBox"));

        gridLayout->addWidget(_nodeEmptyOrConstantCheckBox, 3, 0, 1, 1);

        horizontalSpacer_5 = new QSpacerItem(85, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_5, 3, 1, 1, 3);

        _nodeEmptyOrConstantColorToolButton = new QToolButton(groupBox);
        _nodeEmptyOrConstantColorToolButton->setObjectName(QString::fromUtf8("_nodeEmptyOrConstantColorToolButton"));
        _nodeEmptyOrConstantColorToolButton->setIcon(icon);

        gridLayout->addWidget(_nodeEmptyOrConstantColorToolButton, 3, 4, 1, 1);


        verticalLayout->addWidget(groupBox);

        spacerItem = new QSpacerItem(20, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(spacerItem);

        mSettingsTabs->addTab(mDisplayTab, QString());

        verticalLayout_2->addWidget(mSettingsTabs);

        mDialogButtons = new QDialogButtonBox(GvvQPreferencesDialog);
        mDialogButtons->setObjectName(QString::fromUtf8("mDialogButtons"));
        mDialogButtons->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        verticalLayout_2->addWidget(mDialogButtons);

        QWidget::setTabOrder(mDialogButtons, mSettingsTabs);

        retranslateUi(GvvQPreferencesDialog);
        QObject::connect(mDialogButtons, SIGNAL(accepted()), GvvQPreferencesDialog, SLOT(accept()));
        QObject::connect(mDialogButtons, SIGNAL(rejected()), GvvQPreferencesDialog, SLOT(reject()));

        mSettingsTabs->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(GvvQPreferencesDialog);
    } // setupUi

    void retranslateUi(QDialog *GvvQPreferencesDialog)
    {
        GvvQPreferencesDialog->setWindowTitle(QApplication::translate("GvvQPreferencesDialog", "Preferences", 0, QApplication::UnicodeUTF8));
        label->setText(QString());
        mUnitsFormatGroup->setTitle(QApplication::translate("GvvQPreferencesDialog", "3D Window", 0, QApplication::UnicodeUTF8));
        labelPositionFormatCombo->setText(QApplication::translate("GvvQPreferencesDialog", "Background Color", 0, QApplication::UnicodeUTF8));
        _3DWindowBackgroundColorToolButton->setText(QApplication::translate("GvvQPreferencesDialog", "...", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvvQPreferencesDialog", "Data Structure Appearance", 0, QApplication::UnicodeUTF8));
        _nodeHasBrickTerminalCheckBox->setText(QApplication::translate("GvvQPreferencesDialog", "Node has brick and is terminal", 0, QApplication::UnicodeUTF8));
        _nodeHasBrickTerminalColorToolButton->setText(QApplication::translate("GvvQPreferencesDialog", "...", 0, QApplication::UnicodeUTF8));
        _nodeHasBrickNotTerminalCheckBox->setText(QApplication::translate("GvvQPreferencesDialog", "Node has brick and is not terminal", 0, QApplication::UnicodeUTF8));
        _nodeHasBrickNotTerminalColorToolButton->setText(QApplication::translate("GvvQPreferencesDialog", "...", 0, QApplication::UnicodeUTF8));
        _nodeIsBrickNotInCacheCheckBox->setText(QApplication::translate("GvvQPreferencesDialog", "Node is a brick (not in cache)", 0, QApplication::UnicodeUTF8));
        _nodeIsBrickNotInCacheColorToolButton->setText(QApplication::translate("GvvQPreferencesDialog", "...", 0, QApplication::UnicodeUTF8));
        _nodeEmptyOrConstantCheckBox->setText(QApplication::translate("GvvQPreferencesDialog", "Node empty or constant", 0, QApplication::UnicodeUTF8));
        _nodeEmptyOrConstantColorToolButton->setText(QApplication::translate("GvvQPreferencesDialog", "...", 0, QApplication::UnicodeUTF8));
        mSettingsTabs->setTabText(mSettingsTabs->indexOf(mDisplayTab), QApplication::translate("GvvQPreferencesDialog", "Display", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQPreferencesDialog: public Ui_GvvQPreferencesDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQPREFERENCESDIALOG_H
