/********************************************************************************
** Form generated from reading UI file 'GvQMainWindow.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVQMAINWINDOW_H
#define UI_GVQMAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QStatusBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvQMainWindow
{
public:
    QAction *actionOpen;
    QAction *actionExit;
    QAction *actionHelp;
    QAction *actionAbout;
    QAction *actionFull_Screen;
    QAction *actionPreferences;
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QGroupBox *_3DViewGroupBox;
    QMenuBar *menubar;
    QMenu *menuFile;
    QMenu *menu;
    QMenu *menuDisplay;
    QMenu *menuEdition;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *GvQMainWindow)
    {
        if (GvQMainWindow->objectName().isEmpty())
            GvQMainWindow->setObjectName(QString::fromUtf8("GvQMainWindow"));
        GvQMainWindow->resize(541, 463);
        actionOpen = new QAction(GvQMainWindow);
        actionOpen->setObjectName(QString::fromUtf8("actionOpen"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/Icons/FileOpen.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionOpen->setIcon(icon);
        actionExit = new QAction(GvQMainWindow);
        actionExit->setObjectName(QString::fromUtf8("actionExit"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/Icons/Exit.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionExit->setIcon(icon1);
        actionHelp = new QAction(GvQMainWindow);
        actionHelp->setObjectName(QString::fromUtf8("actionHelp"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/Icons/Help.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionHelp->setIcon(icon2);
        actionAbout = new QAction(GvQMainWindow);
        actionAbout->setObjectName(QString::fromUtf8("actionAbout"));
        actionFull_Screen = new QAction(GvQMainWindow);
        actionFull_Screen->setObjectName(QString::fromUtf8("actionFull_Screen"));
        actionPreferences = new QAction(GvQMainWindow);
        actionPreferences->setObjectName(QString::fromUtf8("actionPreferences"));
        centralwidget = new QWidget(GvQMainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        _3DViewGroupBox = new QGroupBox(centralwidget);
        _3DViewGroupBox->setObjectName(QString::fromUtf8("_3DViewGroupBox"));

        verticalLayout->addWidget(_3DViewGroupBox);

        GvQMainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(GvQMainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 541, 21));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        menu = new QMenu(menubar);
        menu->setObjectName(QString::fromUtf8("menu"));
        menuDisplay = new QMenu(menubar);
        menuDisplay->setObjectName(QString::fromUtf8("menuDisplay"));
        menuEdition = new QMenu(menubar);
        menuEdition->setObjectName(QString::fromUtf8("menuEdition"));
        GvQMainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(GvQMainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        GvQMainWindow->setStatusBar(statusbar);

        menubar->addAction(menuFile->menuAction());
        menubar->addAction(menuEdition->menuAction());
        menubar->addAction(menuDisplay->menuAction());
        menubar->addAction(menu->menuAction());
        menuFile->addAction(actionOpen);
        menuFile->addSeparator();
        menuFile->addAction(actionExit);
        menu->addSeparator();
        menu->addAction(actionHelp);
        menu->addSeparator();
        menu->addAction(actionAbout);
        menuDisplay->addAction(actionFull_Screen);
        menuEdition->addAction(actionPreferences);

        retranslateUi(GvQMainWindow);

        QMetaObject::connectSlotsByName(GvQMainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *GvQMainWindow)
    {
        GvQMainWindow->setWindowTitle(QApplication::translate("GvQMainWindow", "GigaVoxels Viewer", 0, QApplication::UnicodeUTF8));
        actionOpen->setText(QApplication::translate("GvQMainWindow", "Open", 0, QApplication::UnicodeUTF8));
        actionExit->setText(QApplication::translate("GvQMainWindow", "Exit", 0, QApplication::UnicodeUTF8));
        actionHelp->setText(QApplication::translate("GvQMainWindow", "Help", 0, QApplication::UnicodeUTF8));
        actionAbout->setText(QApplication::translate("GvQMainWindow", "About", 0, QApplication::UnicodeUTF8));
        actionFull_Screen->setText(QApplication::translate("GvQMainWindow", "Full Screen", 0, QApplication::UnicodeUTF8));
        actionPreferences->setText(QApplication::translate("GvQMainWindow", "Preferences", 0, QApplication::UnicodeUTF8));
        _3DViewGroupBox->setTitle(QString());
        menuFile->setTitle(QApplication::translate("GvQMainWindow", "File", 0, QApplication::UnicodeUTF8));
        menu->setTitle(QApplication::translate("GvQMainWindow", "?", 0, QApplication::UnicodeUTF8));
        menuDisplay->setTitle(QApplication::translate("GvQMainWindow", "Display", 0, QApplication::UnicodeUTF8));
        menuEdition->setTitle(QApplication::translate("GvQMainWindow", "Edition", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvQMainWindow: public Ui_GvQMainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVQMAINWINDOW_H
