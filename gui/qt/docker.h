#pragma once
#include "ui_mainwindow.h"
#include <QDockWidget>
#include <QHBoxLayout>
#include <QPushButton>

/** Helper class to provide basic QDockWidget functionality. **/
class Docker : public QDockWidget {
	Q_OBJECT;
public:
	/** GUI Fields **/
	QPushButton *m_minButton, *m_maxButton, *m_closeButton;
	Docker(QString name = "", QWidget *parent = nullptr);

	QHBoxLayout* m_layOut;
};
