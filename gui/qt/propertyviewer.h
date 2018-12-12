#pragma once
#include "ui_mainwindow.h"
#include <QTreeWidget>
#include <QTimer>
#include <gui/qt/console.h>

/** This widget displays all uniforms in the simulation and allows editing them
 * (if they are not fixed values) while the simulation is running. The GUI
 * elements for the parameters are created by delegate.h. **/
class PropertyViewer : public Docker {
	Q_OBJECT
protected:
	QTreeWidget * m_parameterTree = nullptr;
	QTimer* m_refreshTimer = nullptr;
public:
	explicit PropertyViewer(QWidget *parent = nullptr);
};
