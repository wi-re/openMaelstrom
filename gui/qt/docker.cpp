#include <QFontDatabase>
#include <QGroupBox>
#include <QLabel>
#include <QScrollBar>
#include <QStyle>
#include <QVBoxLayout>
#include <ctime>
#include <gui/qt/console.h>
#include <utility/helpers/log.h>

Docker::Docker(QString name, QWidget *parent) : QDockWidget(name, parent) {
  setAllowedAreas(Qt::AllDockWidgetAreas);
  QGroupBox *groupBox = new QGroupBox(tr(""));
  QHBoxLayout *vbox = new QHBoxLayout;
  auto topWidget = new QWidget();
  m_layOut = new QHBoxLayout;
  groupBox->setLayout(vbox);
  m_layOut->addWidget(groupBox);
  QStyle *appStyle = qApp->style();

  m_layOut->addWidget(m_maxButton = new QPushButton());
  m_layOut->addWidget(m_closeButton = new QPushButton());

  auto setSpacing = [&](auto wid) {
    wid->setContentsMargins(0, 0, 0, 0);
    wid->setSpacing(0);
  };
  setSpacing(m_layOut);

  topWidget->setLayout(m_layOut);
  setTitleBarWidget(topWidget);

  m_maxButton->setObjectName("SizeButton");
  m_maxButton->setIcon(appStyle->standardIcon(QStyle::SP_TitleBarMaxButton));

  m_closeButton->setObjectName("ExitButton");
  m_closeButton->setIcon(
      appStyle->standardIcon(QStyle::SP_TitleBarCloseButton));

  m_maxButton->setMaximumWidth(25);
  m_closeButton->setMaximumWidth(25);

  connect(
      this, &QDockWidget::dockLocationChanged,
      [=]([[maybe_unused]] Qt::DockWidgetArea area) { m_maxButton->show(); });

  connect(m_closeButton, &QPushButton::released, [=]() { this->hide(); });
  connect(m_maxButton, &QPushButton::released, [=]() {
    if (!isFloating()) {
      setFloating(true);
      m_maxButton->hide();
    }
  });
}
