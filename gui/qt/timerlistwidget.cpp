#include <QListWidget>
#include <QListWidgetItem>
#include <QSpacerItem>
#include <gui/qt/timerlistwidget.h>
#include <utility/helpers/timer.h>

TimerListWidget::TimerEntry::TimerEntry(Timer *t, QWidget *parent)
    : QWidget(parent), m_referenceTimer(t) {
  QVBoxLayout *vbx = new QVBoxLayout;
  setLayout(vbx);

  vbx->addWidget(m_minmaxLabel = new QLabel());
  vbx->addWidget(m_avgdevLabel = new QLabel());

  vbx->setContentsMargins(0, 0, 0, 0);
  vbx->setSpacing(0);

  updateLabels();
}

void TimerListWidget::TimerEntry::updateLabels() {
  auto t = m_referenceTimer;
  m_minmaxLabel->setText(
      QString("med: ")
          .append(QString::number(static_cast<float>(t->getMin()), 'f', 2))
          .append(QString("ms max: "))
          .append(QString::number(static_cast<float>(t->getMax()), 'f', 2))
          .append(QString("ms")));
  m_avgdevLabel->setText(
      QString("avg: ")
          .append(QString::number(static_cast<float>(t->getAverage()), 'f', 2))
          .append(QString("ms dev: "))
          .append(QString::number(static_cast<float>(t->getStddev()), 'f', 2))
          .append(QString("ms")));
}

TimerListWidget::TimerListWidget(QWidget *parent)
    : Docker("Timer list", parent) {

  setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  setMinimumSize(384, 512);
  setWidget(m_timerList = new QTreeWidget);
  m_timerList->setHeaderHidden(true);

  for (auto t : TimerManager::getTimers())
    addTimer(t);

  // Setting up time that auto updates the timers every 500ms
  m_updateTimer = new QTimer();
  m_updateTimer->setInterval(500);
  m_updateTimer->start(500);
  connect(m_updateTimer, &QTimer::timeout, [&]() {
    for (auto t : TimerManager::getTimers()) {
      // Check if new timers have been added while the labels are being updated
      bool found = false;
      for (auto w : m_timerEntries) {
        if (w->m_referenceTimer == t) {
          w->updateLabels();
          found = true;
        }
      }
      if (!found)
        addTimer(t);
    }
    QDockWidget::update();
  });
}

void TimerListWidget::addTimer(Timer *t) {
  QTreeWidgetItem *item = new QTreeWidgetItem();

  m_timerList->addTopLevelItem(item);

  QWidget *wid_h = new QWidget;
  QHBoxLayout *hbx = new QHBoxLayout;
  wid_h->setLayout(hbx);
  QLabel *color_label = new QLabel;
  color_label->setMinimumSize(25, 25);
  color_label->setMaximumSize(25, 25);
  auto col_hex = QColor(static_cast<QRgb>(enum_value(t->getColor()))).name();
  QString style = "QLabel { background-color : $COLOR;}";
  style.replace("$COLOR", col_hex);
  color_label->setStyleSheet(style);
  hbx->setContentsMargins(0, 0, 0, 0);
  hbx->setSpacing(0);
  hbx->setAlignment(Qt::AlignLeft);
  hbx->addWidget(color_label);
  hbx->addSpacing(10);
  hbx->addWidget(new QLabel(QString::fromStdString(t->getDecriptor())));
  m_timerList->setItemWidget(item, 0, wid_h);

  auto w = new TimerEntry(t);
  QTreeWidgetItem *item2 = new QTreeWidgetItem();
  item->addChild(item2);
  m_timerList->setItemWidget(item2, 0, w);
  m_timerEntries.push_back(w);
}
