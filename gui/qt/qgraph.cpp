#include <QColor>
#include <QDebug>
#include <QHBoxLayout>
#include <QMatrix>
#include <QPainter>
#include <gui/qt/qgraph.h>
#include <iostream>
#include <random>
#include <utility/helpers/color.h>
#include <utility/helpers/log.h>
#include <utility/identifier/uniform.h>
#ifndef _WIN32
#include <float.h>
#endif

QGraph::QGraph(QWidget *parent)
    : Docker("Timer graphing window", parent), m_graphLabel(nullptr) {
  m_graphLabel = new QLabel;
  m_graphLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  m_graphPixmap = QPixmap::fromImage(m_graphImage);
  updatePixMap();
  setWidget(m_graphLabel);
  hasChanged = true;

  m_timer = new QTimer;
  connect(m_timer, SIGNAL(timeout()), this, SLOT(update()),
	  Qt::DirectConnection);
  m_timer->setInterval(50);
  m_timer->start(50);
}
QGraph::~QGraph() {  }
void QGraph::setPixmap(const QPixmap &p) {
  //m_graphPixmap = p;
  //m_graphLabel->setPixmap(scaledPixmap());
  //auto lsize = this->size();
  //qDebug() << lsize.width() << " : " << lsize.height() << "\n";
}

void QGraph::updatePixMap() {
  auto labelSize = m_graphLabel->size();
  m_graphImage =
      QImage(labelSize.width(), labelSize.height(), QImage::Format_ARGB32);
  m_graphImage.fill(qRgb(51, 51, 51));
  m_graphPixmap = QPixmap::fromImage(m_graphImage);
  auto scaled = m_graphPixmap.scaled(labelSize);
  m_graphLabel->setPixmap(scaled);
}

void QGraph::resizeEvent(QResizeEvent *e) {
  if (!m_graphPixmap.isNull()) updatePixMap();
  hasChanged = true;
  QDockWidget::resizeEvent(e);
}

void QGraph::setupValues() {
  m_graphSize = this->size();
  int32_t x = m_graphSize.width();
  int32_t y = m_graphSize.height();
  m_topLeftPoint = QPointF(20, 20);
  m_topRightPoint = QPointF(x - 20, 20);
  m_lowLeftPoint = QPointF(20, y - 20);
  m_lowRightPoint = QPointF(x - 20, y - 20);
  m_width = x - 40;
  m_height = y - 40;

  m_dataMinX = 0;
  m_dataMinY = FLT_MAX;
  m_dataMaxX = 0;
  m_dataMaxY = -FLT_MAX;
}

void QGraph::draw_overlapping(QPainter &painter) {
  auto timers_t = TimerManager::getTimers();
  std::vector<Timer *> timers;
  for (auto t : timers_t)
    if (t->graph)
      timers.push_back(t);

  for (auto t : timers) {
    t->generateStats();
    m_dataMaxX = (int32_t)std::max((size_t)m_dataMaxX, t->getSamples().size());
    m_dataMinY = std::min(m_dataMinY, t->getMin());
    m_dataMaxY = std::max(m_dataMaxY, t->getMax());
  }

  std::sort(timers.begin(), timers.end(), [](const auto &lhs, const auto &rhs) {
    return lhs->getSum() > rhs->getSum();
  });

  for (auto t : timers) {
    QPainterPath path;
    path.moveTo(m_lowLeftPoint);

    Timer::sample last_sam;
    for (const auto &sample : t->getSamples()) {
      path.lineTo(convert_data(sample.first, sample.second));
      last_sam = sample;
    }
    auto p = convert_data(last_sam.first, last_sam.second);

    path.lineTo(QPointF(p.x(), m_graphSize.height() - 20));
    path.lineTo(m_lowLeftPoint);
    painter.fillPath(
        path, QBrush(QColor(static_cast<QRgb>(enum_value(t->getColor())))));
  }
  for (auto t : timers) {
    if (t->getSamples().size() > 0) {
      QPainterPath path;

      auto first = t->getSamples()[0];
      path.moveTo(convert_data(first.first, first.second));
      for (const auto &sample : t->getSamples())
        path.lineTo(convert_data(sample.first, sample.second));
      painter.setPen(QColor(static_cast<QRgb>(enum_value(t->getColor()))));
      painter.drawPath(path);
    }
  }
}
bool QGraph::needsUpdate() {
  if (m_graphSize != this->size())
    return true;
  static int32_t frame = INT_MIN;
  if (parameters::frame{} == frame)
    return false;
  frame = parameters::frame{};
  return true;
}
void QGraph::paintEvent(QPaintEvent *e) {
  if (needsUpdate()) {
    setupValues();
    m_graphPixmap = QPixmap::fromImage(m_graphImage);
    QPainter painter(&m_graphPixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    draw_overlapping(painter);
    drawCoords(painter);
    m_graphLabel->setPixmap(m_graphPixmap);

    hasChanged = false;
  }
  QDockWidget::paintEvent(e);
}

void QGraph::drawCoords(QPainter &painter) {
  QPen DarkGrey((QColor(200, 200, 200)), 1);

  painter.setPen(DarkGrey);

  for (int32_t i = 0; i <= m_width / 100; ++i) {
    auto offset_a = m_lowLeftPoint + QPointF(i * 100, 0);
    auto offset_b = offset_a + QPointF(0, 10);
    painter.drawLine(offset_a, offset_b);
    auto scale = (static_cast<double>(m_dataMaxX - m_dataMinX)) /
                 static_cast<double>(m_width);
    auto cx = m_dataMinX + static_cast<double>(i) * 100.0 * scale;
    painter.drawText(
        offset_b + QPointF(3, 0),
        QString::fromStdString(std::to_string(static_cast<int32_t>(cx))), 14,
        0);
  }
  for (int32_t i = 0; i <= m_height / 100; ++i) {
    auto offset_a = m_lowLeftPoint + QPointF(0, -i * 100);
    auto offset_b = offset_a + QPointF(-10, 0);
    painter.drawLine(offset_a, offset_b);

    auto scale = (static_cast<double>(m_dataMaxY - m_dataMinY)) /
                 static_cast<double>(m_height);
    auto cx = m_dataMinY + static_cast<double>(i) * 100.0 * scale;
    painter.rotate(-90);
    auto text_p = offset_b + QPointF(8, -5);
    auto text_yx = QPointF(-text_p.y(), text_p.x());
    painter.drawText(
        text_yx,
        QString::fromStdString(std::to_string(static_cast<double>(cx))), 14, 0);
    painter.rotate(90);
  }

  painter.drawLine(m_lowLeftPoint, m_lowLeftPoint);
  painter.drawLine(m_lowLeftPoint, m_lowRightPoint);
}
