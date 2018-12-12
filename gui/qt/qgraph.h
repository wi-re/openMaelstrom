#pragma once
#include <QImage>
#include <QLabel>
#include <QPixmap>
#include <QTimer>
#include <utility/helpers/timer.h>
#include <gui/qt/console.h>

/** This docker displays a window that contains a graph showing the time taken
 * for every timer that is set to be drawn. This docker takes some time for
 * every redraw so keeping it open while running the simulation might make the
 * simulation appear "stuttering", however this graph only requires CPU
 * performance and should not slow doen the simulation as the simulation is ran
 * asynchronously in the background. **/
class QGraph : public Docker {
  Q_OBJECT

  void drawCoords(QPainter &painter);
  void setupValues();
  /** GUI elements**/
  QLabel *m_graphLabel;

  /** The graph is drawn into a pixmap which is based on an image that is
   * display inside of a label in the GUI. **/
  QPixmap m_graphPixmap;
  QImage m_graphImage;
  /** Store some information about the label properties for easier access **/
  QSize m_graphSize;
  QPointF m_topLeftPoint, m_topRightPoint, m_lowLeftPoint, m_lowRightPoint;
  /** Graph Window sizes **/
  int32_t m_width, m_height;
  /** Extent of the data in x (indices) and y (data) **/
  int32_t m_dataMinX, m_dataMaxX;
  float m_dataMinY, m_dataMaxY;
  
  inline double dx_to_sx(const float &i_x) {
    auto scale = static_cast<float>(i_x - m_dataMinX) / static_cast<float>(m_dataMaxX - m_dataMinX);
    return static_cast<int32_t>(m_width * scale + 20);
  }
  inline double dy_to_sy(const float &i_y) {
    const float slope = 1.f * (20 - (m_graphSize.height() - 20)) / (m_dataMaxY - m_dataMinY);
    return m_graphSize.height() - 20 + slope * (static_cast<float>(i_y) - m_dataMinY);
  }
  inline auto convert_data(const float &dx, const float &dy) {
    return QPointF(dx_to_sx(dx), dy_to_sy(dy));
  }

  void draw_overlapping(QPainter &painter);

  bool needsUpdate();

public:
  explicit QGraph( QWidget *parent = nullptr);
  ~QGraph();
  void updatePixMap();

  bool hasChanged;

signals:

public slots:
  void setPixmap(const QPixmap &);
  void resizeEvent(QResizeEvent *);
  void paintEvent(QPaintEvent *);

private:
  QTimer* m_timer;
};
