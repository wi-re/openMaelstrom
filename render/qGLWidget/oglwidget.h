#pragma once
#include <QOpenGLFunctions_3_3_Compatibility>
#include <QTimer>
#include <QtGui/QInputEvent>
#include <QtOpenGL/QGLWidget>
#include <QOpenGLPaintDevice>
#include <QOpenGLShaderProgram>
#include <render/qGLWidget/base_renderer.h>
#include <render/util/camera.h>
#include <render/util/gl_delegates.h>

/** Keybinds (might be out of date): 
H: Toggle display of onscreen information
P: Toggle the running state of the simulation
G: Toggle between GPU and CPU simulation (currently not working due to a driver bug)
X: Update the snapshot of the simulation
Z: Reload the snapshot
C: Clear the snapshot
M: Flip the color mapping
T: Toggle auto-scaling for color mapping
-: Return to default color mapping based on configuration

If these hotkeys are used with alt they only change the colormap
1: Display density color mapped using jet (fixed)
2: Display neighbor list length using inferno
3: Display MLM resolution using viridi
4: Display velocity using Blue/White (fixed)
5: Display lifetime of particles using plasma
6: Display lifetime of particles using plasma (fixed)
7: Display volume of particles using jet
8: Display support radius of particles using jet
9: Display surface distance using plasma
0: Display adaptive classifcation using Red/Blue (fixed)

W: move camera forward
S: move camera backward
A: move camera left
D: move camera right
E: Raise camera
Q: Lower camera
F1: Display simulation from right
F2: Display simulation from front
F3: Display simulation from left
F4: Display simulation from behind
F5: Display simulation from above
F6: Display simulation from below
R: Return to camera position in configuration
L: Print camera information to log

LMB: rotate camera (first person)
RMB: Slide camera forward and back
MMB: Slide camera on camera plane
**/
/** This widget is used to display the simulation and some overlay text. **/
class OGLWidget : public QGLWidget, protected QOpenGLFunctions_3_3_Compatibility {
	std::vector<Renderer *> m_renderFunctions;
  std::vector<Renderer *> m_volumeRenderFunctions;
	bool m_showText = true;
	int m_frame = 0;
	QOpenGLContext *m_context = nullptr;
	QOpenGLPaintDevice *m_device = nullptr;
	QTimer m_timer;
#ifdef WIN32
	FILE *m_ffmpegPipe = nullptr;
#endif
	float m_frameTimer = 0.f;

	// These 3 members contain a map of uniforms/vbos that need to be set.
	std::map<array_enum, cuda_buffer_base *> m_arrayMappings;
	std::map<uniforms, gl_uniform_base *> m_parameterMappings;
	std::map<std::string, gl_uniform_base *> m_uniformMappings;
public:  
  OGLWidget(QWidget *parent = nullptr);
  /** The destructor has to close the pipe used for ffmpeg/ **/
  ~OGLWidget();

  /** This function is used to bind all bindings to a specific program. **/
  void bind(QOpenGLShaderProgram* program);

  /** Callback functions. The events are automatically passed onto the camera
   * instance. **/
  virtual void resizeGL(int w, int h);
  virtual void resizeEvent(QResizeEvent *event);
  virtual void keyPressEvent(QKeyEvent *event);
  virtual void keyReleaseEvent(QKeyEvent *event);
  virtual void mousePressEvent(QMouseEvent *event);
  virtual void mouseReleaseEvent(QMouseEvent *event);
  virtual void mouseDoubleClickEvent(QMouseEvent *event);
  virtual void mouseMoveEvent(QMouseEvent *event);
  virtual void wheelEvent(QWheelEvent *event);
  virtual void setKeyboardModifiers(QInputEvent *event);

protected:
  /** This function is used to call all render functions. It also sets up the
   * required matrix modes and generic OpenGL state so we can use "normal"
   * OpenGL functions. This function also quits the application if the threshold
   * of frames or simulation time has been met.
   * THIS FUNCTION LOCKS THE SIMULATION MUTEX. **/
  void renderFunctions();
  /** This function draws on screen information, e.g. common parameters like
   * number of particles or timestep. Depending on the modules active different
   * elements will be displayed. depends on the m_showText variable. **/
  void drawText(QPainter &painter);
  /** This function is used to initialize the proper global OpenGL state.
   * Additionally this function sets up the camera for rendering and fills the
   * lists of uniforms with all default parameters. Additionally it creates
   * view_matrix and perspective_matrix. **/
  virtual void initializeGL();
  /** Empty function. **/
  virtual void draw();
  /** This function just passes along the call to update() which in turn calls
   * paintEvent on this window so we can redraw the frame. **/
  virtual void paintGL() { QGLWidget::update(); }
  /** This function handles calling the actual render functions inside of a
   * properly setup QPainter. After the render functions this function will draw
   * the on screen text display and finally grab the frame from the framebuffer
   * to pipe it into ffmpeg (windows only) for recording. **/
  void paintEvent(QPaintEvent *event);
};
