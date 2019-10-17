Getting started
---

This document intends to provide a brief overview on how to start and run the simulation, including command line options and key bindings.

Simulation
---
The simulation itself can be run in either a GUI or command line only interface. The GUI is based on Qt and can be found under `bin/gui`. The command line only interface does not require an OpenGL context to be created and avoids some overhead due to visualizing the data and can be found under `bin/consoleParticles`. The recommend program is the GUI based interface.

The configuration format is explained [here](./config.md).

__Command Line Options__

|Name|Example|Description|
|---|---|---|
|__-h__ [--help]||Print a help message on the command line|
|__-l__[--list] |  | Lists all known configuration files saved in cfg/configs.sph |
|__--config__ _arg_| --config ../Dambreak/config.json| Load configuration from file|
|__-c__ [--config_id] _arg_| -c 0| Load from known configurations by index| 
|__-s__ [--snap]_arg_| -s frame_1600.dump| Reload simulation timepoint| 
|__-o__ [--option] _arg_| -o 1| Pick optional parameters in config|
|__-j__ [--json]_arg_| -j modules.adaptive=true color_map.min=0|Manual parameters |
|__-f__ [--frame]_arg_| -f 10| Stop after _arg_ frames|
|__-t__ [--time]_arg_| -t 30.0| Stop after _arg_ seconds simulated |
|__--pause__ _arg_| --pause 30.0| Pause after _arg_ seconds simulated |
|__-v__ [--verbose]| |Print additional log information|
|__--no-timer__| |Hide timer output on command line|
|__--memory__| |Prints information about memory consumption|
|__-i__ [--info]| |Prints additional simulation information|
|__-p__ [--params]| |Prints stats about some parameters |
|__--log__ _arg_| --log ~/logs/ | Log to file _arg_|
|__--config_log__ _arg_| ~/config.json| Store actual configuration in _arg_ |
|__--neighbordump__ _arg_| ~/neighs.dump| Store neighbor list at end in _arg_|
|__-r__ [--record] _arg_| -r ~/video.mp4| Record OpenGL output to file using ffmpeg|
|__--render__ _arg_| --render ~/simulation/| Store ray traced images in folder|
|__-G__ _arg_| -G .\*density.\*|Matched functions are executed on GPU|
|__-H__ _arg_| -H .\*density.\*|Matched functions are executed on CPU|
|__-D__ _arg_| -D .\*density.\*|Matched functions are executed in Debug Mode|

If the simulation is started with either __-f__ or __-t__ a progress bar, with estimated time remaining, is shown on the command line.

__Hotkeys__

|Modifier|Key|Function|
|---|---|---|
||J|Hide/Show color map on side of window (stored in images)|
||H|Hide/Show simulation stats on window (stored in images)|
||P|Pause/Resume simulation|
|Shift|P|Do single timestep of simulation|
||G|Toggle between CPU and GPU execution model|
||X|Save current simulation point in memory|
||Z|Return to saved simulation point|
||C|Clear saved simulation point|
||M|Flip color map|
||V|Display inlet/outlet volumes|
||T|Toggle auto scaling for color map on/off|
||1-0|Load preset color mapping configuration|
|Alt|1-0|Only loads preset color map color values|
||-|Reset to color mapping in configuration|
|Alt|-|Reset to color map values based on configuration|
||O|Store simulation standpoint in file|
||I|Store particles as Alembic file|
||F7|Toggle camera tracking of AABB on/off|
||F8|Toggle particle picker on/off|
||F9|Toggle ray tracing on/off|
||F10|Toggle particle color mapping with directions|
|Alt|F10|Flip scale of visualized directions to inverse|
|Shift|F10|Set visualized directions to auto scale|
||LMB|Look around via mouse interaction|
||RMB|Move camera forward and backward|
||WASD|Lateral camera movement|
||EQ|Camera up/down movement|
||F1-F6|Preset Camera Positions|
||L|Show current camera position in log|
|Shift|L|Save current camera position|
||R|Reset camera position to stored position|
