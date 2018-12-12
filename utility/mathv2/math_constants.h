#pragma once
#include <cmath>
/* bounding values */
#define MATH_INF_F HUGE_VALF
#define MATH_INF HUGE_VAL
#define MATH_NAN_F NAN

/* simple values */
#define MATH_ZERO_F           0.0f
#define MATH_ONE_F            1.0f
#define MATH_ZERO             0.0
#define MATH_ONE              1.0
#define MATH_THIRD_F          0.333333333f
#define MATH_THIRD            3.3333333333333333e-1
#define MATH_TWOTHIRD_F       0.666666666f
#define MATH_TWOTHIRD         6.6666666666666667e-1

/* sqrt defines*/
#define MATH_SQRT_HALF_F      0.707106781f
#define MATH_SQRT_HALF        7.0710678118654757e-1
#define MATH_SQRT_TWO_F       1.414213562f
#define MATH_SQRT_TWO         1.4142135623730951e+0

/* PI defines*/
#define MATH_PI_F             3.141592654f
#define MATH_PI               3.1415926535897931e+0
#define MATH_PIO4_F           0.785398163f
#define MATH_PIO4             7.8539816339744828e-1
#define MATH_PIO2_F           1.570796327f
#define MATH_PIO2             1.5707963267948966e+0
#define MATH_3PIO4_F          2.356194490f
#define MATH_3PIO4            2.3561944901923448e+0
#define MATH_4PIO3_F		  4.188790204f
#define MATH_4PIO3			  4.18879020478639098+0
#define MATH_4PIO3_1_F		  0.238732414f
#define MATH_4PIO3_1		  2.38732414637843003-1
#define MATH_2_OVER_PI_F      0.636619772f
#define MATH_2_OVER_PI        6.3661977236758138e-1
#define MATH_SQRT_2_OVER_PI_F 0.797884561f
#define MATH_SQRT_2_OVER_PI   7.9788456080286536e-1
#define MATH_SQRT_2PI_F       2.506628274f
#define MATH_SQRT_2PI         2.5066282746310007e+0
#define MATH_SQRT_PIO2_F      1.253314137f
#define MATH_SQRT_PIO2        1.2533141373155003e+0

/* LOG defines */
#define MATH_L2E_F            1.442695041f // log_2(e)
#define MATH_L2E              1.4426950408889634e+0
#define MATH_L2T_F            3.321928094f // log_2(10)
#define MATH_L2T              3.3219280948873622e+0
#define MATH_LG2_F            0.301029996f // log_10(2)
#define MATH_LG2              3.0102999566398120e-1
#define MATH_LGE_F            0.434294482f // log_10(e)
#define MATH_LGE              4.3429448190325182e-1
#define MATH_LN2_F            0.693147181f // ln(2)
#define MATH_LN2              6.9314718055994529e-1
#define MATH_LNT_F            2.302585093f // ln(10)
#define MATH_LNT              2.3025850929940459e+0
#define MATH_LNPI_F           1.144729886f // ln(pi)
#define MATH_LNPI             1.1447298858494002e+0