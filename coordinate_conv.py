from collections import namedtuple
import numpy as np
from numpy import sin, cos, sqrt, arctan2, pi
import pymap3d as pm

D2R = pi / 180.0
R2D = 180.0 / pi

# 定数の設定
WGS84 = namedtuple('WGS84', ['re_a',  # [m] WGS84の長軸
                             'eccen1',  # First Eccentricity
                             'eccen1sqr',  # First Eccentricity squared
                             'one_f',  # 扁平率fの1/f（平滑度）
                             're_b',  # [m] WGS84の短軸
                             'e2',  # 第一離心率eの2乗
                             'ed2'  # 第二離心率e'の2乗
                             ])
wgs84 = WGS84(6378137.0, 8.1819190842622e-2, 6.69437999014e-3, 298.257223563,
              6356752.314245, 6.6943799901414e-3, 6.739496742276486e-3)

## テストデータ ##################
# 変換する位置座標(B27 海側)
lat = 38.14227288
lon = 140.93265738
hig = 45.664
# 原点の座標(B09 山側)
lat_o = 38.13877338
lon_o = 140.89872429
hig_o = 44.512
##################################

def test_ecef2ned():    
    dst_ecef = my_blh2ecef(lat, lon, hig)
    obs_ecef = my_blh2ecef(lat_o, lon_o, hig_o)

    ned_lib_ans = pm.ecef2ned(dst_ecef[0], dst_ecef[1], dst_ecef[2], 
                              lat_o, lon_o, hig_o)
    
    my_ned_ans = my_ecef2ned(dst_ecef[0], dst_ecef[1], dst_ecef[2],
                             obs_ecef[0], obs_ecef[1], obs_ecef[2])
    print("result_ned : ", my_ned_ans)
    print("answer_ned : ", ned_lib_ans)

    diff = [abs(j - i) for (i, j) in zip(my_ned_ans, ned_lib_ans)]
    print("ned diff : ", diff)
    return my_ned_ans

def test_ned2ecef():
    ned_pos = test_ecef2ned()
    obs_ecef = my_blh2ecef(lat_o, lon_o, hig_o)
    result_ecef = my_ned2ecef(ned_pos[0], ned_pos[1], ned_pos[2],
                              obs_ecef[0], obs_ecef[1], obs_ecef[2])
    
    answer_ecef = pm.ned2ecef(ned_pos[0], ned_pos[1], ned_pos[2],
                              lat_o, lon_o, hig_o)
    print("result_ecef : ", result_ecef)
    print("answer_ecef : ", answer_ecef)

    diff = [abs(j - i) for (i, j) in zip(result_ecef, answer_ecef)]
    print("ecef diff : ", diff)
    return 

def my_blh2ecef(lat, lon, height):
    """WGS84(緯度／経度／高度)からECEF座標に変換
    WGS84座標系からECEF座標系に変換する
    引数 :
        lat : WGS84 緯度 [deg]
        lon : WGS84 経度 [deg]
        height : WGS84 ジオイド高 [m]
    """

    s_lat = sin(lat * D2R)
    c_lat = cos(lat * D2R)
    s_lon = sin(lon * D2R)
    c_lon = cos(lon * D2R)

    re_n = wgs84.re_a / sqrt(1 - (wgs84.eccen1 * s_lat) ** 2)
    ecef_x = (re_n + height) * c_lat * c_lon
    ecef_y = (re_n + height) * c_lat * s_lon
    ecef_z = (re_n * (1 - wgs84.eccen1sqr) + height) * s_lat

    return [ecef_x, ecef_y, ecef_z]

def my_ecef2blh(x, y, z):
    """ECEF座標からWGS84(緯度/経度/高度)に変換
    引数 :
        x,y,z: ECEF座標での位置[m]
    返り値 :
        phi: 緯度[deg]
        lam: 経度[deg]
        height: WGS84の平均海面高度[m]
    """

    p = sqrt(x ** 2 + y ** 2)  # 現在位置での地心からの距離[m]
    theta = arctan2(z * wgs84.re_a, p * wgs84.re_b)  # [rad]

    phi = R2D * arctan2(z + wgs84.ed2 * wgs84.re_b * sin(theta) ** 3,
                        p - wgs84.e2 * wgs84.re_a * cos(theta) ** 3)
    lam = R2D * arctan2(y, x)
    height = p / cos(D2R * phi) - wgs84.re_a / \
        sqrt(1.0 - wgs84.e2 * sin(D2R * phi) ** 2)

    return [phi, lam, height]

def my_ned2ecef(n, e, d, xr, yr, zr):
    """ NED座標系からECEF座標系へ座標変換
    引数 :
        n,e,d    : NED座標系上の着目点(変換したい点)[m]
        xr,yr,zr : ECEF座標系上の原点:[m]
    返り値 :
        x,y,z : ECEF座標系上の着目点[m]
    """

    # NED座標の緯度経度
    phi, lam, _ = my_ecef2blh(xr, yr, zr)
    phi *= D2R
    lam *= D2R

    x = (-sin(phi) * cos(lam) * n) + \
        (-sin(lam)            * e) + \
        (-cos(phi) * cos(lam) * d) + xr
    y = (-sin(phi) * sin(lam) * n) + \
        (cos(lam)             * e) + \
        (-cos(phi) * sin(lam) * d) + yr
    z = (cos(phi)             * n) + \
        (-sin(phi)            * d) + zr

    return [x, y, z]

def my_ecef2ned(x, y, z, xo, yo, zo):
    """ ECEF座標系からNED座標系へ座標変換
    引数 :
        x,y,z    : ECEF座標系上の着目点(変換したい点)[m]
        xo,yo,zo : ECEF座標系上の原点[m]
    返り値 :
        n,e,d    : NED座標系の着目点[m]
    """

    # NED座標の緯度経度
    phi, lam, _ = my_ecef2blh(xo, yo, zo)
    phi *= D2R
    lam *= D2R

    n = -sin(phi) * cos(lam) * (x - xo) + \
        -sin(phi) * sin(lam) * (y - yo) + \
        cos(phi)             * (z - zo)
    e = -sin(lam)            * (x - xo) + \
        cos(lam)             * (y - yo)
    d = -cos(phi) * cos(lam) * (x - xo) + \
        -cos(phi) * sin(lam) * (y - yo) + \
        -sin(phi)            * (z - zo)

    return [n, e, d]

#test_ecef2ned()
test_ned2ecef()