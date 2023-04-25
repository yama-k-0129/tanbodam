import pandas as pd 
import math
import streamlit as st

def paddy_flow(pnum, ir, r_in, etp, dt, pa, wh1, lh, device, ww1, wtyp, ww2, ca, dd, dld, dh, pd, ph, dn, rn, sb, qh, pv, etp2):
    #定数を定義する
    pi = math.pi #円周率
    g= 9.8 #重力加速度
    sqrt_2g = math.sqrt(2 * g)#重力加速度的なやつ
    
    # 水田からの流出計算
    qp1 = [0] * (pnum + 1)
    qp2 = [0] * (pnum + 1)
    qp3 = [0] * (pnum + 1)
    total_qp = [0] * (pnum + 1)

    for i in range(1, pnum + 1):
        qp1[i] = 0
        qp2[i] = 0
        qp3[i] = 0
        total_qp[i] = 0

        if etp2[i] == 0:
            if r_in[ir][rn[i]] == 0:
                etp2[i] = 0
            else:
                etp2[i] = etp[i]

        qh[i][1] = qh[i][0] + r_in[ir][rn[i]] * dt - (etp2[i] / 86400 / 1000) * dt
        pv[i][1] = qh[i][1] * pa[i]

        if qh[i][1] <= wh1[i]:
            qp = 0
        elif qh[i][1] >= lh[i]:
            qp = (qh[i][1] - lh[i]) * pa[i]
        else:
            if device[i] == 0:#田んぼダムなしの場合
                qp1[i] = 0.6 * (2 / 3) * sqrt_2g * ww1[i] * ((qh[i][1] - wh1[i]) ** (3 / 2))
                qp2[i] = 100000
            elif device[i] == 1:#一体型の場合
                if qh[i][1] <= (wh1[i] + ww2[i]):
                    if wtyp[i] == 1:
                        qp1[i] = 0.6 * (2 / 3) * sqrt_2g * ww2[i] * ((qh[i][1] - wh1[i]) ** (3 / 2))
                    else:
                        qp1[i] = 0.6 * (8 / 15) * math.tan((ca[i] * pi) / 180) * sqrt_2g * ((qh[i][1] - wh1[i]) ** (5 / 2))
                else:
                    if wtyp[i] == 1:
                        qp1[i] = 0.6 * (2 / 3) * sqrt_2g * ww2[i] * (ww2[i] ** (3 / 2)) + 0.6 * (2 / 3) * sqrt_2g * ww1[i] * ((qh[i][1] - (wh1[i] + ww2[i])) ** (3 / 2))
                    else:
                        qp1[i] = 0.6 * (8 / 15) * math.tan((ca[i] * pi) / 180) * sqrt_2g * (wh1[i] ** (5 / 2)) + 0.6 * (2 / 3) * sqrt_2g * ww1[i] * ((qh[i][1] - (wh1[i] + ww2[i])) ** (3 / 2))
                    qp2[i] = 100000
            elif device[i] == 2:#分離型の場合
                if qh[i][1] <= (lh[i] - dld[i]):
                    qp1[i] = 0.6 * (2 / 3) * sqrt_2g * ww1[i] * ((qh[i][1] - wh1[i]) ** (3 / 2))
                    qp2[i] = 0.6 * (((dd[i] / 2) ** 2) * pi) * ((2 * g * (qh[i][1] + dh[i])) ** 0.5)
                else:
                    qp1[i] = 0.6 * (((dd[i] / 2) ** 2) * pi) * ((2 * g * (qh[i][1] + dh[i])) ** 0.5) + 0.6 * (2 / 3) * sqrt_2g * ww1[i] * ((qh[i][1] - (lh[i] - dld[i])) ** (3 / 2))
                    qp2[i] = 100000

            qp3[i] = 0.6 * (((pd[i] / 2) ** 2) * pi) * (2 * g * (qh[i][1] + ph[i])) ** (1 / 2)

            if qp1[i] >= qp2[i]:
                qp = qp2[i]
            else:
                qp = qp1[i]

            if qp >= qp3[i]:
                qp = qp3[i]

            total_qp[i] = (qp * dt) * dn[i]

            pv[i][0] = pv[i][1] - total_qp[i]
            qh[i][0] = pv[i][0] / pa[i]

            if pv[i][0] <= 0:
                pv[i][0] = 0

            if qh[i][0] <= 0:
                qh[i][0] = 0

    return total_qp, qh, pv, etp2





def input_c(input_c_csv):
    # Read input data from the Excel file
    df = pd.read_csv(input_c_csv)

    # Initialize variables
    runtime = int(df.iloc[3, 0])
    tout = int(df.iloc[3, 1])
    dt = int(df.iloc[3, 2])
    rdn = int(df.iloc[3, 3])

    pnum = 0
    for i in range(100):
        if pd.isna(df.iloc[3 + i, 4]):
            break
        pnum += 1

    # Initialize dictionaries
    pn = {}
    lb = {}
    sb = {}
    pa = {}
    lh = {}
    dn = {}
    ih = {}
    etp = {}
    rn = {}
    Qh = {}
    PV = {}
    c = {}
    ww1 = {}
    wh1 = {}
    pd = {}
    ph = {}
    device = {}
    wtyp = {}
    ww2 = {}
    wh2 = {}
    ca = {}
    dd = {}
    dh = {}
    dld = {}

    for i in range(1, pnum + 1):
        pn[i] = df.iloc[3 + i, 4]
        lb[i] = df.iloc[3 + i, 5]
        sb[i] = df.iloc[3 + i, 6]
        pa[i] = df.iloc[3 + i, 7]
        lh[i] = df.iloc[3 + i, 8]
        dn[i] = df.iloc[3 + i, 9]
        ih[i] = df.iloc[3 + i, 10]
        etp[i] = df.iloc[3 + i, 11]
        rn[i] = df.iloc[3 + i, 12]

        Qh[i, 1] = ih[i]
        PV[i, 1] = pa[i] * Qh[i, 1]
        c[i] = 0

        ww1[i] = df.iloc[3 + i, 14]
        wh1[i] = df.iloc[3 + i, 15]
        pd[i] = df.iloc[3 + i, 16]
        ph[i] = df.iloc[3 + i, 17]

        device[i] = df.iloc[3 + i, 19]
        wtyp[i] = df.iloc[3 + i, 20]
        ww2[i] = df.iloc[3 + i, 21]
        wh2[i] = df.iloc[3 + i, 22]
        ca[i] = df.iloc[3 + i, 23]

        dd[i] = df.iloc[3 + i, 25]
        dh[i] = df.iloc[3 + i, 26]
        dld[i] = df.iloc[3 + i, 27]

    return runtime, tout, dt, rdn, pn, lb, sb, pa, lh, dn, ih, etp, rn, Qh, PV, c, ww1, wh1, pd, ph, device, wtyp,


def precipitation(runtime, rdn, precipitation_csv):
    """
    Precipitation function: Reads precipitation data from an Excel file and
    stores it in two matrices: rain and r_in.
    """
    rain = {}
    r_in = {}

    # Read precipitation data from the Excel file設定必要あり
    df = pd.read_csv(precipitation_csv)

    s = 0
    while True:
        s += 1
        ss = s * 2

        for ir in range(1, runtime + 1):
            rain[(ir, s)] = df.iloc[2 + ir, ss]
            r_in[(ir, s)] = rain[(ir, s)] / 3600 / 1000

        if s > rdn:
            break

    return rain, r_in


def main(input_c, precipitation, paddy_flow, dt, tout, runtime):
    # シート初期化 (Pythonではシートを扱わないため、この部分は削除)

    # 計算諸元読み込み
    pnum, ir, r_in, etp, dt, pa, wh1, lh, device, ww1, wtyp, ww2, ca, dd, dld, dh, pd, ph, dn, rn, sb, qh, pv, etp2 = input_c()

    # 降雨読み込み
    rain = precipitation()

    # tcal = 累積時間
    t = 0
    ir = 1
    iii = 0
    jj = 0
    tcal = 0

    # 計算結果の出力用データフレームの初期化
    qp_results = pd.DataFrame()
    qh_results = pd.DataFrame()

    # 計算開始
    while True:
        t += dt
        jj += 1

        # 水田流出の計算
        qp, qh, pv, etp2 = paddy_flow(pnum, ir, r_in, etp, dt, pa, wh1, lh, device, ww1, wtyp, ww2, ca, dd, dld, dh, pd, ph, dn, rn, sb, qh, pv, etp2)

        # 計算結果の出力
        if t / (3600 * tout) == int(t / (3600 * tout)):
            for ii in range(1, pnum + 1):
                qp_results.loc[ir, ii] = qp[ii]
                qh_results.loc[ir, ii] = qh[ii][1]

        if t / 3600 == int(t / 3600):
            ir += 1

        stoptime = int(t / 3600)

        if stoptime > runtime:
            break

    print("計算が終わりました")

    return qp_results, qh_results
