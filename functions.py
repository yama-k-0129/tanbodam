import pandas as pd
import numpy as np
import math
import matplotlib as plt

pmax = 100
cmax = 240
rmax = 500
Pi = 3.14159265358979

data_c = 'inputc.csv'
data_p = 'precipitation.csv'

def input_c(data_c):
    with open(data_c, "rb") as file:
        content = file.read()
        decoded_content = content.decode("shift_jis", errors="replace")
    from io import StringIO
    data = pd.read_csv(StringIO(decoded_content))
    df = pd.DataFrame(data)
    runtime = int(df.iloc[1, 0])
    tout = int(df.iloc[1, 1])
    dt = int(df.iloc[1, 2])
    rdn = int(df.iloc[1, 3])

    pnum = 0
    pnum = df.shape[0]
    pnum = pnum-2

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
    p_data = {}
    ph = {}
    device = {}
    wtyp = {}
    ww2 = {}
    wh2 = {}
    ca = {}
    dd = {}
    dh = {}
    dld = {}

    for i in range(pnum+1):
        pn[i] = df.iloc[1 + i, 4] #水田番号
        lb[i] = df.iloc[1 + i, 5] #長辺長
        sb[i] = df.iloc[1 + i, 6] #短辺長
        pa[i] = float(df.iloc[1 + i, 7]) #menseki
        lh[i] = float(df.iloc[1 + i, 8]) #keihanndaka
        dn[i] = float(df.iloc[1 + i, 9]) #rakusuikou
        ih[i] = float(df.iloc[1 + i, 10]) #syokisuishin
        etp[i] = float(df.iloc[1 + i, 11]) #gensuishin
        rn[i] = df.iloc[1 + i, 12]#入力降雨NO

        Qh[i] = float(ih[i]) #suidensuishin
        PV[i] = float(pa[i]) * Qh[i] #suidentyoryuuryou
        c[i] = 0 #kousuikaishikaunta-

############落水桝諸元###############################################
        ww1[i] = float(df.iloc[1 + i, 14]) #落水口幅
        wh1[i] = float(df.iloc[1 + i, 15]) #セキ板高
        p_data[i] = float(df.iloc[1 + i, 16]) #排水溝パイプ直径
        ph[i] = float(df.iloc[1 + i, 17]) #田面からパイプ中心までの高さ

        device[i] = int(df.iloc[1 + i, 19])
        wtyp[i] = int(df.iloc[1 + i, 20])
        ww2[i] = float(df.iloc[1 + i, 21])
        wh2[i] = float(df.iloc[1 + i, 22])
        ca[i] = int(df.iloc[1 + i, 23])

        dd[i] = float(df.iloc[1 + i, 25])
        dh[i] = float(df.iloc[1 + i, 26])
        dld[i] = float(df.iloc[1 + i, 27])

    return runtime, tout, dt, rdn, pn, lb, sb, pa, lh, dn, ih, etp, rn, Qh, PV, c, ww1, wh1, p_data, ph, device, wtyp, ww2, wh2, ca, dd, dh, dld, pnum

runtime, tout, dt, rdn, pn, lb, sb, pa, lh, dn, ih, etp, rn, Qh, PV, c, ww1, wh1, p_data, ph, device, wtyp, ww2, wh2, ca, dd, dh, dld, pnum = input_c(data_c)

def precipitation(data_p, rdn,runtime):
    with open(data_p, "rb") as file:
        content = file.read()
        decoded_content = content.decode("shift_jis", errors="replace")
    from io import StringIO
    data = pd.read_csv(StringIO(decoded_content))
    df = pd.DataFrame(data)
    columns_to_drop = df.columns[2::2]
    df = df.drop(columns_to_drop, axis=1)
    rain = df.iloc[2:].reset_index(drop=True)#雨のデータの中で使う部分を抽出r_inも同様に行う
    rain = rain.set_index(rain.columns[0])
    r_in = rain
    r_in = r_in.applymap(lambda x: float(x) / 3600 / 1000)
    #################この部分はpythonには不要だった###########################
    # # for i in range(rdn):
    #     s = i+1
    #     for j in range(runtime):
    #         rain[j][s - 1] = df.iloc[3 + j, s]  # インデックス指定を整数に修正
    #         r_in[j][s - 1] = int(rain[j][s - 1]) / 3600 / 1000  # インデックス指定を整数に修正
    ######################################################
    return rain, r_in

rain, r_in= precipitation(data_p, rdn, runtime)
rain = rain.astype(float)
r_in = r_in.astype(float)
r_in

def paddy_flow(dt, pnum, r_in, dn, rn, device, wtyp, c, pa, lh, etp, ww1, wh1, wh2, p_data, ph, ww2, dd, dh, dld, ca, ir):
    Qp1 = [0.0] * pnum
    Qp2 = [0.0] * pnum
    Qp3 = [0.0] * pnum
    TotalQp = [0.0] * pnum
    etp2 = [0.0] * pnum
    Pi = math.pi
    Qh_2 = {}
    PV_2 = {}
    Qp = [0.0] * pnum
    TotalQp = [0.0] * pnum
    rn_int = 0

    for i in range(pnum):
        Qp1[i], Qp2[i], Qp3[i] = 0, 0, 0
        TotalQp[i] = 0
        rn_int = int(rn[i])-1

        if c[i] == 0:
            if r_in.iloc[ir, rn_int] == 0:
                etp2[i] = 0
            if r_in.iloc[ir, rn_int] > 0:
                c[i] += 1
                etp2[i] = etp[i]
                  
        Qh_2[i] = Qh[i] + r_in.iloc[ir, rn_int] * dt - (etp2[i] / 86400 / 1000) * dt
        PV_2[i] = Qh_2[i] * pa[i]
        wh1[i] = float(wh1[i])
        lh[i] = float(lh[i])

        if Qh_2[i] <= wh1[i]: #水深がセキ板よりも低い場合は流出0
            Qp[i] = 0
            continue
        elif Qh_2[i] >= lh[i]: #水深が畦畔高より高い場合、水田畦畔から流出
            Qp[i] = (Qh_2[i] - lh[i]) * pa[i]
            continue
        else:
            if device[i] == 0:#田んぼダムなし
                Qp1[i] = 0.6 * (2 / 3) * ((2 * 9.8) ** 0.5) * ww1[i] * ((Qh_2[i] - wh1[i]) ** (3 / 2))
                Qp2[i] = 100000

            elif device[i] == 1:#一体型
                if Qh_2[i] <= (wh1[i] + wh2[i]):#田面水深<器具の高さ
                    if wtyp[i] == 1:
                        Qp1[i] = 0.6 * (2 / 3) * ((2 * 9.8) ** 0.5) * ww2[i] * ((Qh_2[i] - wh1[i]) ** (3 / 2))
                    if wtyp[i] == 2:
                        Qp1[i] = 0.6 * (8 / 15) * np.tan((ca[i] * Pi) / 180) * ((2 * 9.8) ** 0.5) * ((Qh_2[i] - wh1[i]) ** (5 / 2))

                elif Qh_2[i] > (wh1[i] + wh2[i]):#田面水深が器具より高いとき
                    if wtyp[i] == 1:
                        Qp1[i] = 0.6 * (2 / 3) * ((2 * 9.8) ** 0.5)* ww2[i] * (wh2[i] ** (3 / 2)) + 0.6 * (2 / 3) * ((2 * 9.8) ** 0.5) * ww1[i] * ((Qh_2[i] - (wh1[i] + wh2[i])) ** (3 / 2))
                    if wtyp[i] == 2:
                        Qp1[i] = 0.6 * (8 / 15) * np.tan((ca[i] * Pi) / 180) * ((2 * 9.8) ** 0.5) * (wh1[i] ** (5 / 2)) + 0.6 * (2 / 3) * ((2 * 9.8) ** 0.5) * ww1[i] * ((Qh_2[i] - (wh1[i] + wh2[i])) ** (3 / 2))
                    Qp2[i] = 100000
            
            elif device[i] == 2:#分離型
                if Qh_2[i] <= (lh[i] - dld[i]):
                    Qp1[i] = 0.6 * (2 / 3) * ((2 * 9.8) ** 0.5) * ww1[i] * ((Qh_2[i] - wh1[i]) ** (3 / 2))
                    Qp2[i] = 0.6 * (((dd[i]/2)**2)*Pi)*((2 * 9.8 * (Qh_2[i] + dh[i]))**0.5)
                    
                elif Qh_2[i] >= (lh[i] - dld[i]):
                    Qp1[i] = 0.6 * (((dd[i]/2)**2)*Pi)*((2 * 9.8 * (Qh_2[i] + dh[i]))**0.5) + 0.6 * (2 / 3) * ((2 * 9.8) ** 0.5) * ww1[i] * ((Qh_2[i] - (lh[i] - dld[i])) ** (3 / 2))
                    Qp2[i] = 100000       
            ##流出計算ここまで
            
            #落水桝排水口のオリフィス流出
            Qp3[i] = 0.6 * (((p_data[i]/2) ** 2) * Pi) * (2 * 9.8 * (Qh_2[i] + ph[i])) ** 0.5
            
            #規定要因の判定　最も小さい値を流出量にする
            Qp[i] = min(Qp1[i], Qp2[i], Qp3[i])
        
            #流出計算修了→値の更新
        #総流出量 = Qp(i) * 落水桝の数 
        TotalQp[i] = (Qp[i] * dt) * dn[i]
            
        #水田貯留量と田面水深の更新→次の時間ステップへ引継ぎ
        PV[i] = PV_2[i] - TotalQp[i]
        Qh[i] = PV[i]/pa[i] #貯留量/面積
        
        if PV[i] <= 0:
            PV[i] = 0
            
        if Qh[i] <= 0:
            Qh[i] = 0
    return Qh, Qp


def main(runtime, tout, dt, rn, pnum, rain, r_in, dn, device, wtyp, c, pa, lh, etp, ww1, wh1, wh2, p_data, ph, ww2, dd, dh, dld, ca):

    # Initialize variables
    t = 0
    ir = 0
    df_Qp = pd.DataFrame(columns=range(pnum+1), index=range(runtime+1))
    df_Qh = pd.DataFrame(columns=range(pnum+1), index=range(runtime+1))
    df_rain = pd.DataFrame(columns=range(pnum+1), index=range(runtime+1))
    
    # Calculation loop
    while t <= runtime * 3600:
        t += dt
        Qh, Qp = paddy_flow(dt, pnum, r_in, dn, rn, device, wtyp, c, pa, lh, etp, ww1, wh1, wh2, p_data, ph, ww2, dd, dh, dld, ca, ir)
        # Store results
        if t / (3600 * tout) == int(t / (3600 * tout)):
            tt = int(t / (3600 * tout))
            for ii in range(pnum):
                rn_int2 = int(rn[ii])-1
                df_Qp.iloc[ir, ii] = Qp[ii]
                df_Qh.iloc[ir, ii] = Qh[ii]
                #田面水深
                df_rain.iloc[ir, ii] = rain.iloc[tt-1, rn_int2]
        if t / 3600 == int(t / 3600):
            ir += 1
        if ir == 120:
            break
    return df_Qh, df_Qp, df_rain

df_Qh, df_Qp, df_rain = main(runtime, tout, dt, rn, pnum, rain, r_in, dn, device, wtyp, c, pa, lh, etp, ww1, wh1, wh2, p_data, ph, ww2, dd, dh, dld, ca)




df_Qp= df_Qp.astype(float)
df_rain= df_rain.astype(float)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

##############################################グラフ描画(流出量)########################################
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax2.set_ylabel('Rainfall')

# Loop through the paddy numbers (1 to 4)
for paddy_number in range(1, 5):
    # Initialize empty lists to store values for each time step
    outflow_values = []
    rainfall_values = []

    # Loop through each row
    for i in range(len(df_Qp)):
        # Append the outflow and rainfall values for this time step
        outflow_values.append(df_Qp[paddy_number][i])
        rainfall_values.append(df_rain[paddy_number][i])

    # Plot the outflow (Qp) on the primary y-axis
    ax1.plot(df_Qp.index, outflow_values, label=f"Outflow (Paddy {paddy_number})")
    # Plot the rainfall on the secondary y-axis (inverted)
    ax2.plot(df_rain.index, [-x for x in rainfall_values], label=f"Rainfall (Paddy {paddy_number})", linestyle="--")

ax1.set_xlabel('Time Step')
ax1.set_ylabel('Outflow')
ax1.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()

##############################################グラフ描画(田面水深)########################################
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax2.set_ylabel('Rainfall')

# Loop through the paddy numbers (1 to 4)
for paddy_number in range(1, 5):
    # Initialize empty lists to store values for each time step
    outflow_values = []
    rainfall_values = []

    # Loop through each row
    for i in range(len(df_Qh)):
        # Append the outflow and rainfall values for this time step
        outflow_values.append(df_Qh[paddy_number][i])
        rainfall_values.append(df_rain[paddy_number][i])

    # Plot the outflow (Qp) on the primary y-axis
    ax1.plot(df_Qh.index, outflow_values, label=f"Outflow (Paddy {paddy_number})")
    # Plot the rainfall on the secondary y-axis (inverted)
    ax2.plot(df_rain.index, [-x for x in rainfall_values], label=f"Rainfall (Paddy {paddy_number})", linestyle="--")

ax1.set_xlabel('Time Step')
ax1.set_ylabel('Outflow')
ax1.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
