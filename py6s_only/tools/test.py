import numpy as np
import datetime
import h5py
import glob
import os
import logging
import pickle
from logger_config import setup_logger
import read_landsat8
import get_sensor_info
import rayleigh
import get_f0
import get_anc
import read_mask_file
import gas_trans
import whitecaps
import predefine
import getglint
import get_rhown_nir
import atmocor2
import get_chl
import nlw_outband
import brdf as brdfmodel
import pandas as pd
import concurrent.futures
from get_nc_height import DEMManager

def process_group(args):
    sensor_name = predefine.parameters().sensor_name
    date, time, group, Fonom, bands, f0, aw, bbw, Tau_r, refl_scale, refl_offset, nwvis = args
    num_443 = np.argmin(np.abs(bands - 443))
    num_490 = np.argmin(np.abs(bands - 490))
    num_520 = np.argmin(np.abs(bands - 520))
    num_555 = np.argmin(np.abs(bands - 555))
    num_670 = np.argmin(np.abs(bands - 670))
    print(f"处理日期：{date} 时间：{time}，共 {len(group)} 行数据")
    b1_row = group['B1'].values.reshape(1, -1)
    id_row = group['id'].values.reshape(1, -1)
    Lt = np.full(shape=(len(bands), b1_row.shape[0], b1_row.shape[1]), fill_value=np.nan)
    bandlist = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    for i, band in enumerate(bandlist):
        Lt[i, :, :] = group[band].values.reshape(1, -1)
    SolarAzimuth = group['SAA'].values.reshape(1, -1) * 0.01
    SolarZenith = group['SZA'].values.reshape(1, -1) * 0.01
    SensorAzimuth = group['VAA'].values.reshape(1, -1) * 0.01
    SensorZenith = group['VZA'].values.reshape(1, -1) * 0.01
    Latitude = group['lat'].values.reshape(1, -1)
    Longitude = group['lon'].values.reshape(1, -1)

    year = int(date.split('-')[0])
    month = int(date.split('-')[1])
    day = int(date.split('-')[2])
    hour = int(time.split(':')[0])
    minute = int(time.split(':')[1])
    second = float(time.split(':')[2].split('Z')[0])
    msec = int(1000 * (second + 60 * (minute + 60 * hour)))
    date = datetime.datetime(year, month, day)
    doy = date.timetuple().tm_yday
    d = rayleigh.esdist(year, doy, msec)
    F1 = (1. / d) ** 2
    FoBAR = f0 * F1

    Lt = np.array(Lt).astype(float)

    # Apply vicarious calibration
    if sensor_name == 'oli':
        vicarious_gain = [1.011100, 1.010100, 1.007000, 1.009800, 1.000000, 1.000000, 1.000000]
        vicarious_offset = [0, 0, 0, 0, 0, 0, 0]
        for i, band in enumerate(bands):
            Lt[i, :, :] = ((Lt[i, :, :] * refl_scale[i] + refl_offset[i]) * FoBAR[i] / np.pi) * vicarious_gain[i] + \
                          vicarious_offset[i]
            # Lt[i, :, :] = (Lt[i, :, :] * refl_scale[i] + refl_offset[i]) / 10.0  * vicarious_gain[i] + \
            #               vicarious_offset[i]

    elif sensor_name in ['l7etmp', 'l5tm']:
        # 其中refl_scale 和 refl_offset 实际上为radiance_scale radiance_offset
        for i, band in enumerate(bands):
            Lt[i, :, :] = (Lt[i, :, :] * refl_scale[i] + refl_offset[i]) / 10.0
    else:
        print("this sensor is not support!")
        exit(1)

    # Starting to read anc data
    ws = get_anc.get_windspeed("common/", Longitude, Latitude, month)
    dem = DEMManager("common/gebco_ocssw_v2020.nc")
    height = dem.get_elevation_array(Latitude, Longitude)
    pr = get_anc.get_pressure("common/", Longitude, Latitude, month)
    pr = pr * np.exp(-1 * height / 8434)
    rh = get_anc.get_relative_humidity("common/", Longitude, Latitude, month)
    water_mask = np.full_like(Longitude, fill_value=True)

    # Starting to get rayleigh radiance
    rayleigh_lut_path = f"share/{sensor_name}/l8/rayleigh/"
    # 计算相对方位角
    reaa = SensorAzimuth - 180.0 - SolarAzimuth
    reaa[reaa < -180] = reaa[reaa < -180] + 360.0
    reaa[reaa > 180] = reaa[reaa > 180] - 360.0
    sigma = 0.0731 * np.sqrt(ws)
    Lr, L_q, L_u = rayleigh.get_rayleigh(rayleigh_lut_path, SolarZenith, SensorZenith, reaa, sigma, pr, FoBAR,
                                         bands)
    Lr = np.array(Lr)

    # Starting to calculate gas transmittance
    ozone = get_anc.get_ozone("common/", Longitude, Latitude, doy)
    no2_frac = get_anc.get_no2_frac("common/", Longitude, Latitude)
    no2_strat, no2_tropo = get_anc.get_no2_climatology("common/", Longitude, Latitude, month)
    wv = get_anc.get_water_vapor("common/", Longitude, Latitude, month)
    tg_sol, tg_sen = gas_trans.calculate(oz=ozone, no2_tropo=no2_tropo, no2_frac=no2_frac,
                                         no2_strat=no2_strat, wv=wv, solz=SolarZenith, senz=SensorZenith)

    # Starting to calculate white caps radiance
    p0 = 1013.25
    mu0 = np.cos(np.deg2rad(SolarZenith))
    mu = np.cos(np.deg2rad(SensorZenith))
    airmass = 1.0 / mu0 + 1 / mu
    t_sol = ()
    t_sen = ()
    for ib, tau_r in enumerate(Tau_r):
        t_sol_ = np.exp(-0.5 * pr / p0 * tau_r / mu0)
        t_sen_ = np.exp(-0.5 * pr / p0 * tau_r / mu)
        t_sol = t_sol + (t_sol_,)
        t_sen = t_sen + (t_sen_,)
    t_sol = np.array(t_sol)
    t_sen = np.array(t_sen)
    rho_f = whitecaps.whitecap_reflectance(ws, bands)
    tLf = ()
    for ib, wave in enumerate(bands):
        tLf_ = rho_f[ib, :, :] * t_sen[ib, :, :] * t_sol[ib, :, :] * FoBAR[ib] * mu0 / np.pi
        tLf = tLf + (tLf_,)
    tLf = np.array(tLf)

    # Starting to calculate glint_coef
    zero = np.zeros_like(ws)
    glint_coef, glint_coef_q, glint_coef_u = getglint.getglint_iqu(X1=SensorZenith, X2=SolarZenith, X3=reaa, X4=ws,
                                                                   X5=zero)
    GLINT_MIN = 0.0001

    Taur = np.array([pr / p0 * i for i in Tau_r])
    Ltemp = np.copy(Lt)

    # modisa don't neet to smile correction
    # correct for gas absorption
    Ltemp = Ltemp / tg_sol / tg_sen
    # modisa don't neet to do cirrus correction
    # Ltemp = Ltemp / pol
    # remove whitecap radiance
    Ltemp = Ltemp - tLf
    # subtract the Rayleigh contribution for this geometry
    Ltemp = Ltemp - Lr
    Ltemp = np.where(water_mask[None, :, :], Ltemp, np.nan)

    if 1:
        last_tLw_nir = np.zeros_like(Ltemp)
        tLw_nir = np.zeros_like(Ltemp)
        Rrs = np.zeros_like(Ltemp)
        Rrs[num_555, :, :] = predefine.parameters().seed_green
        Rrs[num_670, :, :] = predefine.parameters().seed_red

    brdf = np.ones_like(SolarZenith)
    last_iter = np.zeros_like(SolarZenith)  # 判断是否迭代完成 1为完成迭代
    last_iter = np.where(water_mask, last_iter, 1.0)  # 若该点位不为水体则直接结束迭代
    iter_num = 0
    iterx = np.zeros_like(SolarZenith)  # 记录每个像素的迭代次数
    last_refl_nir = np.full(shape=SolarZenith.shape, fill_value=100.)
    iter_reset = np.zeros_like(SolarZenith)
    iter_max = np.full_like(SolarZenith, fill_value=10)  # 定义最大迭代次数
    tLw_final = np.full_like(Ltemp, fill_value=np.nan)
    nLw_final = np.full_like(Ltemp, fill_value=np.nan)
    TLg = np.full_like(Ltemp, fill_value=0)

    cslp = 1.0 / (predefine.parameters().ctop - predefine.parameters().cbot)
    cint = -cslp * predefine.parameters().cbot
    glint_min = 0.0001
    chl = np.full(shape=Ltemp[0, :, :].shape, fill_value=predefine.parameters().seed_chl)

    taua = 0
    # 大气校正算法使用波段
    aer_s = predefine.parameters().aer_s
    aer_l = predefine.parameters().aer_l
    # 只要last_iter存在0值则继续迭代
    while last_iter.min() == 0:
        iter_num += 1
        iterx = iterx + 1
        status = 0

        # Initialize tLw as surface + aerosol radiance
        tLw = Ltemp * 1.
        # Compute and subtract glint radiance
        if iter_num <= 2:
            TLg = getglint.glint_rad(glint_coef=glint_coef, sza=SolarZenith, vza=SensorZenith, iter_num=iterx,
                                     La=tLw,
                                     Fo=FoBAR, taur=Taur, taua=taua, nir_s=aer_s, nir_l=aer_l)
        glint_loc = glint_coef > GLINT_MIN
        for ib in range(bands.size):
            tLw[ib, :, :][glint_loc] = tLw[ib, :, :][glint_loc] - TLg[ib, :, :][glint_loc]

        want_nirLw = 1
        # Adjust for non-zero NIR water-leaving radiances using IOP model
        if want_nirLw == 1:
            rhown_nir = get_rhown_nir.get_rhown_eval(fqfile=None, wave=bands, Rrs=Rrs, nir_s=aer_s, nir_l=aer_l,
                                                     aw=aw,
                                                     bbw=bbw, chl=chl, solz=SolarZenith, senz=SensorZenith,
                                                     phi=reaa)
            for ib in [aer_s, aer_l]:
                # Convert NIR reflectance to TOA W-L radiance
                tLw_nir[ib, :, :] = rhown_nir[ib, :, :] / np.pi * f0[ib] * mu0 * t_sol[ib] * t_sen[ib] / brdf
                # Iteration damping
                tLw_nir[ib, :, :] = (1.0 - predefine.parameters().df) * tLw_nir[ib, :,
                                                                        :] + predefine.parameters().df * last_tLw_nir[
                                                                                                         ib, :, :]

                # Ramp-up
                tLw_nir[ib, :, :][(chl > 0.0) & (chl < predefine.parameters().cbot)] = 0.0
                loc_tmp = (chl > predefine.parameters().cbot) & (chl < predefine.parameters().ctop)
                tLw_nir[ib, :, :][loc_tmp] = tLw_nir[ib, :, :][loc_tmp] * (cslp * chl + cint)[loc_tmp]

                # Remove estimated NIR water-leaving radiance
                tLw[ib, :, :] = tLw[ib, :, :] - tLw_nir[ib, :, :]
                del loc_tmp
        else:
            tLw_nir = None

        l_nir1 = tLw[aer_s, :, :]
        l_nir2 = tLw[aer_l, :, :]

        aero_out = atmocor2.calculate(bands=bands, l_a_nir1=l_nir1, l_a_nir2=l_nir2, lon=Longitude, lat=Latitude,
                                      F0=FoBAR, sza=SolarZenith, saa=SolarAzimuth, vza=SensorZenith,
                                      vaa=SensorAzimuth, nirs_num=aer_s, nirl_num=aer_l,
                                      aerosol_lut_filepath=f"share/{sensor_name}/l8/aerosol/",
                                      winds_peed=ws, pressure=pr, relative_humidity=rh, wv=wv, taur=Tau_r, month=month)

        if aero_out is None:
            break

        La, t_sensor, t_solar, taua, aer1, aer2 = aero_out

        # print("Ltemp[0]", Ltemp[0])
        # print("La[0], ",La[0])
        # Subtract aerosols and normalize
        tLw = tLw - La
        Lw = tLw / t_sensor * tg_sol
        nLw = Lw / t_solar / tg_sol / mu0 / F1 * brdf

        if want_nirLw == 1:
            refl_nir = Rrs[num_670, :, :] * 1.
            for ib in [aer_s, aer_l]:
                last_tLw_nir[ib, :, :] = tLw_nir[ib, :, :]
            del ib

            for ib in range(nwvis):
                Rrs[ib, :, :] = nLw[ib, :, :] / f0[ib]

            red = num_670
            chl = get_chl.get_default_chl(rrs=Rrs, bands=bands, b443=num_443, b490=num_490, b520=num_520,
                                          b555=num_555,
                                          b670=num_670)

            # if we passed atmospheric correction but the spectral distribution of
            # Rrs is bogus (chl failed), assume this is a turbid-water case and
            # reseed iteration as if all 670 reflectance is from water.
            loc_temp = ((chl == predefine.parameters().chlbad) & (iter_reset == 0) & (iterx < iter_max))
            chl[loc_temp] = 10
            Rrs[red, :, :][loc_temp] = 1.0 * (Ltemp[red, :, :][loc_temp] - TLg[red, :, :][loc_temp]) / \
                                       t_sol[red, :, :][loc_temp] / tg_sol[red, :, :][loc_temp] / mu0[loc_temp] / \
                                       FoBAR[
                                           red]
            iter_reset[loc_temp] = 1
            del loc_temp

            # if we already tried a reset, and still no convergence, force one last
            # pass with an assumption that all red radiance is water component, and
            # force iteration to end.  this will be flagged as atmospheric correction
            # failure, but a qualitatively useful retrieval may still result.
            loc_temp = ((chl == predefine.parameters().chlbad) & (iter_reset == 1) & (iterx < iter_max))
            chl[loc_temp] = 10
            Rrs[red, :, :][loc_temp] = 1.0 * (Ltemp[red, :, :][loc_temp] - TLg[red, :, :][loc_temp]) / \
                                       t_sol[red, :, :][loc_temp] / tg_sol[red, :, :][loc_temp] / mu0[loc_temp] / \
                                       FoBAR[
                                           red]
            iter_reset[loc_temp] = 2
            del loc_temp

        tLw_final_temp = tLw * 1.
        nLw_final_temp = nLw * 1.

        if iter_num > predefine.parameters().aer_iter_max:
            last_iter = np.ones_like(tLw_final[0])
            for ib in range(bands.size):
                tLw_final[ib, :, :] = np.nanmean(np.array([tLw_final[ib, :, :], tLw_final_temp[ib, :, :]]), axis=0)
                nLw_final[ib, :, :] = np.nanmean(np.array([nLw_final[ib, :, :], nLw_final_temp[ib, :, :]]), axis=0)
        else:
            loc_temp = ((np.abs(refl_nir - last_refl_nir) < np.abs(predefine.parameters().nir_chg * refl_nir)) | (
                    refl_nir < 0.0)) | np.isnan(aer1[1])
            last_iter[loc_temp] = 1
            tLw_final_temp = np.where(loc_temp[None, :, :], tLw_final_temp, np.nan)
            nLw_final_temp = np.where(loc_temp[None, :, :], nLw_final_temp, np.nan)

            Ltemp = np.where(loc_temp[None, :, :], np.nan, Ltemp)
            # tLw_final_temp[~loc_temp] = np.nan # don't neet to iteration
            # tLw[loc_temp] = np.nan  # continue iteration
            for ib in range(bands.size):
                tLw_final[ib, :, :] = np.nanmean(np.array([tLw_final[ib, :, :], tLw_final_temp[ib, :, :]]), axis=0)
                nLw_final[ib, :, :] = np.nanmean(np.array([nLw_final[ib, :, :], nLw_final_temp[ib, :, :]]), axis=0)

        last_refl_nir = refl_nir * 1.
        if iter_num > predefine.parameters().aer_iter_max:
            break
    # convert water-leaving radiances from band-averaged to band-centered
    # Switch mean solar irradiance from band-averaged to band centered also.
    tmp_nLw = nLw_final * 1.
    # 波段外Lw校正
    outband_correction = nlw_outband.get_outband_correction(f"share/{sensor_name}/msl12_sensor_info.dat", bands,
                                                            tmp_nLw)
    tmp_nLw *= outband_correction

    # Compute f/Q correction and apply to nLw 只对可见光波段进行校正
    brdf_mod = brdfmodel.BRDF(vza=SensorZenith, sza=SolarZenith, vaa=SensorAzimuth, saa=SolarAzimuth, bands=bands,
                              F0=Fonom, chl=chl, nlw=tmp_nLw, b443=num_443, b490=num_490, b520=num_520,
                              b670=num_670,
                              b555=num_555, foqopt='FOQMOREL', ws=ws, fqfile="common/morel_fq.nc")

    brdf = brdf_mod.ocbrdf()
    for ib in range(nwvis):
        tmp_nLw[ib, :, :] = tmp_nLw[ib, :, :] * brdf[ib, :, :]

    # 除开两个用来做大气校正的波段，其余波段的Rrs需要使用nlwoutband 和 brdf校正后的nlw重新计算
    Rrs = tmp_nLw / Fonom[:, np.newaxis, np.newaxis]
    for ib in range(len(bands)):
        if ib == aer_s | ib == aer_l:
            Rrs[ib, :, :] = nLw_final[ib, :, :] / f0[ib, np.newaxis, np.newaxis]

    # Compute final Rrs
    Rrs = tmp_nLw / f0[:, np.newaxis, np.newaxis]
    group_result = {
        'id': np.squeeze(id_row),
        'lon': np.squeeze(Longitude),
        'lat': np.squeeze(Latitude),
        'DATE_ACQUIRED': date,
        'SCENE_CENTER_TIME': time,
        'Rrs_443': np.squeeze(Rrs[0, :, :]),
        'Rrs_482': np.squeeze(Rrs[1, :, :]),
        'Rrs_561': np.squeeze(Rrs[2, :, :]),
        'Rrs_655': np.squeeze(Rrs[3, :, :]),
        'Rrs_865': np.squeeze(Rrs[4, :, :]),
        'Rrs_1609': np.squeeze(Rrs[5, :, :]),
        'Rrs_2201': np.squeeze(Rrs[6, :, :]),
    }
    return group_result


def main(args):
    csvfile_path = args
    sensor_name = predefine.parameters().sensor_name
    outputfile_path = csvfile_path.replace("/M/global_watercolor_river/", "/public/result_l8_zd/global_watercolor_river/")
    start_time = datetime.datetime.now()
    logger = setup_logger(__name__)
    logger.info('Beginning to Atmospheric correction')
    meta_file = "test_file/LC08_L1TP_022030_20240303_20240314_02_T1/LC08_L1TP_022030_20240303_20240314_02_T1_MTL.txt"
    refl_scale, refl_offset, year, month, doy, msec = read_landsat8.get_info(meta_file)

    logger.info('=====Starting to read sensor info=====')
    f0 = get_sensor_info.read_sensor_info(file_path=f"share/{sensor_name}/msl12_sensor_info.dat", parameter="F0")
    aw = get_sensor_info.read_sensor_info(file_path=f"share/{sensor_name}/msl12_sensor_info.dat", parameter="aw")
    bbw = get_sensor_info.read_sensor_info(file_path=f"share/{sensor_name}/msl12_sensor_info.dat", parameter="bbw")
    Tau_r = get_sensor_info.read_sensor_info(file_path=f"share/{sensor_name}/msl12_sensor_info.dat", parameter="Tau_r")
    bands = get_sensor_info.read_sensor_info(file_path=f"share/{sensor_name}/msl12_sensor_info.dat", parameter="Lambda")
    nwvis = np.searchsorted(bands, predefine.parameters().MAXWAVE_VIS, side='right')
    d = rayleigh.esdist(year, doy, msec)
    F1 = (1. / d) ** 2
    FoBAR = f0 * F1
    Fonom = get_f0.get_fo_list_from_bands(f0file="common/Thuillier_F0.dat", band_list=bands, window=5)

    try:
        csv_data = pd.read_csv(csvfile_path)
    except:
        print("this file is error!")
        return 0
    csv_data['DATE_ACQUIRED'] = csv_data['DATE_ACQUIRED'].ffill()
    csv_data['SCENE_CENTER_TIME'] = csv_data['SCENE_CENTER_TIME'].ffill()
    csv_data['SUN_AZIMUTH'] = csv_data['SUN_AZIMUTH'].ffill()
    csv_data['SUN_ELEVATION'] = csv_data['SUN_ELEVATION'].ffill()
    # 删除B1为空值的行
    csv_data = csv_data.dropna(subset=['B1'])
    csv_data = csv_data.dropna(subset = ['QA_PIXEL'])
    if csv_data.__len__() == 0:
        return 0
    def is_clear(value):
        """
        判断一个 flag 是否为有效（没有云或冰）
        bit 3 = 云（第3位）
        bit 4 = 云阴影（第4位）
        bit 5 = 冰（第5位）
        """
        mask = (1 << 3) | (1 << 4) | (1 << 5)
        return (value & mask) == 0 and (value & (1 << 7)) != 0
    # 通过QA_PIXEL删除异常点
    csv_data['QA_PIXEL'] = csv_data['QA_PIXEL'].astype(int)
    csv_data = csv_data[csv_data['QA_PIXEL'].apply(is_clear)]
    if csv_data.__len__() == 0:
        return 0
    # 分组：以 DATE_ACQUIRED 和 SCENE_CENTER_TIME 作为一个组的标志
    grouped = csv_data.groupby(['DATE_ACQUIRED', 'SCENE_CENTER_TIME'])
    tasks = [(date, time, group, Fonom, bands, f0, aw, bbw, Tau_r, refl_scale, refl_offset, nwvis) for (date, time), group in grouped]
    # 使用线程池并行处理
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=70) as executor:  # max_workers 根据CPU核数调整
        for result in executor.map(process_group, tasks):
            results.append(result)

    df_list = []
    for i in range(len(results)):
        try:
            df_result = pd.DataFrame(results[i])
        except:
            df_result = pd.DataFrame([results[i]])
        df_list.append(df_result)
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(outputfile_path, index=False)
    end_time = datetime.datetime.now()
    print('processing time: ' + str((end_time - start_time).seconds) + ' seconds')

if __name__ == '__main__':
    import os
    import glob

    # 读取某一个文件夹下所有子文件夹的名称
    def get_subfolder_names(folder_path):
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        return subfolders

    subfolder = sorted(get_subfolder_names("/public/global_watercolor_river"))

    # 确定一个输出盘

    for folder in subfolder:
        # 如果文件夹不存在则创建一个文件夹
        os.makedirs("/public/result_l8_zd/global_watercolor_river/" + folder, exist_ok=True)
        # 读取该文件夹中包含L8的文件
        folder_path = os.path.join("/public/global_watercolor_river", folder)
        file_path = os.path.join(folder_path, "*L8*")
        files = glob.glob(file_path)
        for file in files:
            main(file)
    # main("M:/global_watercolor_river/100_200/newGQpoint_133_dnL8_2.csv")
    # main("J:/newGQpoint_19_dnL8_9.csv")
    # main("M:/global_watercolor_river/100_200/newGQpoint_133_dnL8_16.csv")