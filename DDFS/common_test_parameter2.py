from DDFS.element import Environment, Particle, Emitter, Detector, SiLayer, Result
import pandas as pd
import math


def get_normal_dec_par_envir_set():
    dec = Detector()
    envir = Environment()
    emit = Emitter()

    par = Particle(charge=-1, mass=0.106)

    emit_para1 = {
        "type": "steps",
        "maxvalue": 10,
        "minvalue": 15,
        "steps": 2,
        "count": 0
    }

    emit_para2 = {
        "type": "even",
        "maxvalue": 10e-3,
        "minvalue": 10e-3
    }

    emit_para3 = {
        "type": "even",
        "maxvalue": 0.5,
        "minvalue": 0.5
    }

    emit_mode = {
        "p": emit_para1,
        "theta": emit_para2,
        "phi": emit_para3
    }

    emit.add_particle(par, 1, emit_mode)
    print(emit)
    # emit.add_particle(par1, 1/3, "normal")
    # emit.add_particle(par2, 1/3, "normal")

    radius_list = [10, 20, 120, 170, 220, 270, 320, 370, 420, 470, 520, 570, 620, 670, 720, 770, 820, 870, 920, 970, 1020]

    budget_list = [0.0015, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]

    loc0 = [9900, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # loc0 = [1]
    # for i in range(20):
    #     loc0.append(4)

    loc1 = [9900, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # loc1 = [1]
    #
    # for i in range(20):
    #     loc1.append(4)

    effi = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # radius_list = [i * 1000. for i in radius_list]
    loc0 = [i * 1e-1 for i in loc0]
    loc1 = [i * 1e-1 for i in loc1]
    # budget_list = [i * 1e-3 for i in budget_list]

    for i in range(0, len(radius_list)):
        dec.add_layer(
            SiLayer(material_budget=budget_list[i], radius=radius_list[i], efficiency=effi[i],
                    loc0=loc0[i], loc1=loc1[i]))

    # print(dec)

    envir.update_environment("B", 3)
    envir.update_environment("multiple_scattering", True)
    # envir.update_environment("multiple_scattering", False)

    return dec, emit, envir


def load_design(file_path):
    # Read the CSV file into a DataFrame


    if file_path:
        print("选择的文件路径:", file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(e)
        return -1

    # Display the DataFrame
    print(df)

    radius_list = df['Radius'].tolist()
    budget_list = df['Budget'].tolist()
    half_z_list = df['Half_z'].tolist()
    loc0 = df['Location 0'].tolist()
    loc1 = df['Location 1'].tolist()
    effi = df['Efficiency'].tolist()

    dec = Detector()


    for i in range(0, len(radius_list)):
        dec.add_layer(
            SiLayer(material_budget=budget_list[i], radius=radius_list[i], efficiency=effi[i],
                    loc0=loc0[i], loc1=loc1[i], half_z=half_z_list[i]))


    return dec




