import numpy as np

from DDFS.element import Environment, Particle, Emitter, Detector, SiLayer, Result
import math
import pandas as pd


def export_design(filename, radius_list, budget_list, half_z_list, loc0, loc1, effi):
    data = {
        'Radius': radius_list,
        'Budget': budget_list,
        'Half_z': half_z_list,
        'Location 0': loc0,
        'Location 1': loc1,
        'Efficiency': effi
    }

    df = pd.DataFrame(data)
    print(1)
    # Export the data to a CSV file
    # Export the data to a CSV file test

    df.to_csv(filename, index=False)


if __name__ == "__main__":
    N= 10
    radius_list = [10]
    for i in np.linspace(20, 1020, N):
        radius_list.append(i)

    budget_list = [0.0015]
    for i in range(N):
        budget_list.append(0.002)



    loc0 = [9900]
    for i in range(N):
        loc0.append(4)

    loc1 = [9900]
    for i in range(N):
        loc1.append(4)

    effi = [0.0]
    for i in range(N):
        effi.append(1.0)

    loc0 = [i * 1e-3 for i in loc0]
    loc1 = [i * 1e-3 for i in loc1]
    # budget_list = [i * 1e3 for i in budget_list]

    half_z_list = [3000 for i in range(N+1)]

    # radius_list = [10.35, 12.35, 14.35, 34.324999999999996, 36.324999999999996, 58.3, 60.3, 65.0, 100, 200.0, 390, 400,
    #                411.0, 429.0, 447.0, 465.0, 483.0, 501.0, 519.0, 537.0, 555.0, 573.0, 591.0, 609.0, 627.0, 645.0,
    #                663.0, 681.0, 699.0, 717.0, 735.0, 753.0, 771.0, 789.0, 807.0, 825.0, 843.0, 861.0, 879.0, 897.0,
    #                915.0, 933.0, 951.0, 969.0, 987.0, 1005.0, 1023.0, 1041.0, 1059.0, 1077.0, 1095.0, 1113.0, 1131.0,
    #                1149.0, 1167.0, 1185.0, 1203.0, 1221.0, 1239.0, 1257.0, 1275.0, 1293.0, 1311.0, 1329.0, 1347.0,
    #                1365.0, 1383.0, 1401.0, 1419.0, 1437.0, 1455.0, 1473.0, 1491.0, 1509.0, 1527.0, 1545.0, 1563.0,
    #                1581.0, 1599.0, 1617.0, 1635.0, 1653.0, 1671.0, 1689.0, 1707.0, 1725.0, 1743.0, 1761.0, 1779.0, 1800,
    #                1810]
    #
    # budget_list = [0.0017179033925197344, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.0015, 0.0065, 0.0065, 0.0065,
    #                0.0011, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445, 0.00013164444444444445,
    #                0.00013164444444444445, 0.00013164444444444445, 0.013, 0.0065]
    #
    # loc0 = [9.9, 0.0028, 0.006, 0.004, 0.004, 0.004, 0.004, 9.9, 0.0072, 0.0072, 0.0072, 9.9, 0.1, 0.1, 0.1, 0.1, 0.1,
    #         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    #         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    #         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    #         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.9, 0.0072]
    #
    # half_z_list = [3000 for i in range(len(budget_list))]
    #
    # loc1 = [9.9, 0.0028, 0.006, 0.004, 0.004, 0.004, 0.004, 9.9, 0.086, 0.086, 0.086, 9.9, 2.83, 2.83, 2.83, 2.83, 2.83,
    #         2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83,
    #         2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83,
    #         2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83,
    #         2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83, 2.83,
    #         9.9, 0.086]
    #
    # effi = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #         1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]

    # loc0 = [i * 1e2 for i in loc0]
    # loc1 = [i * 1e0 for i in loc1]

    print(len(radius_list), len(budget_list), len(half_z_list), len(loc0), len(loc1), len(effi))

    export_design('design_N_' + str(N) + '.csv', radius_list, budget_list, half_z_list, loc0, loc1, effi)
