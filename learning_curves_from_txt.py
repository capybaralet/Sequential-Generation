
import numpy as np

# not general!
def lcs(fpath='/data/lisatmp4/kruegerd/RAM_TEST_RESULTS/VID_SCG_dk_2objects_with_noise_results.txt'):
    ff = open(fpath)
    lines = ff.readlines()
    cost_lines = [lines[3 + 83*x] for x in range(int((len(lines) -3) / 83))]
    for cl in cost_lines:
        print cl
    costs = [float(cl[16:23]) for cl in cost_lines]
    return costs
