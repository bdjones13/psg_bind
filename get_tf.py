import dionysus
from scipy.spatial.distance import squareform
import diode
import gc

def get_tf(d, is_vr, dim=None, cutoff=None):
    # See https://mrzv.org/software/dionysus2/tutorial/rips.html for example and methodology
    # convert distance matrix to a 'condensed distance matrix'
    # then pass to Dionysus to form a VR filtration
    if is_vr:
        sq_dist = squareform(d)
        f = dionysus.fill_rips(sq_dist, dim, cutoff)
    else:  # Alpha
        alpha_complex = diode.fill_alpha_shapes(d)
        f = dionysus.Filtration(alpha_complex)
    p = dionysus.homology_persistence(f)
    dgms = dionysus.init_diagrams(p, f)
    del p
    del d
    gc.collect()
    # dionysus.plot.plot_bars(dgms[0], show=True)
    return dgms[0:3]
