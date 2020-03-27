import numpy as np


def rotate_jet(jet, phi):
    jet_copy = jet.copy()
    jet[:,1] = jet_copy[:,1] * np.cos(phi) - jet_copy[:,2] * np.sin(phi)
    jet[:,2] = jet_copy[:,1] * np.sin(phi) + jet_copy[:,2] * np.cos(phi)


def mirror_jet_eta(jet):
    jet[:, 2] = jet[:, 2] * (-1.0)


def mirror_jet_phi(jet):
    jet[:, 1] = jet[:, 1] * (-1.0)


def rotate_jet_to_pos_phi(jet):
    sum_pt_pos_phi = np.sum(jet[jet[:, 1] >= 0.0][:, 0])
    sum_pt_neg_phi = np.sum(jet[jet[:, 1] < 0.0][:, 0])
    if sum_pt_pos_phi < sum_pt_neg_phi:
        rotate_jet(jet, np.pi)


def mirror_jet_to_pos_eta(jet):
    sum_pt_pos_eta = np.sum(jet[jet[:, 2] >= 0.0][:, 0])
    sum_pt_neg_eta = np.sum(jet[jet[:, 2] < 0.0][:, 0])
    if sum_pt_pos_eta < sum_pt_neg_eta:
        mirror_jet_eta(jet)


def align_jet_pc_to_pos_phi(jet, ptweight=True, mirror_pos_eta=True):

    #one-constituent jet
    if np.size(jet, 0) == 1:
        return False

    if ptweight is True:
        covX = np.cov(jet[:,1], jet[:,2], aweights=jet[:,0])
    else:
        covX = np.cov(jet[:,1], jet[:,2])

    w, v = np.linalg.eig(covX)
    idxmax = w.argmax()
    x = v[0,idxmax]
    y = v[1,idxmax]
    phi = np.arctan(y/x)
    if x < 0.0 and y > 0.0:
        phi = np.pi - abs(phi)
    elif x < 0.0 and y < 0.0:
        phi = np.pi + abs(phi)
    elif x > 0.0 and y < 0.0:
        phi = 2.*np.pi - abs(phi)

    rotate_jet(jet, 2.*np.pi - phi + np.pi/2.)
    #ensure more scalar pt at positive phi
    rotate_jet_to_pos_phi(jet)
    #can eliminate left-right mirror symmetry
    #ensure more scalar pt at positive eta
    #that's a trickier part
    if mirror_pos_eta is True:
        mirror_jet_to_pos_eta(jet)
    return phi