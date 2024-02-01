import torch
from torch import nn
import torch.nn.functional as F
from pytorch_radon.utils import PI, SQRT2, deg2rad, affine_grid, grid_sample
import skimage.data as d
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from fastatomography.util import sector_mask
import os
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.nn.functional import mse_loss
from utils import ray_transform
from fastatomography.util import plotmosaic
from ccpi.filters.regularisers import ROF_TV, FGP_TV, PD_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
from numpy.fft import fft2, ifft2

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(False)

device = th.device('cuda:0')
# device = th.device('cpu')

img = np.load('shepp_phantom.npy')
# img[:] = 0
# img[28:37, 23:42, 30 - 5:35 - 5] = 1
img = gaussian(img, 1)
target = th.as_tensor(img).unsqueeze(0).unsqueeze(0).to(device)
# img *= sector_mask(img.shape, np.array(img.shape) / 2, img.shape[0] / 2, (0, 360))
# %%
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img.sum(0))
ax[1].imshow(img.sum(1))
ax[2].imshow(img.sum(2))
plt.show()
# %%


n_theta = 50
phi_deg = th.linspace(0, 180, n_theta)
theta_deg = th.linspace(0, 0, n_theta)
psi_deg = th.linspace(0, 0, n_theta)

phi_rad_target = th.deg2rad(phi_deg).to(device)
theta_rad_target = th.deg2rad(theta_deg).to(device)
psi_rad_target = th.deg2rad(psi_deg).to(device)

translation_target = th.randn((2, n_theta), device=device) * 0.1

theta_rad_model = theta_rad_target.clone()
psi_rad_model = psi_rad_target.clone()

sino_target = ray_transform(target, phi_rad_target, theta_rad_target, psi_rad_target, th.zeros_like(translation_target))

# translation_shifts_target[:] = 0
for i in range(translation_target.shape[1]):
    sino_target[i, 0] = th.as_tensor(
        ifft2(fourier_shift(fft2(sino_target[i, 0].cpu().numpy()),
                            translation_target[:, i].cpu().numpy())).real)

plotmosaic(sino_target.squeeze().cpu().numpy(), 'Sino target')
# %%


pars = {'algorithm': FGP_TV, \
        'regularisation_parameter': 7e-2, \
        'number_of_iterations': 50, \
        'tolerance_constant': 1e-06, \
        'methodTV': 0, \
        'nonneg': 1}

# %%

model = th.zeros_like(target).to(device)
phi_rad_model = th.deg2rad(phi_deg).to(device)
theta_rad_model = th.deg2rad(theta_deg).to(device)
psi_rad_model = th.deg2rad(psi_deg).to(device)
# translation_model = th.zeros_like(translation_target).to(device)
translation_model = th.as_tensor(np.random.uniform(-100, 100, translation_target.shape) / 4000, device=device,
                                 dtype=translation_target.dtype)

phi_rad_model += th.as_tensor(np.random.uniform(-3.14, 3.14, phi_rad_target.shape[0]) / 50,
                              device=device)
theta_rad_model += th.as_tensor(np.random.uniform(-3.14, 3.14, phi_rad_target.shape[0]) / 200,
                                device=device)
psi_rad_model += th.as_tensor(np.random.uniform(-3.14, 3.14, phi_rad_target.shape[0]) / 200,
                              device=device)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
axs = ax.ravel
ax[0].scatter(np.arange(len(phi_rad_model)), phi_rad_model.detach().cpu().numpy().squeeze())
ax[0].scatter(np.arange(len(phi_rad_model)), phi_rad_target.detach().cpu().numpy().squeeze(), marker='x')
ax[1].scatter(np.arange(len(phi_rad_model)), theta_rad_model.detach().cpu().numpy().squeeze())
ax[1].scatter(np.arange(len(phi_rad_model)), theta_rad_target.detach().cpu().numpy().squeeze(), marker='x')
ax[2].scatter(np.arange(len(phi_rad_model)), psi_rad_model.detach().cpu().numpy().squeeze())
ax[2].scatter(np.arange(len(phi_rad_model)), psi_rad_target.detach().cpu().numpy().squeeze(), marker='x')
fig.suptitle('Before optimization')
ax[0].set_title('phi')
ax[1].set_title('theta')
ax[2].set_title('psi')
plt.show()

translation_errors = []
phi_errors = []
theta_errors = []
psi_errors = []
translation_errors.append(th.mean(th.norm(translation_model - translation_target, dim=0)).detach().cpu().item())
phi_errors.append(th.norm(phi_rad_model - phi_rad_target).detach().cpu().item())
theta_errors.append(th.norm(theta_rad_model - theta_rad_target).detach().cpu().item())
psi_errors.append(th.norm(psi_rad_model - psi_rad_target).detach().cpu().item())
print(f'L2 translation error: {translation_errors[0]}')
print(f'L2 phi         error: {phi_errors[0]}')

sino_model = ray_transform(target, phi_rad_model, theta_rad_model, psi_rad_model, translation_model)

plotmosaic(sino_model.squeeze().cpu().numpy(), 'Sino model')
# %%
losses = []
model.requires_grad = True
phi_rad_model.requires_grad = True
theta_rad_model.requires_grad = True
psi_rad_model.requires_grad = True
translation_model.requires_grad = False

lr_phi = 8e-6
lr_theta = 8e-6
lr_psi = 1e-6
lr_translation = 1e-6
lr_model = 40

optimizer_model = Adam([model], lr_model)
optimizer_phi = Adam([phi_rad_model], lr_phi)
optimizer_theta = Adam([theta_rad_model], lr_theta)
optimizer_psi = Adam([psi_rad_model], lr_psi)
optimizer_translations = Adam([translation_model], lr_translation)

lam = 5e-3
loss_fn = mse_loss
scheduler = ExponentialLR(optimizer_model, gamma=1 - 1e-4)
i = 0
j = 0
start_refine = 100

random_order = np.random.permutation(n_theta)
translation_shifts = np.zeros((n_theta, 2))

for epoch in range(500):
    if epoch % 2 == 0 or epoch < start_refine:
        optimizer_model.zero_grad()
        # model.requires_grad = True
        # angles_model.requires_grad = True
    else:
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
        optimizer_psi.zero_grad()
        # model.requires_grad = True
        # angles_model.requires_grad = True
    # vol, phi_rad, theta_rad, psi_rad, translation
    sino_model = ray_transform(model, phi_rad_model, theta_rad_model, psi_rad_model, translation_model)
    loss = loss_fn(sino_model, sino_target)  # + lam * regularization_term(model)
    losses.append(loss.item())
    loss.backward()

    # if epoch % 20 == 0:
    #     plotmosaic(sino_target.squeeze().cpu().numpy(), f'Sino target eopch {epoch}')

    if epoch % 2 == 0 or epoch < start_refine:
        if epoch % 10 == 0:
            print(f'{epoch}: loss = {loss.item()}')
        optimizer_model.step()
        model.requires_grad = False
        model[model < 0] = 0

        # m = model.detach().cpu().numpy().squeeze()
        # (fgp_gpu3D, info_vec_gpu) = FGP_TV(m,
        #                                    pars['regularisation_parameter'],
        #                                    pars['number_of_iterations'],
        #                                    pars['tolerance_constant'],
        #                                    pars['methodTV'],
        #                                    pars['nonneg'], 'gpu')
        # model[0, 0, :, :, :] = th.as_tensor(fgp_gpu3D, device=device)
        model.requires_grad = True
    else:
        # if i % 2 == 0:
        optimizer_phi.step()
        optimizer_theta.step()
        optimizer_psi.step()

        phi_errors.append(th.norm(phi_rad_model - phi_rad_target).detach().cpu().item())
        theta_errors.append(th.norm(theta_rad_model - theta_rad_target).detach().cpu().item())
        psi_errors.append(th.norm(psi_rad_model - psi_rad_target).detach().cpu().item())
        translation_errors.append(th.mean(th.norm(translation_model - translation_target, dim=0)).detach().cpu().item())

        phimg = th.max(th.abs(phi_rad_model.grad)) * lr_phi
        thimg = th.max(th.abs(theta_rad_model.grad)) * lr_theta
        psimg = th.max(th.abs(psi_rad_model.grad)) * lr_psi

        optim_index = random_order[j]
        norm_before = th.norm(sino_model[optim_index, 0].detach() - sino_target[optim_index, 0].detach())
        shift, _, diffphase = phase_cross_correlation(sino_model[optim_index, 0].detach().cpu().numpy(),
                                                      sino_target[optim_index, 0].detach().cpu().numpy(),
                                                      upsample_factor=10)
        shift = np.clip(shift, a_max=0.02, a_min=-0.02)
        # print(shift)
        translation_model[:, optim_index] -= th.as_tensor(shift, device=device)
        sino_target[optim_index, 0] = th.as_tensor(ifft2(fourier_shift(
            fft2(sino_target[optim_index, 0].detach().cpu().numpy()), shift)).real)
        norm_after = th.norm(sino_model[optim_index, 0].detach() - sino_target[optim_index, 0].detach())
        # print(f'norm before, after align {norm_before:2.2f},{norm_after:2.2f}')
        # if i % 10 == 0:
        #     print(f'phi theta psi max grad: {phimg},{thimg},{psimg}')
        # else:
        #     # translation_model.requires_grad = False
        #     translation_model.grad *= -1
        #     # translation_model.requires_grad = True
        #     optimizer_translations.step()
        #     translation_errors.append(th.norm(translation_model - translation_target).detach().cpu().item())
        # print(f'transl max grad: {th.max(th.abs(translation_model.grad)) * lr_translation}')
        # print(th.max(th.abs(phi_rad_model.grad)) * lr_angles)

        # print(f'th.norm(theta_target-theta_model) = {th.norm(theta_target-theta_model)}')
        # print(f'theta_model[:5] = {angles_model[:5]}')
        i += 1
        j += 1
        j = j % n_theta
    # scheduler.step()
# %%
# fig, ax = plt.subplots()
# ax.scatter(np.arange(len(phi_rad_model)), phi_rad_target.detach().cpu().numpy().squeeze())
# ax.scatter(np.arange(len(phi_rad_model)), phi_rad_model.detach().cpu().numpy().squeeze(), marker='x')
#
# plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
axs = ax.ravel
ax[0].scatter(np.arange(len(phi_rad_model)), phi_rad_model.detach().cpu().numpy().squeeze())
ax[0].scatter(np.arange(len(phi_rad_model)), phi_rad_target.detach().cpu().numpy().squeeze(), marker='x')
ax[1].scatter(np.arange(len(phi_rad_model)), translation_model[0].detach().cpu().numpy().squeeze())
ax[1].scatter(np.arange(len(phi_rad_model)), translation_target[0].detach().cpu().numpy().squeeze(), marker='x')
ax[2].scatter(np.arange(len(phi_rad_model)), translation_model[1].detach().cpu().numpy().squeeze())
ax[2].scatter(np.arange(len(phi_rad_model)), translation_target[1].detach().cpu().numpy().squeeze(), marker='x')
fig.suptitle('After optimization')
ax[0].set_title('phi')
ax[1].set_title('theta')
ax[2].set_title('psi')
plt.show()
# %%
translation_errors = np.array(translation_errors)
phi_errors = np.array(phi_errors)
losses = np.array(losses)
fig, ax = plt.subplots(3, 2, figsize=(10, 5))
# ax[0].scatter(np.arange(len(translation_errors)),translation_errors)
ax[0,0].scatter(np.arange(len(phi_errors)), phi_errors)
ax[0,1].scatter(np.arange(len(theta_errors)), theta_errors)
ax[1,0].scatter(np.arange(len(psi_errors)), psi_errors)
ax[1,1].scatter(np.arange(len(losses)), np.log10(losses))
ax[2,0].scatter(np.arange(len(psi_errors)), translation_errors)
# ax[0].set_title('Translation error')
ax[0,0].set_title('Phi error')
ax[0,1].set_title('theta error')
ax[1,0].set_title('psi error')
ax[1,1].set_title('reconstruction loss')
ax[2,0].set_title('translation errors')
plt.show()
# %%``````````````````````````````````````````````````````````````````````````````````````````````````
m = model.detach().cpu().numpy().squeeze()
t = target.detach().cpu().numpy().squeeze()
fig, ax = plt.subplots()
# imax = ax.imshow(np.abs(m - t)/t)
imax = ax.imshow(m[32])
plt.colorbar(imax)
plt.show()
# %%


# pars = {'algorithm' : TGV, \
#         'regularisation_parameter': 7e-2, \
#         'alpha1':1.0,\
#         'alpha0':2.0,\
#         'number_of_iterations' :500 ,\
#         'LipshitzConstant' :12 ,\
#         'tolerance_constant':1e-06}
#
# (tgv_gpu3D,info_vec_gpu)  = TGV(m,
#               pars['regularisation_parameter'],
#               pars['alpha1'],
#               pars['alpha0'],
#               pars['number_of_iterations'],
#               pars['LipshitzConstant'],
#               pars['tolerance_constant'],'gpu')
fig, ax = plt.subplots()
imax = ax.imshow(tgv_gpu3D[32])
plt.colorbar(imax)
plt.show()
