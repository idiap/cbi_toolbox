import argparse
import json
import numpy as np
import os

import cbi_toolbox as cbi
import cbi_toolbox.files
import cbi_toolbox.images
import cbi_toolbox.arrays

SINO_SUFFIX = 'sino'
THETA_SUFFIX = 'theta'

parser = argparse.ArgumentParser()

parser.add_argument('conf', metavar='CONFIG_FILE.json', help='json config file', type=str)
parser.add_argument('--overwrite', action='store_true', help='force overwrite existing output dir')

args = parser.parse_args()

print('Loading config file {}'.format(args.conf))
with open(args.conf, 'r') as conf_file:
    conf_dict = json.load(conf_file)
print(conf_dict)

image_path = conf_dict['preprocessing']['tiff']
reference_path = conf_dict['preprocessing']['ref_tiff']
name = conf_dict['name']
padding = conf_dict['preprocessing']['padding']

output_dir = os.path.join(conf_dict['output_path'], name)
print('Creating output directory: {}'.format(output_dir))
if os.path.exists(output_dir):
    if args.overwrite:
        answer = 'Y'
    else:
        answer = input('\nExperiment exists, overwrite? (y/[n]) > ')
    if answer.upper() != 'Y':
        print('Skip preprocessing')
        exit(0)
    else:
        print('Overriding existing experiment at {}'.format(output_dir))
        import shutil
        shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=False)

output_file = os.path.join(output_dir, '{}_{}.npy'.format(name, SINO_SUFFIX))

print('Loading images')

sinogram, metadata = cbi.files.load_mm_ome_tiff(image_path)
angle_step = metadata['Summary']['UserData']['AngleStep']['PropVal']

reference, ref_metadata = cbi.files.load_mm_ome_tiff(reference_path, True)

sinogram = np.squeeze(sinogram)
sinogram = np.ascontiguousarray(np.transpose(sinogram, (2, 1, 0)))

reference = np.squeeze(reference)
reference = reference[..., np.newaxis]
reference = np.ascontiguousarray(np.transpose(reference, (2, 1, 0)))

# Preprocess the image
print('Preprocessing')
sinogram = reference - sinogram
sinogram = cbi.images.erase_corners(sinogram, 100)
sinogram = cbi.images.remove_background_illumination(sinogram, threshold=500, hole_size=50, margin_size=300)

# Compensate lateral shift
print('Compensating lateral shift')
x_project = sinogram.sum(-1)
x_center = sinogram.shape[-1] / 2

x_com = cbi.arrays.center_of_mass(x_project)
x_com = x_com - x_center
x_com = np.round(x_com).astype(int)

sinogram = cbi.arrays.roll_array(sinogram, -x_com, axis=1)

print('Padding')
sinogram = sinogram[:, padding:-padding, ...]

# TODO remove this
sinogram = sinogram[:180, :, 1250:2000]

n_angles = sinogram.shape[0]
theta = np.arange(n_angles) * angle_step

# TODO fold sinogram if bigger than 360

print('Saving to {}'.format(output_file))
np.save(output_file, sinogram)

theta_file = os.path.join(output_dir, '{}_{}.npy'.format(name, THETA_SUFFIX))
print('Saving theta to {}'.format(theta_file))
np.save(theta_file, theta)
