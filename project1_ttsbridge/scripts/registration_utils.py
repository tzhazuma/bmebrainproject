"""
Image registration and spatial normalization utilities.
Uses SimpleITK as a lightweight replacement for ANTs/FSL.
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path


def rigid_register(
    moving_path,
    fixed_path,
    output_path=None,
    transform_path=None,
):
    """
    Rigid (6-DOF) registration using mutual information.

    Args:
        moving_path: Path to moving image (NIfTI)
        fixed_path: Path to fixed/template image (NIfTI)
        output_path: Path to save registered image
        transform_path: Path to save transform

    Returns:
        registered_image: sitk.Image
        transform: sitk.Transform
    """
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(initial_transform)

    final_transform = reg.Execute(fixed, moving)

    registered = sitk.Resample(
        moving, fixed, final_transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID(),
    )

    if output_path:
        sitk.WriteImage(registered, str(output_path))
    if transform_path:
        sitk.WriteTransform(final_transform, str(transform_path))

    return registered, final_transform


def affine_register(
    moving_path,
    fixed_path,
    output_path=None,
    transform_path=None,
):
    """
    Affine (12-DOF) registration.

    Args:
        moving_path: Moving image path
        fixed_path: Fixed/template image path
        output_path: Output path for registered image
        transform_path: Output path for transform

    Returns:
        registered_image, transform
    """
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.5,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(initial_transform)

    final_transform = reg.Execute(fixed, moving)

    registered = sitk.Resample(
        moving, fixed, final_transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID(),
    )

    if output_path:
        sitk.WriteImage(registered, str(output_path))
    if transform_path:
        sitk.WriteTransform(final_transform, str(transform_path))

    return registered, final_transform


def resample_to_target(
    input_path,
    output_path,
    target_spacing=(0.8, 0.8, 0.8),
    interpolator='linear',
):
    """
    Resample NIfTI to target spacing.

    Args:
        input_path: Input NIfTI
        output_path: Output NIfTI
        target_spacing: (z, y, x) in mm
        interpolator: 'linear' or 'nearest' (for labels)
    """
    img = sitk.ReadImage(str(input_path))
    original_spacing = np.array(img.GetSpacing())
    original_size = np.array(img.GetSize())

    new_size = (original_size * original_spacing / np.array(target_spacing)).astype(int)
    new_size = new_size.tolist()[::-1]  # ITK uses (x, y, z) order

    interp = sitk.sitkLinear if interpolator == 'linear' else sitk.sitkNearestNeighbor

    resampled = sitk.Resample(
        img, new_size,
        sitk.Transform(),
        interp,
        img.GetOrigin(),
        target_spacing[::-1],  # ITK (x, y, z) order
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )

    sitk.WriteImage(resampled, str(output_path))
    return resampled


def coregister_pet_to_mri(
    pet_path,
    mri_path,
    output_path=None,
    dof=6,
):
    """
    Co-register PET to MRI.

    Args:
        pet_path: PET image path
        mri_path: MRI reference image path
        output_path: Output path
        dof: Degrees of freedom (6=rigid, 12=affine)

    Returns:
        registered_pet
    """
    if dof == 6:
        return rigid_register(pet_path, mri_path, output_path)
    else:
        return affine_register(pet_path, mri_path, output_path)


def spatial_normalize_to_template(
    input_path,
    template_path,
    output_path=None,
    level='affine',
):
    """
    Spatial normalization to a template (MNI-like).

    Args:
        input_path: Subject image
        template_path: Template image (e.g., MNI152)
        output_path: Output image in template space
        level: 'affine' or 'rigid'

    Returns:
        normalized_image
    """
    if level == 'rigid':
        return rigid_register(input_path, template_path, output_path)
    return affine_register(input_path, template_path, output_path)


def create_infant_template(
    image_paths,
    output_path,
):
    """
    Build a study-specific template from a list of infant images.
    Iterative rigid+affine registration to mean.

    Args:
        image_paths: List of NIfTI paths
        output_path: Output template path
    """
    if len(image_paths) == 0:
        raise ValueError("No images provided")

    images = [sitk.ReadImage(str(p), sitk.sitkFloat32) for p in image_paths]

    # Initialize template as first image
    template = images[0]

    for iteration in range(3):
        registered = [template]
        transforms = []

        for img in images[1:]:
            reg, _ = affine_register(
                img_path=None,
                fixed_path=None,
                moving=img,
                fixed=template,
            )
            registered.append(reg)

        template = sitk.Mean(registered)
        print(f'  Iteration {iteration + 1}: template built from {len(registered)} images')

    sitk.WriteImage(template, str(output_path))
    return template


# Helper to handle in-memory images
def rigid_register_img(moving, fixed):
    """In-memory rigid registration."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerAsGradientDescent(1.0, 200, 1e-6, 10)
    reg.SetOptimizerScalesFromPhysicalShift()

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(initial)

    transform = reg.Execute(fixed, moving)
    registered = sitk.Resample(
        moving, fixed, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID(),
    )
    return registered, transform
