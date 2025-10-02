import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import time

# Load original image
original_image = cv2.imread('./data/9FdkRecon-ushort-300x300x300.view.raw.to-byte299.png', cv2.IMREAD_GRAYSCALE)
original_image = cv2.resize(original_image, (300, 300))

# Display original image
plt.figure(figsize=(10, 8))
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.show()

def optimize_metal_artifact_removal(input_image, threshold_range=(80, 180), step=5,
                                    filter_options=['hann', 'ramp', 'cosine']):
    """
    Parameter optimization for metal artifact removal
    :param input_image: Input image
    :param threshold_range: Threshold search range
    :param step: Threshold search step size
    :param filter_options: Inverse Radon filter options
    :return: Optimal parameters and results
    """
    best_parameters = {}
    best_score = -np.inf
    best_result = None
    optimization_results = []

    thresholds = range(threshold_range[0], threshold_range[1] + 1, step)
    total_combinations = len(thresholds) * len(filter_options)

    with tqdm(total=total_combinations, desc="Parameter Optimization") as progress_bar:
        for threshold_value in thresholds:
            for filter_name in filter_options:
                start_time = time.time()

                # 1. Threshold-based segmentation
                metal_mask = np.zeros_like(input_image)
                non_metal_image = np.zeros_like(input_image)
                metal_mask[input_image >= threshold_value] = 255
                non_metal_image[input_image < threshold_value] = input_image[input_image < threshold_value]

                # 2. Radon Transform
                projection_angles = np.arange(0, 180, 1)
                radon_metal = radon(metal_mask, theta=projection_angles, circle=False)
                radon_non_metal = radon(non_metal_image, theta=projection_angles, circle=False)

                # 3. Boundary detection and interpolation
                metal_boundary = np.zeros_like(radon_metal)
                for projection_idx in range(radon_metal.shape[1]):
                    for distance_idx in range(1, radon_metal.shape[0] - 1):
                        if radon_metal[distance_idx, projection_idx] > 0:
                            if (radon_metal[distance_idx + 1, projection_idx] > 0 and
                                radon_metal[distance_idx - 1, projection_idx] == 0) or \
                               (radon_metal[distance_idx + 1, projection_idx] == 0 and
                                radon_metal[distance_idx - 1, projection_idx] > 0):
                                metal_boundary[distance_idx, projection_idx] = 100

                for projection_idx in range(radon_metal.shape[1]):
                    upper_index = -1
                    for distance_idx in range(radon_metal.shape[0]):
                        if metal_boundary[distance_idx, projection_idx] == 100:
                            if upper_index == -1:
                                upper_index = distance_idx
                            else:
                                midpoint = (upper_index + distance_idx) // 2
                                if radon_metal[midpoint, projection_idx] > 0:
                                    for k in range(upper_index, distance_idx + 1):
                                        weight = (k - upper_index) / (distance_idx - upper_index)
                                        radon_non_metal[k, projection_idx] = (1 - weight) * radon_non_metal[upper_index, projection_idx] + \
                                                                          weight * radon_non_metal[distance_idx, projection_idx]
                                    upper_index = -1
                                else:
                                    upper_index = distance_idx

                # 4. Inverse Radon Transform
                reconstructed_image = iradon(radon_non_metal, theta=projection_angles, circle=False, filter_name=filter_name)
                reconstructed_image = cv2.resize(reconstructed_image, (300, 300))

                # 5. Metal compensation and normalization
                max_intensity = np.max(reconstructed_image)
                for i in range(300):
                    for j in range(300):
                        if metal_mask[i, j] > 0:
                            reconstructed_image[i, j] = max_intensity

                result_image = (reconstructed_image - np.min(reconstructed_image)) / \
                              (np.max(reconstructed_image) - np.min(reconstructed_image))

                # 6. Quality assessment (SSIM in non-metal regions)
                non_metal_region = (metal_mask == 0)
                if np.any(non_metal_region):
                    reference_image = input_image.copy()
                    reference_image[metal_mask > 0] = np.mean(input_image[non_metal_region])
                    quality_score = ssim(reference_image[non_metal_region].flatten(),
                                        result_image[non_metal_region].flatten(),
                                        data_range=1.0)
                else:
                    quality_score = -np.inf

                # Record results
                elapsed = time.time() - start_time
                optimization_results.append({
                    'threshold': threshold_value,
                    'filter': filter_name,
                    'score': quality_score,
                    'time': elapsed,
                    'result': result_image
                })

                # Update best solution
                if quality_score > best_score:
                    best_score = quality_score
                    best_parameters = {'threshold': threshold_value, 'filter': filter_name}
                    best_result = result_image

                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Thresh={threshold_value}, Filter={filter_name}, Score={quality_score:.4f}")

    return best_parameters, best_result, optimization_results

# Execute optimization
optimal_params, optimal_result, all_results = optimize_metal_artifact_removal(
    original_image,
    threshold_range=(90, 150),
    step=5,
    filter_options=['hann', 'ramp', 'cosine', 'shepp-logan']
)

# Output optimal parameters
print(f"\nOptimal Parameters: Threshold={optimal_params['threshold']}, Filter={optimal_params['filter']}")

# Visualize optimal result
plt.figure(figsize=(10, 8))
plt.imshow(optimal_result, cmap='gray')
plt.title(f"Optimal Reconstruction (Threshold={optimal_params['threshold']}, Filter={optimal_params['filter']})")
plt.show()

# Analyze threshold effects
threshold_values = [r['threshold'] for r in all_results]
quality_scores = [r['score'] for r in all_results]
filter_types = [r['filter'] for r in all_results]

plt.figure(figsize=(12, 6))
plt.subplot(121)
for filter_name in set(filter_types):
    mask = [f == filter_name for f in filter_types]
    plt.plot(np.array(threshold_values)[mask], np.array(quality_scores)[mask], 'o-', label=filter_name)
plt.xlabel('Threshold Value')
plt.ylabel('SSIM Quality Score')
plt.title('Threshold Optimization with Different Filters')
plt.legend()
plt.grid(True)

# Display top results
top_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:3]
plt.subplot(122)
for i, res in enumerate(top_results):
    plt.plot(res['threshold'], res['score'], 'o', markersize=10, label=f"Rank {i + 1}")
plt.xlabel('Threshold Value')
plt.ylabel('SSIM Quality Score')
plt.title('Optimal Results')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize top reconstructions
plt.figure(figsize=(15, 5))
for i, res in enumerate(top_results):
    plt.subplot(1, 3, i + 1)
    plt.imshow(res['result'], cmap='gray')
    plt.title(f"Threshold={res['threshold']}, Filter={res['filter']}\nScore={res['score']:.4f}")
plt.tight_layout()
plt.show()

def reconstruct_image(input_image, threshold, filter_name):
    """Image reconstruction pipeline with specified parameters"""
    # Threshold segmentation
    metal_mask = np.zeros_like(input_image)
    non_metal_image = np.zeros_like(input_image)
    metal_mask[input_image >= threshold] = 255
    non_metal_image[input_image < threshold] = input_image[input_image < threshold]

    # Radon Transform
    projection_angles = np.arange(0, 180, 1)
    radon_metal = radon(metal_mask, theta=projection_angles, circle=False)
    radon_non_metal = radon(non_metal_image, theta=projection_angles, circle=False)

    # Boundary detection and interpolation
    metal_boundary = np.zeros_like(radon_metal)
    for projection_idx in range(radon_metal.shape[1]):
        for distance_idx in range(1, radon_metal.shape[0] - 1):
            if radon_metal[distance_idx, projection_idx] > 0:
                if (radon_metal[distance_idx + 1, projection_idx] > 0 and
                    radon_metal[distance_idx - 1, projection_idx] == 0) or \
                   (radon_metal[distance_idx + 1, projection_idx] == 0 and
                    radon_metal[distance_idx - 1, projection_idx] > 0):
                    metal_boundary[distance_idx, projection_idx] = 100

    for projection_idx in range(radon_metal.shape[1]):
        upper_index = -1
        for distance_idx in range(radon_metal.shape[0]):
            if metal_boundary[distance_idx, projection_idx] == 100:
                if upper_index == -1:
                    upper_index = distance_idx
                else:
                    midpoint = (upper_index + distance_idx) // 2
                    if radon_metal[midpoint, projection_idx] > 0:
                        for k in range(upper_index, distance_idx + 1):
                            weight = (k - upper_index) / (distance_idx - upper_index)
                            radon_non_metal[k, projection_idx] = (1 - weight) * radon_non_metal[upper_index, projection_idx] + \
                                                              weight * radon_non_metal[distance_idx, projection_idx]
                        upper_index = -1
                    else:
                        upper_index = distance_idx

    # Inverse Radon Transform
    reconstructed_image = iradon(radon_non_metal, theta=projection_angles, circle=False, filter_name=filter_name)
    reconstructed_image = cv2.resize(reconstructed_image, (300, 300))

    # Metal compensation and normalization
    max_intensity = np.max(reconstructed_image)
    for i in range(300):
        for j in range(300):
            if metal_mask[i, j] > 0:
                reconstructed_image[i, j] = max_intensity

    result_image = (reconstructed_image - np.min(reconstructed_image)) / \
                  (np.max(reconstructed_image) - np.min(reconstructed_image))
    return result_image, metal_mask, non_metal_image

# Final reconstruction with optimal parameters
final_reconstruction, metal_mask, non_metal_region = reconstruct_image(
    original_image,
    optimal_params['threshold'],
    optimal_params['filter']
)

# Comprehensive visualization
plt.figure(figsize=(20, 12))

plt.subplot(231)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(232)
plt.imshow(metal_mask, cmap='gray')
plt.title(f'Metal Mask (Threshold={optimal_params["threshold"]})')

plt.subplot(233)
plt.imshow(non_metal_region, cmap='gray')
plt.title('Non-Metal Region')

projection_angles = np.arange(0, 180, 1)
radon_original = radon(original_image, theta=projection_angles, circle=False)
plt.subplot(234)
plt.imshow(radon_original, cmap='gray', aspect='auto',
           extent=(0, 180, -radon_original.shape[0] // 2, radon_original.shape[0] // 2))
plt.title('Original Sinogram')
plt.xlabel('Projection Angle (degrees)')
plt.ylabel('Detector Position')

radon_metal = radon(metal_mask, theta=projection_angles, circle=False)
plt.subplot(235)
plt.imshow(radon_metal, cmap='gray', aspect='auto',
           extent=(0, 180, -radon_metal.shape[0] // 2, radon_metal.shape[0] // 2))
plt.title('Metal-Only Sinogram')
plt.xlabel('Projection Angle (degrees)')
plt.ylabel('Detector Position')

plt.subplot(236)
plt.imshow(final_reconstruction, cmap='gray')
plt.title(f'Final Reconstruction (Filter={optimal_params["filter"]})')

plt.tight_layout()
plt.show()

# Save optimal result
# cv2.imwrite('optimal_reconstruction.png', (final_reconstruction * 255).astype(np.uint8))