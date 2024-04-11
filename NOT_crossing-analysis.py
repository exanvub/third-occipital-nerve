import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
from numpy.linalg import norm
import pandas as pd
import os

# Set the working directory
# os.chdir('/Users/nicolas/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Onderzoek/n occipitalis tertius/microscribe DATA/microscribe n occ tertius/')

# Read the data from the Excel file
df = pd.read_excel('DATA/NOT 25.xlsx')

Specmen = df.loc[df['label'] == 'Specimen', 'x1'].values[0]
print(f"Specmen: {Specmen}")
# Extract coordinates for various labels
L_IVF = df.loc[df['label'] == 'L_IVF', ['x1', 'y1', 'z1']].values
L_dors_ramus_mid = df.loc[df['label'] == 'L_dors_ramus_mid', ['x1', 'y1', 'z1']].values
L_Split = df.loc[df['label'] == 'L_Split', ['x1', 'y1', 'z1']].values
L_half_NOT = df.loc[df['label'] == 'L_half_NOT', ['x1', 'y1', 'z1']].values
L_bocht_NOT = df.loc[df['label'] == 'L_bocht_NOT', ['x1', 'y1', 'z1']].values
L_Facet_MED = df.loc[df['label'] == 'L_Facet_MED', ['x1', 'y1', 'z1']].values
L_Facet_Mid = df.loc[df['label'] == 'L_Facet_Mid', ['x1', 'y1', 'z1']].values
L_Facet_POST = df.loc[df['label'] == 'L_Facet_POST', ['x1', 'y1', 'z1']].values
L_Kruising = df.loc[df['label'] == 'L_Kruising', ['x1', 'y1', 'z1']].values

# If L_Kruising coordinates are NaN, use values from L_Facet_POST
if np.isnan(L_Kruising).all():
    L_Kruising = L_Facet_POST
    print("L_Kruising is missing, using L_Facet_POST coordinates instead.")

R_IVF = df.loc[df['label'] == 'R_IVF', ['x1', 'y1', 'z1']].values
R_dors_ramus_mid = df.loc[df['label'] == 'R_dors_ramus_mid', ['x1', 'y1', 'z1']].values
R_Split = df.loc[df['label'] == 'R_Split', ['x1', 'y1', 'z1']].values
R_half_NOT = df.loc[df['label'] == 'R_half_NOT', ['x1', 'y1', 'z1']].values
R_bocht_NOT = df.loc[df['label'] == 'R_bocht_NOT', ['x1', 'y1', 'z1']].values
R_Facet_MED = df.loc[df['label'] == 'R_Facet_MED', ['x1', 'y1', 'z1']].values
R_Facet_Mid = df.loc[df['label'] == 'R_Facet_Mid', ['x1', 'y1', 'z1']].values
R_Facet_POST = df.loc[df['label'] == 'R_Facet_POST', ['x1', 'y1', 'z1']].values
R_Kruising = df.loc[df['label'] == 'R_Kruising', ['x1', 'y1', 'z1']].values

# If R_Kruising coordinates are NaN, use values from R_Facet_POST
if np.isnan(R_Kruising).all():
    R_Kruising = R_Facet_POST
    print("R_Kruising is missing, using R_Facet_POST coordinates instead.")

# Define coordinates for sets A and B
coordinates_L_NOT = {
    "A_1": L_IVF[0],
    "A_2": L_dors_ramus_mid[0],
    "A_3": L_Split[0],
    "A_4": L_half_NOT[0],
    "A_5": L_bocht_NOT[0],
}

coordinates_L_Facet = {
    "B_1": L_Facet_MED[0],
    "B_2": L_Facet_Mid[0],
    "B_3": L_Facet_POST[0],
}

coordinates_R_NOT = {
    "C_1": R_IVF[0],
    "C_2": R_dors_ramus_mid[0],
    "C_3": R_Split[0],
    "C_4": R_half_NOT[0],
    "C_5": R_bocht_NOT[0],
}

coordinates_R_Facet = {
    "D_1": R_Facet_MED[0],
    "D_2": R_Facet_Mid[0],
    "D_3": R_Facet_POST[0],
}

# Extract the x, y, and z values from the coordinates for set A
x_A = [coord[0] for coord in coordinates_L_NOT.values()]
y_A = [coord[1] for coord in coordinates_L_NOT.values()]
z_A = [coord[2] for coord in coordinates_L_NOT.values()]

# Extract the x, y, and z values from the coordinates for set B
x_B = [coord[0] for coord in coordinates_L_Facet.values()]
y_B = [coord[1] for coord in coordinates_L_Facet.values()]
z_B = [coord[2] for coord in coordinates_L_Facet.values()]

# Extract the x, y, and z values from the coordinates for set C
x_C = [coord[0] for coord in coordinates_R_NOT.values()]
y_C = [coord[1] for coord in coordinates_R_NOT.values()]
z_C = [coord[2] for coord in coordinates_R_NOT.values()]

# Extract the x, y, and z values from the coordinates for set D
x_D = [coord[0] for coord in coordinates_R_Facet.values()]
y_D = [coord[1] for coord in coordinates_R_Facet.values()]
z_D = [coord[2] for coord in coordinates_R_Facet.values()]


# Add virtual data points for set A
t = np.linspace(0, 1, 30)  # additional points
x_new_A, y_new_A, z_new_A = splev(t, splprep([x_A, y_A, z_A], s=0.0, k=2)[0])

# Add virtual data points for set B
x_new_B, y_new_B, z_new_B = splev(t, splprep([x_B, y_B, z_B], s=0.0, k=2)[0])

# Add virtual data points for set C
x_new_C, y_new_C, z_new_C = splev(t, splprep([x_C, y_C, z_C], s=0.0, k=2)[0])

# Add virtual data points for set D
x_new_D, y_new_D, z_new_D = splev(t, splprep([x_D, y_D, z_D], s=0.0, k=2)[0])

# Find the point on each curve closest to the other curve
# and calculate the tangent vectors at the closest points
closest_point_A = None
closest_point_B = None
closest_point_C = None
closest_point_D = None
min_distance_L = None
min_distance_R = None
tangent_vector_A = None
tangent_vector_B = None
tangent_vector_C = None
tangent_vector_D = None

for i in range(len(x_new_A) - 1):
    for j in range(len(x_new_B) - 1):
        current_distance_L = distance.euclidean((x_new_A[i], y_new_A[i], z_new_A[i]), (x_new_B[j], y_new_B[j], z_new_B[j]))
        if min_distance_L is None or current_distance_L < min_distance_L:
            min_distance_L = current_distance_L
            closest_point_A = (x_new_A[i], y_new_A[i], z_new_A[i])
            closest_point_B = (x_new_B[j], y_new_B[j], z_new_B[j])
            # tangent_vector_A = np.array([x_new_A[i + 1] - x_new_A[i], y_new_A[i + 1] - y_new_A[i], z_new_A[i + 1] - z_new_A[i]])
            # tangent_vector_B = np.array([x_new_B[j + 1] - x_new_B[j], y_new_B[j + 1] - y_new_B[j], z_new_B[j + 1] - z_new_B[j]])

for i in range(len(x_new_C) - 1):
    for j in range(len(x_new_D) - 1):
        current_distance_R = distance.euclidean((x_new_C[i], y_new_C[i], z_new_C[i]), (x_new_D[j], y_new_D[j], z_new_D[j]))
        if min_distance_R is None or current_distance_R < min_distance_R:
            min_distance_R = current_distance_R
            closest_point_C = (x_new_C[i], y_new_C[i], z_new_C[i])
            closest_point_D = (x_new_D[j], y_new_D[j], z_new_D[j])
            # tangent_vector_C = np.array([x_new_C[i + 1] - x_new_C[i], y_new_C[i + 1] - y_new_C[i], z_new_C[i + 1] - z_new_C[i]])
            # tangent_vector_D = np.array([x_new_D[j + 1] - x_new_D[j], y_new_D[j + 1] - y_new_D[j], z_new_D[j + 1] - z_new_D[j]])


# # Calculate the angle between the two tangent vectors
# angle_rad_L = np.arccos(np.dot(tangent_vector_A, tangent_vector_B) / (norm(tangent_vector_A) * norm(tangent_vector_B)))
# angle_deg_L = np.degrees(angle_rad_L)

# angle_rad_R = np.arccos(np.dot(tangent_vector_C, tangent_vector_D) / (norm(tangent_vector_C) * norm(tangent_vector_D)))
# angle_deg_R = np.degrees(angle_rad_R)

# # Print the angle between the two curves
# print(f"Angle between the two curves at the closest points: {angle_deg_L:.2f} degrees")
# print(f"Angle between the two curves at the closest points: {angle_deg_R:.2f} degrees")

# Coordinates for 'L_Kruising'
L_Kruising_coords = L_Kruising[0]
R_Kruising_coords = R_Kruising[0]

# Find the closest points on set A and set B to 'L_Kruising'
closest_point_A_to_L_Kruising = None
closest_point_B_to_L_Kruising = None
closest_point_C_to_R_Kruising = None
closest_point_D_to_R_Kruising = None

min_distance_A_to_L_Kruising = None
min_distance_B_to_L_Kruising = None
min_distance_C_to_R_Kruising = None
min_distance_D_to_R_Kruising = None


for i in range(len(x_new_A)):
    current_distance_A_to_L = distance.euclidean((x_new_A[i], y_new_A[i], z_new_A[i]), L_Kruising_coords)
    if min_distance_A_to_L_Kruising is None or current_distance_A_to_L < min_distance_A_to_L_Kruising:
        min_distance_A_to_L_Kruising = current_distance_A_to_L
        closest_point_A_to_L_Kruising = (x_new_A[i], y_new_A[i], z_new_A[i])

for j in range(len(x_new_B)):
    current_distance_B_to_L = distance.euclidean((x_new_B[j], y_new_B[j], z_new_B[j]), L_Kruising_coords)
    if min_distance_B_to_L_Kruising is None or current_distance_B_to_L < min_distance_B_to_L_Kruising:
        min_distance_B_to_L_Kruising = current_distance_B_to_L
        closest_point_B_to_L_Kruising = (x_new_B[j], y_new_B[j], z_new_B[j])

for i in range(len(x_new_C)):
    current_distance_C_to_L = distance.euclidean((x_new_C[i], y_new_C[i], z_new_C[i]), R_Kruising_coords)
    if min_distance_C_to_R_Kruising is None or current_distance_C_to_L < min_distance_C_to_R_Kruising:
        min_distance_C_to_R_Kruising = current_distance_C_to_L
        closest_point_C_to_R_Kruising = (x_new_C[i], y_new_C[i], z_new_C[i])

for j in range(len(x_new_D)):
    current_distance_D_to_L = distance.euclidean((x_new_D[j], y_new_D[j], z_new_D[j]), R_Kruising_coords)
    if min_distance_D_to_R_Kruising is None or current_distance_D_to_L < min_distance_D_to_R_Kruising:
        min_distance_D_to_R_Kruising = current_distance_D_to_L
        closest_point_D_to_R_Kruising = (x_new_D[j], y_new_D[j], z_new_D[j])

# Calculate the tangent vectors at the closest points to 'L_Kruising'
num_points_for_tangent = 10  # Number of points used to calculate the tangent vector
max_index_A = len(x_new_A) - 1
max_index_B = len(x_new_B) - 1
max_index_C = len(x_new_C) - 1
max_index_D = len(x_new_D) - 1

def calculate_tangent(curve_points, t_values, t):
    
    # Check if points are collinear
    if np.linalg.matrix_rank(curve_points) == 1:
        # Points are collinear, use finite difference method to calculate tangent
        tangent_vector = np.gradient(curve_points, t_values, axis=0)[-1]
    else:
        # Interpolate the curve
        tck, _ = splprep(curve_points.T, s=0)
        # Evaluate the derivative of the spline at the given parameter value
        tangent_vector = splev(t, tck, der=1)
    
    return tangent_vector


# Now call the function with the combined curve points
tangent_vector_A_to_L_Kruising = calculate_tangent(np.array([x_new_A, y_new_A, z_new_A]).T, t, t[np.argmin(np.linalg.norm(np.array([x_new_A, y_new_A, z_new_A]).T - closest_point_A_to_L_Kruising, axis=1))])
tangent_vector_B_to_L_Kruising = calculate_tangent(np.array([x_new_B, y_new_B, z_new_B]).T, t, t[np.argmin(np.linalg.norm(np.array([x_new_B, y_new_B, z_new_B]).T - closest_point_B_to_L_Kruising, axis=1))])
tangent_vector_C_to_R_Kruising = calculate_tangent(np.array([x_new_C, y_new_C, z_new_C]).T, t, t[np.argmin(np.linalg.norm(np.array([x_new_C, y_new_C, z_new_C]).T - closest_point_C_to_R_Kruising, axis=1))])
tangent_vector_D_to_R_Kruising = calculate_tangent(np.array([x_new_D, y_new_D, z_new_D]).T, t, t[np.argmin(np.linalg.norm(np.array([x_new_D, y_new_D, z_new_D]).T - closest_point_D_to_R_Kruising, axis=1))])

# Calculate the angle between the tangent vectors at the closest points to 'L_Kruising'
angle_rad_to_L_Kruising = np.arccos(np.dot(tangent_vector_A_to_L_Kruising, tangent_vector_B_to_L_Kruising) / (norm(tangent_vector_A_to_L_Kruising) * norm(tangent_vector_B_to_L_Kruising)))
angle_deg_to_L_Kruising = np.degrees(angle_rad_to_L_Kruising)

angle_rad_to_R_Kruising = np.arccos(np.dot(tangent_vector_C_to_R_Kruising, tangent_vector_D_to_R_Kruising) / (norm(tangent_vector_C_to_R_Kruising) * norm(tangent_vector_D_to_R_Kruising)))
angle_deg_to_R_Kruising = np.degrees(angle_rad_to_R_Kruising)


# Print the angle between the two curves at the closest points to 'L_Kruising'
print(f"Angle from tangent NOT-Facet L: {angle_deg_to_L_Kruising:.2f} degrees")
print(f"Angle from tangent NOT-Facet R: {angle_deg_to_R_Kruising:.2f} degrees")

# Calculate the distance between closest points on set A and set B
distance_A_to_B = distance.euclidean(closest_point_A, closest_point_B)
distance_C_to_D = distance.euclidean(closest_point_C, closest_point_D)

# Print the distance between closest points on set A and set B
print(f"Distance between Facet and NOT at closest points L: {distance_A_to_B:.2f} mm")
print(f"Distance between Facet and NOT at closest points R: {distance_C_to_D:.2f} mm")

# Calculate the distance between Facet and NOT at the closest points to 'L_Kruising'
L_distance_facet_to_NOT = distance.euclidean(closest_point_A_to_L_Kruising, closest_point_B_to_L_Kruising)
R_distance_facet_to_NOT = distance.euclidean(closest_point_C_to_R_Kruising, closest_point_D_to_R_Kruising)

# Print the distance between Facet and NOT at the closest points to 'L_Kruising'
print(f"Distance between Facet and NOT at the closest points to 'L_Kruising': {L_distance_facet_to_NOT:.2f} mm")
print(f"Distance between Facet and NOT at the closest points to 'R_Kruising': {R_distance_facet_to_NOT:.2f} mm")

def calculate_direction_vector(x_new, y_new, z_new, closest_point_to_Kruising):
    closest_index = None
    min_distance = None

    for i, (x, y, z) in enumerate(zip(x_new, y_new, z_new)):
        current_distance = np.sqrt((x - closest_point_to_Kruising[0])**2 + 
                                   (y - closest_point_to_Kruising[1])**2 + 
                                   (z - closest_point_to_Kruising[2])**2)
        if min_distance is None or current_distance < min_distance:
            min_distance = current_distance
            closest_index = i

    num_points_before = 6
    num_points_after = 12

    indices_before = np.arange(max(0, closest_index - num_points_before), closest_index)
    indices_after = np.arange(closest_index + 1, min(len(x_new), closest_index + 1 + num_points_after))

    x_before = x_new[indices_before]
    y_before = y_new[indices_before]
    z_before = z_new[indices_before]

    x_after = x_new[indices_after]
    y_after = y_new[indices_after]
    z_after = z_new[indices_after]

    before_points = np.vstack((x_before, y_before, z_before)).T
    after_points = np.vstack((x_after, y_after, z_after)).T

    centroid_after = np.mean(after_points, axis=0)
    direction_vector_after = centroid_after - closest_point_to_Kruising
    direction_vector_after = direction_vector_after / np.linalg.norm(direction_vector_after)

    centroid_before = np.mean(before_points, axis=0)
    direction_vector_before = centroid_before - closest_point_to_Kruising
    direction_vector_before = direction_vector_before / np.linalg.norm(direction_vector_before)

    return centroid_after, direction_vector_after, centroid_before, direction_vector_before

centroid_A_after, direction_vector_A_after, centroid_A_before, direction_vector_A_before = calculate_direction_vector(x_new_A, y_new_A, z_new_A, closest_point_A_to_L_Kruising)
centroid_B_after, direction_vector_B_after, centroid_B_before, direction_vector_B_before = calculate_direction_vector(x_new_B, y_new_B, z_new_B, closest_point_B_to_L_Kruising)
centroid_C_after, direction_vector_C_after, centroid_C_before, direction_vector_C_before = calculate_direction_vector(x_new_C, y_new_C, z_new_C, closest_point_C_to_R_Kruising)
centroid_D_after, direction_vector_D_after, centroid_D_before, direction_vector_D_before = calculate_direction_vector(x_new_D, y_new_D, z_new_D, closest_point_D_to_R_Kruising)

# calculate the angle between vector A and vector B
def calculate_angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    angle_rad = np.arccos(dot_product)
    angle = np.degrees(angle_rad)
    return angle

angle_A_B_after = calculate_angle_between_vectors(direction_vector_A_after, direction_vector_B_after)
angle_C_D_after = calculate_angle_between_vectors(direction_vector_C_after, direction_vector_D_after)
angle_A_B_before = calculate_angle_between_vectors(direction_vector_A_before, direction_vector_B_before)
angle_C_D_before = calculate_angle_between_vectors(direction_vector_C_before, direction_vector_D_before)

print(f"Angle L NOT-Facet (after): {angle_A_B_after:.2f} degrees")
print(f"Angle R NOT-Facet (after): {angle_C_D_after:.2f} degrees")
print(f"Angle L NOT-Facet (before): {angle_A_B_before:.2f} degrees")
print(f"Angle R NOT-Facet (before): {angle_C_D_before:.2f} degrees")

# Calculate the length of the Facet curves
L_Facet_length = 0
R_Facet_length = 0

for i in range(len(x_new_B) - 1):
    L_Facet_length += distance.euclidean((x_new_B[i], y_new_B[i], z_new_B[i]), (x_new_B[i + 1], y_new_B[i + 1], z_new_B[i + 1]))

for j in range(len(x_new_D) - 1):
    R_Facet_length += distance.euclidean((x_new_D[j], y_new_D[j], z_new_D[j]), (x_new_D[j + 1], y_new_D[j + 1], z_new_D[j + 1]))

print(f"Length of the Facet curve: {L_Facet_length:.2f} mm")
print(f"Length of the Facet curve: {R_Facet_length:.2f} mm")

# Calculate the length of the L Facet curve from the closest point to 'L_Kruising' to the end of the curve
L_Facet_length_from_closest_point_to_L_Kruising = 0
R_Facet_length_from_closest_point_to_R_Kruising = 0

for i in range(len(x_new_B) - 1):
    if (x_new_B[i], y_new_B[i], z_new_B[i]) == closest_point_B_to_L_Kruising:
        for j in range(i, len(x_new_B) - 1):
            L_Facet_length_from_closest_point_to_L_Kruising += distance.euclidean((x_new_B[j], y_new_B[j], z_new_B[j]), (x_new_B[j + 1], y_new_B[j + 1], z_new_B[j + 1]))

for i in range(len(x_new_D) - 1):
    if (x_new_D[i], y_new_D[i], z_new_D[i]) == closest_point_D_to_R_Kruising:
        for j in range(i, len(x_new_D) - 1):
            R_Facet_length_from_closest_point_to_R_Kruising += distance.euclidean((x_new_D[j], y_new_D[j], z_new_D[j]), (x_new_D[j + 1], y_new_D[j + 1], z_new_D[j + 1]))

print(f"Length of the L Facet curve from the closest point to 'L_Kruising' to the end of the curve: {L_Facet_length_from_closest_point_to_L_Kruising:.2f} mm")
print(f"Length of the R Facet curve from the closest point to 'R_Kruising' to the end of the curve: {R_Facet_length_from_closest_point_to_R_Kruising:.2f} mm")

# Calculate the length of the L Facet curve from the start of the curve to the closest point to 'L_Kruising'
L_Facet_length_from_start_to_closest_point_to_L_Kruising = 0
R_Facet_length_from_start_to_closest_point_to_R_Kruising = 0

for i in range(len(x_new_B) - 1):
    if (x_new_B[i], y_new_B[i], z_new_B[i]) == closest_point_B_to_L_Kruising:
        for j in range(0, i):
            L_Facet_length_from_start_to_closest_point_to_L_Kruising += distance.euclidean((x_new_B[j], y_new_B[j], z_new_B[j]), (x_new_B[j + 1], y_new_B[j + 1], z_new_B[j + 1]))

for i in range(len(x_new_D) - 1):
    if (x_new_D[i], y_new_D[i], z_new_D[i]) == closest_point_D_to_R_Kruising:
        for j in range(0, i):
            R_Facet_length_from_start_to_closest_point_to_R_Kruising += distance.euclidean((x_new_D[j], y_new_D[j], z_new_D[j]), (x_new_D[j + 1], y_new_D[j + 1], z_new_D[j + 1]))

print(f"Length of the L Facet curve from the start of the curve to the closest point to 'L_Kruising': {L_Facet_length_from_start_to_closest_point_to_L_Kruising:.2f} mm")
print(f"Length of the R Facet curve from the start of the curve to the closest point to 'R_Kruising': {R_Facet_length_from_start_to_closest_point_to_R_Kruising:.2f} mm")

# Calculate the percentage of the curve before and after the closest points
L_Facet_percentage_before_closest_point_to_L_Kruising = L_Facet_length_from_start_to_closest_point_to_L_Kruising / L_Facet_length * 100
L_Facet_percentage_after_closest_point_to_L_Kruising = L_Facet_length_from_closest_point_to_L_Kruising / L_Facet_length * 100

R_Facet_percentage_before_closest_point_to_R_Kruising = R_Facet_length_from_start_to_closest_point_to_R_Kruising / R_Facet_length * 100
R_Facet_percentage_after_closest_point_to_R_Kruising = R_Facet_length_from_closest_point_to_R_Kruising / R_Facet_length * 100

print(f"Percentage of the L Facet curve before the closest point to 'L_Kruising': {L_Facet_percentage_before_closest_point_to_L_Kruising:.2f}%")
print(f"Percentage of the L Facet curve after the closest point to 'L_Kruising': {L_Facet_percentage_after_closest_point_to_L_Kruising:.2f}%")

print(f"Percentage of the R Facet curve before the closest point to 'R_Kruising': {R_Facet_percentage_before_closest_point_to_R_Kruising:.2f}%")
print(f"Percentage of the R Facet curve after the closest point to 'R_Kruising': {R_Facet_percentage_after_closest_point_to_R_Kruising:.2f}%")


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points for set A
ax.scatter(x_A, y_A, z_A, c='grey', marker='o', label='Nerve')
# Plot the points for set B
ax.scatter(x_B, y_B, z_B, c='grey', marker='s', label='Facet')
# Plot the points for set C
ax.scatter(x_C, y_C, z_C, c='grey', marker='o', label='Nerve')
# Plot the points for set D
ax.scatter(x_D, y_D, z_D, c='grey', marker='s', label='Facet')


ax.scatter(L_Kruising[0][0], L_Kruising[0][1], L_Kruising[0][2], c='r', marker='x', label='L_Kruising')
ax.scatter(L_IVF[0][0], L_IVF[0][1], L_IVF[0][2], c='b', marker='d', label='L_IVF')

ax.scatter(R_Kruising[0][0], R_Kruising[0][1], R_Kruising[0][2], c='r', marker='x', label='L_Kruising')
ax.scatter(R_IVF[0][0], R_IVF[0][1], R_IVF[0][2], c='b', marker='d', label='R_IVF')

# Plot the smooth curves
ax.plot(x_new_A, y_new_A, z_new_A, c='m', label='L Nerve Curve')
ax.plot(x_new_B, y_new_B, z_new_B, c='g', label='L Facet Curve')

ax.plot(x_new_C, y_new_C, z_new_C, c='r', label='R Nerve Curve')
ax.plot(x_new_D, y_new_D, z_new_D, c='g', label='R Facet Curve')

# Plot the point where the two curves are closest on each curve
ax.scatter(closest_point_A[0], closest_point_A[1], closest_point_A[2], c='orange', marker='x', s=100)
ax.scatter(closest_point_B[0], closest_point_B[1], closest_point_B[2], c='orange', marker='x', s=100)

ax.scatter(closest_point_C[0], closest_point_C[1], closest_point_C[2], c='orange', marker='x', s=100)
ax.scatter(closest_point_D[0], closest_point_D[1], closest_point_D[2], c='orange', marker='x', s=100)



# Plot tangent vectors at the closest points to 'L_Kruising'
ax.quiver(*closest_point_A_to_L_Kruising, *tangent_vector_A_to_L_Kruising, color='b', label="Tangent Vector L NOT to 'L_Kruising'", pivot='tail', linewidth=1, length=2, arrow_length_ratio=0.1, linestyle='dashed')
ax.quiver(*closest_point_B_to_L_Kruising, *tangent_vector_B_to_L_Kruising, color='b', label="Tangent Vector L Facet to 'L_Kruising'", pivot='tail', linewidth=1, length=1, arrow_length_ratio=0.1, linestyle='dashed')

ax.quiver(*closest_point_C_to_R_Kruising, *tangent_vector_C_to_R_Kruising, color='r', label="Tangent Vector R NOT to 'R_Kruising'", pivot='tail', linewidth=1, length=2, arrow_length_ratio=0.1, linestyle='dashed')
ax.quiver(*closest_point_D_to_R_Kruising, *tangent_vector_D_to_R_Kruising, color='r', label="Tangent Vector R Facet to 'R_Kruising'", pivot='tail', linewidth=1, length=1, arrow_length_ratio=0.1, linestyle='dashed')

ax.quiver(*closest_point_A_to_L_Kruising, *direction_vector_A_after, color='b', label="Direction Vector L NOT (after)", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)
ax.quiver(*closest_point_B_to_L_Kruising, *direction_vector_B_after, color='b', label="Direction Vector L Facet (after)", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)
ax.quiver(*closest_point_C_to_R_Kruising, *direction_vector_C_after, color='r', label="Direction Vector R NOT (after)", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)
ax.quiver(*closest_point_D_to_R_Kruising, *direction_vector_D_after, color='r', label="Direction Vector L NOT (after)", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)

ax.quiver(*closest_point_A_to_L_Kruising, *direction_vector_A_before, color='grey', label="Direction Vector L NOT (before)", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)
ax.quiver(*closest_point_B_to_L_Kruising, *direction_vector_B_before, color='grey', label="Direction Vector L Facet (before)", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)
ax.quiver(*closest_point_C_to_R_Kruising, *direction_vector_C_before, color='grey', label="Direction Vector R NOT (before)", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)
ax.quiver(*closest_point_D_to_R_Kruising, *direction_vector_D_before, color='grey', label="Direction Vector R NOT (before)", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)


# Plot the closest points on set A and set B to 'L_Kruising'
ax.scatter(closest_point_A_to_L_Kruising[0], closest_point_A_to_L_Kruising[1], closest_point_A_to_L_Kruising[2], c='k', marker='x', s=100, label="Closest Point on L NOT to 'L_Kruising'")
ax.scatter(closest_point_B_to_L_Kruising[0], closest_point_B_to_L_Kruising[1], closest_point_B_to_L_Kruising[2], c='k', marker='x', s=100, label="Closest Point on L Facet to 'L_Kruising'")

ax.scatter(closest_point_C_to_R_Kruising[0], closest_point_C_to_R_Kruising[1], closest_point_C_to_R_Kruising[2], c='k', marker='x', s=100, label="Closest Point on R NOT to 'R_Kruising'")
ax.scatter(closest_point_D_to_R_Kruising[0], closest_point_D_to_R_Kruising[1], closest_point_D_to_R_Kruising[2], c='k', marker='x', s=100, label="Closest Point on R Facet to 'R_Kruising'")


# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Add a legend
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Show the plot with updated points
plt.show()
