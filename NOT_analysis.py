import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
from numpy.linalg import norm
import pandas as pd
import os

# Read the data from the Excel file
df = pd.read_excel('DATA_Microscribe/NOT 194(2).xlsx')

# Extract coordinates for various labels
Tubpost_C1 = df.loc[df['label'] == 'Tubpost_C1', ['x1', 'y1', 'z1']].values
PS_C2 = df.loc[df['label'] == 'PS_C2', ['x1', 'y1', 'z1']].values
PS_C3 = df.loc[df['label'] == 'PS_C3', ['x1', 'y1', 'z1']].values
PS_C4 = df.loc[df['label'] == 'PS_C4', ['x1', 'y1', 'z1']].values

L_IVF = df.loc[df['label'] == 'L_IVF', ['x1', 'y1', 'z1']].values
L_dors_ramus_mid = df.loc[df['label'] == 'L_dors_ramus_mid', ['x1', 'y1', 'z1']].values
L_Split = df.loc[df['label'] == 'L_Split', ['x1', 'y1', 'z1']].values
L_half_NOT = df.loc[df['label'] == 'L_half_NOT', ['x1', 'y1', 'z1']].values
L_bocht_NOT = df.loc[df['label'] == 'L_bocht_NOT', ['x1', 'y1', 'z1']].values
L_Facet_MED = df.loc[df['label'] == 'L_Facet_MED', ['x1', 'y1', 'z1']].values
L_Facet_Mid = df.loc[df['label'] == 'L_Facet_Mid', ['x1', 'y1', 'z1']].values
L_Facet_POST = df.loc[df['label'] == 'L_Facet_POST', ['x1', 'y1', 'z1']].values
L_Kruising = df.loc[df['label'] == 'L_Kruising', ['x1', 'y1', 'z1']].values

R_IVF = df.loc[df['label'] == 'R_IVF', ['x1', 'y1', 'z1']].values
R_dors_ramus_mid = df.loc[df['label'] == 'R_dors_ramus_mid', ['x1', 'y1', 'z1']].values
R_Split = df.loc[df['label'] == 'R_Split', ['x1', 'y1', 'z1']].values
R_half_NOT = df.loc[df['label'] == 'R_half_NOT', ['x1', 'y1', 'z1']].values
R_Facet_MED = df.loc[df['label'] == 'R_Facet_MED', ['x1', 'y1', 'z1']].values
R_Facet_Mid = df.loc[df['label'] == 'R_Facet_Mid', ['x1', 'y1', 'z1']].values
R_Facet_POST = df.loc[df['label'] == 'R_Facet_POST', ['x1', 'y1', 'z1']].values
R_Kruising = df.loc[df['label'] == 'R_Kruising', ['x1', 'y1', 'z1']].values

# Define coordinates for sets A, B, C, and D
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
}

coordinates_R_Facet = {
    "D_1": R_Facet_MED[0],
    "D_2": R_Facet_Mid[0],
    "D_3": R_Facet_POST[0],
}

# Extract the x, y, and z values from the coordinates for set A (L_NOT)
x_A = [coord[0] for coord in coordinates_L_NOT.values()] 
y_A = [coord[1] for coord in coordinates_L_NOT.values()]
z_A = [coord[2] for coord in coordinates_L_NOT.values()]

# Extract the x, y, and z values from the coordinates for set B (L_Facet)
x_B = [coord[0] for coord in coordinates_L_Facet.values()]
y_B = [coord[1] for coord in coordinates_L_Facet.values()]
z_B = [coord[2] for coord in coordinates_L_Facet.values()]

# Extract the x, y, and z values from the coordinates for set C (R_NOT)
x_C = [coord[0] for coord in coordinates_R_NOT.values()]
y_C = [coord[1] for coord in coordinates_R_NOT.values()]
z_C = [coord[2] for coord in coordinates_R_NOT.values()]

# Extract the x, y, and z values from the coordinates for set D (R_Facet)
x_D = [coord[0] for coord in coordinates_R_Facet.values()]
y_D = [coord[1] for coord in coordinates_R_Facet.values()]
z_D = [coord[2] for coord in coordinates_R_Facet.values()]


# Add virtual data points for set A (L_NOT)
t = np.linspace(0, 1, 30)  # additional points
x_new_A, y_new_A, z_new_A = splev(t, splprep([x_A, y_A, z_A], s=0.0, k=2)[0])

# Add virtual data points for set B (L_Facet)
x_new_B, y_new_B, z_new_B = splev(t, splprep([x_B, y_B, z_B], s=0.0, k=2)[0])

# Add virtual data points for set C (R_NOT)
x_new_C, y_new_C, z_new_C = splev(t, splprep([x_C, y_C, z_C], s=0.0, k=2)[0])

# Add virtual data points for set D (R_Facet)
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
            tangent_vector_A = np.array([x_new_A[i + 1] - x_new_A[i], y_new_A[i + 1] - y_new_A[i], z_new_A[i + 1] - z_new_A[i]])
            tangent_vector_B = np.array([x_new_B[j + 1] - x_new_B[j], y_new_B[j + 1] - y_new_B[j], z_new_B[j + 1] - z_new_B[j]])

for i in range(len(x_new_C) - 1):
    for j in range(len(x_new_D) - 1):
        current_distance_R = distance.euclidean((x_new_C[i], y_new_C[i], z_new_C[i]), (x_new_D[j], y_new_D[j], z_new_D[j]))
        if min_distance_R is None or current_distance_R < min_distance_R:
            min_distance_R = current_distance_R
            closest_point_C = (x_new_C[i], y_new_C[i], z_new_C[i])
            closest_point_D = (x_new_D[j], y_new_D[j], z_new_D[j])
            tangent_vector_C = np.array([x_new_C[i + 1] - x_new_C[i], y_new_C[i + 1] - y_new_C[i], z_new_C[i + 1] - z_new_C[i]])
            tangent_vector_D = np.array([x_new_D[j + 1] - x_new_D[j], y_new_D[j + 1] - y_new_D[j], z_new_D[j + 1] - z_new_D[j]])


# Calculate the angle between the two tangent vectors
angle_rad_L = np.arccos(np.dot(tangent_vector_A, tangent_vector_B) / (norm(tangent_vector_A) * norm(tangent_vector_B)))
angle_deg_L = np.degrees(angle_rad_L)

angle_rad_R = np.arccos(np.dot(tangent_vector_C, tangent_vector_D) / (norm(tangent_vector_C) * norm(tangent_vector_D)))
angle_deg_R = np.degrees(angle_rad_R)

# Print the angle between the two curves
print(f"Angle between the two curves at the closest points L: {angle_deg_L:.2f} degrees")
print(f"Angle between the two curves at the closest points R: {angle_deg_R:.2f} degrees")

# Coordinates for 'L_Kruising'
L_Kruising_coords = L_Kruising[0]
R_Kruising_coords = R_Kruising[0]

print(f"L_Kruising coordinates: {L_Kruising_coords}")

# Find the closest points on curve to 'L_Kruising' and 'R_Kruising'
closest_point_A_to_L_Kruising = None
closest_point_B_to_L_Kruising = None
closest_point_C_to_R_Kruising = None
closest_point_D_to_R_Kruising = None

min_distance_A_to_L_Kruising = None
min_distance_B_to_L_Kruising = None
min_distance_C_to_R_Kruising = None
min_distance_D_to_R_Kruising = None

# L_NOT
num_points_before_A = int(0.1 * len(t))
num_points_after_A = int(0.2 * len(t))
# num_points_before_A = 20
# num_points_after_A = 20
# L_Facet
num_points_before_B = int(0.1 * len(t))
num_points_after_B = int(0.2 * len(t))
# num_points_before_B = 20
# num_points_after_B = 20
# R_NOT
num_points_before_C = int(0.1 * len(t))
num_points_after_C = int(0.2 * len(t))
# R_Facet
num_points_before_D = int(0.1 * len(t))
num_points_after_D = int(0.2 * len(t))

max_index_A = len(x_new_A) - 1
max_index_B = len(x_new_B) - 1
max_index_C = len(x_new_C) - 1
max_index_D = len(x_new_D) - 1


if L_Kruising_coords is not None and not np.isnan(L_Kruising_coords).any():
    for i in range(len(x_new_A)):
        current_distance_A_to_L_Kruising = distance.euclidean((x_new_A[i], y_new_A[i], z_new_A[i]), L_Kruising_coords)
        if min_distance_A_to_L_Kruising is None or current_distance_A_to_L_Kruising < min_distance_A_to_L_Kruising:
            min_distance_A_to_L_Kruising = current_distance_A_to_L_Kruising
            closest_point_A_to_L_Kruising = (x_new_A[i], y_new_A[i], z_new_A[i])
            # Tangent vector for L_NOT at the closest point to 'L_Kruising'
            # tangent_vector_A_to_L_Kruising = np.array([x_new_A[i + 1] - x_new_A[i], y_new_A[i + 1] - y_new_A[i], z_new_A[i + 1] - z_new_A[i]])
            # Tangent vector for L_NOT at the closest point to 'L_Kruising' with more points
            tangent_vector_A_to_L_Kruising = np.array([x_new_A[min(max_index_A, i + num_points_after_A)] - x_new_A[max(0, i - num_points_before_A)],
                                                        y_new_A[min(max_index_A, i + num_points_after_A)] - y_new_A[max(0, i - num_points_before_A)],
                                                        z_new_A[min(max_index_A, i + num_points_after_A)] - z_new_A[max(0, i - num_points_before_A)]])
            tangent_vector_A_to_L_Kruising = tangent_vector_A_to_L_Kruising / np.linalg.norm(tangent_vector_A_to_L_Kruising)

if L_Kruising_coords is not None and not np.isnan(L_Kruising_coords).any():
    for j in range(len(x_new_B)):
        current_distance_B_to_L_Kruising = distance.euclidean((x_new_B[j], y_new_B[j], z_new_B[j]), L_Kruising_coords)
        if min_distance_B_to_L_Kruising is None or current_distance_B_to_L_Kruising < min_distance_B_to_L_Kruising:
            min_distance_B_to_L_Kruising = current_distance_B_to_L_Kruising
            closest_point_B_to_L_Kruising = (x_new_B[j], y_new_B[j], z_new_B[j])
            # Tangent vector for L_Facet at the closest point to 'L_Kruising'
            # tangent_vector_B_to_L_Kruising = np.array([x_new_B[j + 1] - x_new_B[j], y_new_B[j + 1] - y_new_B[j], z_new_B[j + 1] - z_new_B[j]])

            # Tangent vector for L_Facet at the closest point to 'L_Kruising' with more points
            tangent_vector_B_to_L_Kruising = np.array([x_new_B[min(max_index_B, j + num_points_after_B)] - x_new_B[max(0, j - num_points_before_B)],
                                                        y_new_B[min(max_index_B, j + num_points_after_B)] - y_new_B[max(0, j - num_points_before_B)],
                                                        z_new_B[min(max_index_B, j + num_points_after_B)] - z_new_B[max(0, j - num_points_before_B)]])
            tangent_vector_B_to_L_Kruising = tangent_vector_B_to_L_Kruising / np.linalg.norm(tangent_vector_B_to_L_Kruising)


if R_Kruising_coords is not None and not np.isnan(R_Kruising_coords).any():
    for i in range(len(x_new_C)):
        current_distance_C_to_R_Kruising = distance.euclidean((x_new_C[i], y_new_C[i], z_new_C[i]), R_Kruising_coords)
        if min_distance_C_to_R_Kruising is None or current_distance_C_to_R_Kruising < min_distance_C_to_R_Kruising:
            min_distance_C_to_R_Kruising = current_distance_C_to_R_Kruising
            closest_point_C_to_R_Kruising = (x_new_C[i], y_new_C[i], z_new_C[i])
            # Tangent vector for R_NOT at the closest point to 'R_Kruising'
            # tangent_vector_C_to_R_Kruising = np.array([x_new_C[i + 1] - x_new_C[i], y_new_C[i + 1] - y_new_C[i], z_new_C[i + 1] - z_new_C[i]])

            # Tangent vector for R_NOT at the closest point to 'R_Kruising' with more points
            tangent_vector_C_to_R_Kruising = np.array([x_new_C[min(max_index_C, i + num_points_after_C)] - x_new_C[max(0, i - num_points_before_C)],
                                                        y_new_C[min(max_index_C, i + num_points_after_C)] - y_new_C[max(0, i - num_points_before_C)],
                                                        z_new_C[min(max_index_C, i + num_points_after_C)] - z_new_C[max(0, i - num_points_before_C)]])
            tangent_vector_C_to_R_Kruising = tangent_vector_C_to_R_Kruising / np.linalg.norm(tangent_vector_C_to_R_Kruising)


if R_Kruising_coords is not None and not np.isnan(R_Kruising_coords).any():
    for j in range(len(x_new_D)):
        current_distance_D_to_R_Kruising = distance.euclidean((x_new_D[j], y_new_D[j], z_new_D[j]), R_Kruising_coords)
        if min_distance_D_to_R_Kruising is None or current_distance_D_to_R_Kruising < min_distance_D_to_R_Kruising:
            min_distance_D_to_R_Kruising = current_distance_D_to_R_Kruising
            closest_point_D_to_R_Kruising = (x_new_D[j], y_new_D[j], z_new_D[j])
            # Tangent vector for R_Facet at the closest point to 'R_Kruising'
            # tangent_vector_D_to_R_Kruising = np.array([x_new_D[j + 1] - x_new_D[j], y_new_D[j + 1] - y_new_D[j], z_new_D[j + 1] - z_new_D[j]])

            # Tangent vector for R_Facet at the closest point to 'R_Kruising' with more points
            tangent_vector_D_to_R_Kruising = np.array([x_new_D[min(max_index_D, j + num_points_after_D)] - x_new_D[max(0, j - num_points_before_D)],
                                                        y_new_D[min(max_index_D, j + num_points_after_D)] - y_new_D[max(0, j - num_points_before_D)],
                                                        z_new_D[min(max_index_D, j + num_points_after_D)] - z_new_D[max(0, j - num_points_before_D)]])
            tangent_vector_D_to_R_Kruising = tangent_vector_D_to_R_Kruising / np.linalg.norm(tangent_vector_D_to_R_Kruising)

# Calculate the angle between the tangent vectors at the closest points to 'L_Kruising'
# Left side
if L_Kruising_coords is not None and not np.isnan(L_Kruising_coords).any(): 
    angle_rad_to_L_Kruising = np.arccos(np.dot(tangent_vector_A_to_L_Kruising, tangent_vector_B_to_L_Kruising) / (norm(tangent_vector_A_to_L_Kruising) * norm(tangent_vector_B_to_L_Kruising)))
    angle_deg_to_L_Kruising = np.degrees(angle_rad_to_L_Kruising)
else:
    angle_deg_to_L_Kruising = 0.0000

# Right side
if R_Kruising_coords is not None and not np.isnan(R_Kruising_coords).any():
    angle_rad_to_R_Kruising = np.arccos(np.dot(tangent_vector_C_to_R_Kruising, tangent_vector_D_to_R_Kruising) / (norm(tangent_vector_C_to_R_Kruising) * norm(tangent_vector_D_to_R_Kruising)))
    angle_deg_to_R_Kruising = np.degrees(angle_rad_to_R_Kruising)
else:
    angle_deg_to_R_Kruising = 0.0000

# Print the angle between the two curves at the closest points to 'L_Kruising'
print(f"Angle between the two curves at the closest points to 'L_Kruising': {angle_deg_to_L_Kruising:.2f} degrees")
print(f"Angle between the two curves at the closest points to 'R_Kruising': {angle_deg_to_R_Kruising:.2f} degrees")

# Create a plane using the coordinates for set B
B_1 = np.array(coordinates_L_Facet["B_1"])
B_2 = np.array(coordinates_L_Facet["B_2"])
B_3 = np.array(coordinates_L_Facet["B_3"])

# Calculate the normal vector of the plane using cross product
normal_vector = np.cross(B_2 - B_1, B_3 - B_1)
d = -np.dot(normal_vector, B_1)  # Calculate d parameter of the plane equation (Ax + By + Cz + d = 0)

# Create a grid for the plane B
num_points = 5 # Extend the range of the grid for the plane to make it larger

x_plane, y_plane = np.meshgrid(np.linspace(min(x_B) - 10, max(x_B) + 10, num_points), np.linspace(min(y_B) - 10, max(y_B) + 10, num_points))

# Adjust the size of the plane B
z_plane = (-normal_vector[0] * x_plane - normal_vector[1] * y_plane - d) / normal_vector[2]

# Calculate the angle between "Tangent Vector A to 'L_Kruising'" and the plane for set B

# Unit vector of "Tangent Vector A to 'L_Kruising'"
if L_Kruising_coords is not None and not np.isnan(L_Kruising_coords).any():

    unit_tangent_vector_A_to_L_Kruising = tangent_vector_A_to_L_Kruising / np.linalg.norm(tangent_vector_A_to_L_Kruising)

    # Unit normal vector of the plane defined by coordinates_L_Facet
    unit_normal_vector_of_plane = normal_vector / np.linalg.norm(normal_vector)

    # Calculate the cosine of the angle between the two vectors
    cosine_angle = np.dot(unit_tangent_vector_A_to_L_Kruising, unit_normal_vector_of_plane)

    # Calculate the angle in degrees
    angle_rad_to_normal_vector_of_plane = np.arccos(cosine_angle)
    angle_deg_to_normal_vector_of_plane = np.degrees(angle_rad_to_normal_vector_of_plane)
    angle_deg_to_plane = 90 - angle_deg_to_normal_vector_of_plane

else:
    angle_deg_to_plane = 0.0000

print(f"Angle between 'Tangent Vector A to 'L_Kruising'' and the plane L (B-cyan): {angle_deg_to_plane:.2f} degrees")

# Create a plane using the coordinates for set D
D_1 = np.array(coordinates_R_Facet["D_1"])
D_2 = np.array(coordinates_R_Facet["D_2"])
D_3 = np.array(coordinates_R_Facet["D_3"])

# Calculate the normal vector of the plane using cross product
normal_vector_D = -np.cross(D_2 - D_1, D_3 - D_1)
d_D = -np.dot(normal_vector_D, D_1)  # Calculate d parameter of the plane equation (Ax + By + Cz + d = 0)

# Create a grid for the plane
x_plane_D, y_plane_D = np.meshgrid(np.linspace(min(x_D) - 10, max(x_D) + 10, num_points), np.linspace(min(y_D) - 10, max(y_D) + 10, num_points))

# Adjust the size of the plane
z_plane_D = (-normal_vector_D[0] * x_plane_D - normal_vector_D[1] * y_plane_D - d_D) / normal_vector_D[2]

# Calculate the angle between "Tangent Vector C to 'R_Kruising'" and the plane for set D
if R_Kruising_coords is not None and not np.isnan(R_Kruising_coords).any():
    unit_tangent_vector_C_to_R_Kruising = tangent_vector_C_to_R_Kruising / np.linalg.norm(tangent_vector_C_to_R_Kruising)
    unit_normal_vector_of_plane_D = normal_vector_D / np.linalg.norm(normal_vector_D)
    cosine_angle_D = np.dot(unit_tangent_vector_C_to_R_Kruising, unit_normal_vector_of_plane_D)
    angle_rad_to_normal_vector_of_plane_D = np.arccos(cosine_angle_D)
    angle_deg_to_normal_vector_of_plane_D = np.degrees(angle_rad_to_normal_vector_of_plane_D)
    angle_deg_to_plane_D = 90 - angle_deg_to_normal_vector_of_plane_D
else:
    angle_deg_to_plane_D = 0.0000


print(f"Angle between 'Tangent Vector C to 'R_Kruising'' and the plane R (D-purple): {angle_deg_to_plane_D:.2f} degrees")


# Calculate the distance between Facet and NOT at the closest points to 'L_Kruising'
if closest_point_A_to_L_Kruising is not None:
    L_distance_facet_to_NOT = distance.euclidean(closest_point_A_to_L_Kruising, closest_point_B_to_L_Kruising)
else:
    L_distance_facet_to_NOT = 0.0000

if closest_point_C_to_R_Kruising is not None:
    R_distance_facet_to_NOT = distance.euclidean(closest_point_C_to_R_Kruising, closest_point_D_to_R_Kruising)
else:
    R_distance_facet_to_NOT = 0.0000

# Print the distance between Facet and NOT at the closest points to 'L_Kruising'
print(f"Distance between Facet and NOT at the closest points to 'L_Kruising': {L_distance_facet_to_NOT:.2f} mm")
print(f"Distance between Facet and NOT at the closest points to 'R_Kruising': {R_distance_facet_to_NOT:.2f} mm")

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


# Print values without description
print(angle_deg_L)
print(angle_deg_to_L_Kruising)
print(angle_deg_to_plane)
print(L_distance_facet_to_NOT)
print(L_Facet_length)
print(L_Facet_length_from_closest_point_to_L_Kruising)
print(L_Facet_length_from_start_to_closest_point_to_L_Kruising)
print(L_Facet_percentage_before_closest_point_to_L_Kruising)
print(L_Facet_percentage_after_closest_point_to_L_Kruising)

print(angle_deg_R)
print(angle_deg_to_R_Kruising)
print(angle_deg_to_plane_D)
print(R_distance_facet_to_NOT)
print(R_Facet_length)
print(R_Facet_length_from_closest_point_to_R_Kruising)
print(R_Facet_length_from_start_to_closest_point_to_R_Kruising)
print(R_Facet_percentage_before_closest_point_to_R_Kruising)
print(R_Facet_percentage_after_closest_point_to_R_Kruising)

# Create Left plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points for set A
ax.scatter(x_A, y_A, z_A, c='b', marker='o', label='Nerve')

# Plot the points for set B
ax.scatter(x_B, y_B, z_B, c='g', marker='s', label='Facet')

ax.scatter(L_Kruising[0][0], L_Kruising[0][1], L_Kruising[0][2], c='r', marker='x', label='L_Kruising')
ax.scatter(L_IVF[0][0], L_IVF[0][1], L_IVF[0][2], c='b', marker='d', label='L_IVF')

# Plot the smooth curves
ax.plot(x_new_A, y_new_A, z_new_A, c='r', label='L Nerve Curve', linewidth=2)
ax.plot(x_new_B, y_new_B, z_new_B, c='m', label='L Facet Curve', linewidth=2)

# # Plot the point where the two curves are closest on each curve
# ax.scatter(closest_point_A[0], closest_point_A[1], closest_point_A[2], c='orange', marker='x', s=20, label='Closest Point on L Nerve')
# ax.scatter(closest_point_B[0], closest_point_B[1], closest_point_B[2], c='orange', marker='x', s=20, label='Closest Point on L Facet')

# # Plot tangent vectors at closest points
# ax.quiver(*closest_point_A, *tangent_vector_A, color='c', label='Tangent Vector A', pivot='tail', linewidth=1, length=4, arrow_length_ratio=0.1)
# ax.quiver(*closest_point_B, *tangent_vector_B, color='y', label='Tangent Vector B', pivot='tail', linewidth=1, length=2, arrow_length_ratio=0.1)

# ax.quiver(*closest_point_C, *tangent_vector_C, color='c', label='Tangent Vector C', pivot='tail', linewidth=1, length=4, arrow_length_ratio=0.1)
# ax.quiver(*closest_point_D, *tangent_vector_D, color='y', label='Tangent Vector D', pivot='tail', linewidth=1, length=2, arrow_length_ratio=0.1)

# Plot tangent vectors at the closest points to 'L_Kruising'
if L_Kruising_coords is not None and not np.isnan(L_Kruising_coords).any():
    ax.quiver(*closest_point_A_to_L_Kruising, *tangent_vector_A_to_L_Kruising, color='b', label="Tangent Vector A to 'L_Kruising'", pivot='tail', linewidth=1, length=40, arrow_length_ratio=0.1)
    ax.quiver(*closest_point_B_to_L_Kruising, *tangent_vector_B_to_L_Kruising, color='g', label="Tangent Vector B to 'L_Kruising'", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)
    ax.scatter(closest_point_A_to_L_Kruising[0], closest_point_A_to_L_Kruising[1], closest_point_A_to_L_Kruising[2], c='k', marker='x', s=50, label="Closest Point on A to 'L_Kruising'")
    ax.scatter(closest_point_B_to_L_Kruising[0], closest_point_B_to_L_Kruising[1], closest_point_B_to_L_Kruising[2], c='k', marker='x', s=50, label="Closest Point on B to 'L_Kruising'")

# plot the normal vector of the planes
# ax.quiver(*B_1, *normal_vector, color='c', label='Normal Vector of Plane', pivot='tail', linewidth=1, length=10, arrow_length_ratio=0.1)

# Plot the closest points on set A and set B to 'L_Kruising'

# Add the planes to the 3D plot
ax.plot_surface(x_plane, y_plane, z_plane, color='cyan', alpha=0.1)

ax.scatter(Tubpost_C1[0][0], Tubpost_C1[0][1], Tubpost_C1[0][2], c='g', marker='d', label='Tubpost_C1')
ax.scatter(PS_C2[0][0], PS_C2[0][1], PS_C2[0][2], c='b', marker='d', label='PS_C2')
ax.scatter(PS_C3[0][0], PS_C3[0][1], PS_C3[0][2], c='b', marker='d', label='PS_C3')
ax.scatter(PS_C4[0][0], PS_C4[0][1], PS_C4[0][2], c='b', marker='d', label='PS_C4')

# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # Add a custom legend for the plane
# custom_legend = ax.legend(['L Facet Plane'], loc='center left', bbox_to_anchor=(1, 0.5))

# Show the plot with the updated legend
plt.show()

# Create Right plot
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points for set C
ax.scatter(x_C, y_C, z_C, c='b', marker='o', label='Nerve')

# Plot the points for set D
ax.scatter(x_D, y_D, z_D, c='g', marker='s', label='Facet')

ax.scatter(R_Kruising[0][0], R_Kruising[0][1], R_Kruising[0][2], c='r', marker='x', label='L_Kruising')
ax.scatter(R_IVF[0][0], R_IVF[0][1], R_IVF[0][2], c='b', marker='d', label='R_IVF')

# Plot the smooth curves
ax.plot(x_new_C, y_new_C, z_new_C, c='r', label='R Nerve Curve', linewidth=2)
ax.plot(x_new_D, y_new_D, z_new_D, c='m', label='R Facet Curve', linewidth=2)

# # Plot the point where the two curves are closest on each curve

# ax.scatter(closest_point_C[0], closest_point_C[1], closest_point_C[2], c='orange', marker='x', s=20)
# ax.scatter(closest_point_D[0], closest_point_D[1], closest_point_D[2], c='orange', marker='x', s=20)

# # Plot tangent vectors at closest points
# ax.quiver(*closest_point_A, *tangent_vector_A, color='c', label='Tangent Vector A', pivot='tail', linewidth=1, length=4, arrow_length_ratio=0.1)
# ax.quiver(*closest_point_B, *tangent_vector_B, color='y', label='Tangent Vector B', pivot='tail', linewidth=1, length=2, arrow_length_ratio=0.1)

# ax.quiver(*closest_point_C, *tangent_vector_C, color='c', label='Tangent Vector C', pivot='tail', linewidth=1, length=4, arrow_length_ratio=0.1)
# ax.quiver(*closest_point_D, *tangent_vector_D, color='y', label='Tangent Vector D', pivot='tail', linewidth=1, length=2, arrow_length_ratio=0.1)

# Plot tangent vectors at the closest points to 'R Kruising'
if R_Kruising_coords is not None and not np.isnan(R_Kruising_coords).any():
    ax.quiver(*closest_point_C_to_R_Kruising, *tangent_vector_C_to_R_Kruising, color='b', label="Tangent Vector C to 'R_Kruising'", pivot='tail', linewidth=1, length=40, arrow_length_ratio=0.1)
    ax.quiver(*closest_point_D_to_R_Kruising, *tangent_vector_D_to_R_Kruising, color='g', label="Tangent Vector D to 'R_Kruising'", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)
    # Plot the closest points on set A and set B to 'R_Kruising'
    ax.scatter(closest_point_C_to_R_Kruising[0], closest_point_C_to_R_Kruising[1], closest_point_C_to_R_Kruising[2], c='k', marker='x', s=50, label="Closest Point on C to 'R_Kruising'")
    ax.scatter(closest_point_D_to_R_Kruising[0], closest_point_D_to_R_Kruising[1], closest_point_D_to_R_Kruising[2], c='k', marker='x', s=50, label="Closest Point on D to 'R_Kruising'")


# # plot the normal vector of the planes
# ax.quiver(*D_1, *normal_vector_D, color='c', label='Normal Vector of Plane', pivot='tail', linewidth=1, length=10, arrow_length_ratio=0.1)



# Add the planes to the 3D plot
ax.plot_surface(x_plane_D, y_plane_D, z_plane_D, color='purple', alpha=0.1)

ax.scatter(Tubpost_C1[0][0], Tubpost_C1[0][1], Tubpost_C1[0][2], c='g', marker='d', label='Tubpost_C1')
ax.scatter(PS_C2[0][0], PS_C2[0][1], PS_C2[0][2], c='b', marker='d', label='PS_C2')
ax.scatter(PS_C3[0][0], PS_C3[0][1], PS_C3[0][2], c='b', marker='d', label='PS_C3')
ax.scatter(PS_C4[0][0], PS_C4[0][1], PS_C4[0][2], c='b', marker='d', label='PS_C4')

# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # Add a custom legend for the plane
# custom_legend = ax.legend(['L Facet Plane'], loc='center left', bbox_to_anchor=(1, 0.5))

# Show the plot with the updated legend
plt.show()



# # 3D plot Left and Right
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the points for set A
# ax.scatter(x_A, y_A, z_A, c='b', marker='o', label='Nerve')

# # Plot the points for set B
# ax.scatter(x_B, y_B, z_B, c='g', marker='s', label='Facet')

# # Plot the points for set C
# ax.scatter(x_C, y_C, z_C, c='b', marker='o', label='Nerve')

# # Plot the points for set D
# ax.scatter(x_D, y_D, z_D, c='g', marker='s', label='Facet')


# ax.scatter(L_Kruising[0][0], L_Kruising[0][1], L_Kruising[0][2], c='r', marker='x', label='L_Kruising')
# ax.scatter(L_IVF[0][0], L_IVF[0][1], L_IVF[0][2], c='b', marker='d', label='L_IVF')

# ax.scatter(R_Kruising[0][0], R_Kruising[0][1], R_Kruising[0][2], c='r', marker='x', label='L_Kruising')
# ax.scatter(R_IVF[0][0], R_IVF[0][1], R_IVF[0][2], c='b', marker='d', label='R_IVF')

# # Plot the smooth curves
# ax.plot(x_new_A, y_new_A, z_new_A, c='r', label='L Nerve Curve', linewidth=2)
# ax.plot(x_new_B, y_new_B, z_new_B, c='m', label='L Facet Curve', linewidth=2)

# ax.plot(x_new_C, y_new_C, z_new_C, c='r', label='R Nerve Curve', linewidth=2)
# ax.plot(x_new_D, y_new_D, z_new_D, c='m', label='R Facet Curve', linewidth=2)

# # Plot the point where the two curves are closest on each curve
# ax.scatter(closest_point_A[0], closest_point_A[1], closest_point_A[2], c='orange', marker='x', s=20)
# ax.scatter(closest_point_B[0], closest_point_B[1], closest_point_B[2], c='orange', marker='x', s=20)

# ax.scatter(closest_point_C[0], closest_point_C[1], closest_point_C[2], c='orange', marker='x', s=20)
# ax.scatter(closest_point_D[0], closest_point_D[1], closest_point_D[2], c='orange', marker='x', s=20)

# # # Plot tangent vectors at closest points
# # ax.quiver(*closest_point_A, *tangent_vector_A, color='c', label='Tangent Vector A', pivot='tail', linewidth=1, length=4, arrow_length_ratio=0.1)
# # ax.quiver(*closest_point_B, *tangent_vector_B, color='y', label='Tangent Vector B', pivot='tail', linewidth=1, length=2, arrow_length_ratio=0.1)

# # ax.quiver(*closest_point_C, *tangent_vector_C, color='c', label='Tangent Vector C', pivot='tail', linewidth=1, length=4, arrow_length_ratio=0.1)
# # ax.quiver(*closest_point_D, *tangent_vector_D, color='y', label='Tangent Vector D', pivot='tail', linewidth=1, length=2, arrow_length_ratio=0.1)

# # Plot tangent vectors at the closest points to 'L_Kruising'
# ax.quiver(*closest_point_A_to_L_Kruising, *tangent_vector_A_to_L_Kruising, color='b', label="Tangent Vector A to 'L_Kruising'", pivot='tail', linewidth=1, length=40, arrow_length_ratio=0.1)
# ax.quiver(*closest_point_B_to_L_Kruising, *tangent_vector_B_to_L_Kruising, color='g', label="Tangent Vector B to 'L_Kruising'", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)

# ax.quiver(*closest_point_C_to_R_Kruising, *tangent_vector_C_to_R_Kruising, color='b', label="Tangent Vector C to 'R_Kruising'", pivot='tail', linewidth=1, length=40, arrow_length_ratio=0.1)
# ax.quiver(*closest_point_D_to_R_Kruising, *tangent_vector_D_to_R_Kruising, color='g', label="Tangent Vector D to 'R_Kruising'", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)

# # plot the normal vector of the planes
# ax.quiver(*B_1, *normal_vector, color='c', label='Normal Vector of Plane', pivot='tail', linewidth=1, length=10, arrow_length_ratio=0.1)
# ax.quiver(*D_1, *normal_vector_D, color='c', label='Normal Vector of Plane', pivot='tail', linewidth=1, length=10, arrow_length_ratio=0.1)

# # Plot the closest points on set A and set B to 'L_Kruising'
# ax.scatter(closest_point_A_to_L_Kruising[0], closest_point_A_to_L_Kruising[1], closest_point_A_to_L_Kruising[2], c='k', marker='x', s=50, label="Closest Point on A to 'L_Kruising'")
# ax.scatter(closest_point_B_to_L_Kruising[0], closest_point_B_to_L_Kruising[1], closest_point_B_to_L_Kruising[2], c='k', marker='x', s=50, label="Closest Point on B to 'L_Kruising'")

# ax.scatter(closest_point_C_to_R_Kruising[0], closest_point_C_to_R_Kruising[1], closest_point_C_to_R_Kruising[2], c='k', marker='x', s=50, label="Closest Point on C to 'R_Kruising'")
# ax.scatter(closest_point_D_to_R_Kruising[0], closest_point_D_to_R_Kruising[1], closest_point_D_to_R_Kruising[2], c='k', marker='x', s=50, label="Closest Point on D to 'R_Kruising'")

# # Add the planes to the 3D plot
# ax.plot_surface(x_plane, y_plane, z_plane, color='cyan', alpha=0.1)
# ax.plot_surface(x_plane_D, y_plane_D, z_plane_D, color='purple', alpha=0.1)

# ax.scatter(Tubpost_C1[0][0], Tubpost_C1[0][1], Tubpost_C1[0][2], c='g', marker='d', label='Tubpost_C1')
# ax.scatter(PS_C2[0][0], PS_C2[0][1], PS_C2[0][2], c='b', marker='d', label='PS_C2')
# ax.scatter(PS_C3[0][0], PS_C3[0][1], PS_C3[0][2], c='b', marker='d', label='PS_C3')
# ax.scatter(PS_C4[0][0], PS_C4[0][1], PS_C4[0][2], c='b', marker='d', label='PS_C4')

# # Set labels for the axes
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')

# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # Add a custom legend for the plane
# custom_legend = ax.legend(['L Facet Plane'], loc='center left', bbox_to_anchor=(1, 0.5))

# Show the plot with the updated legend
# plt.show()

## Plot for paper image

# Create Left plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points for set A
# ax.scatter(x_A, y_A, z_A, c='r', marker='o')

# Plot the points for set B
# ax.scatter(x_B, y_B, z_B, c='m', marker='s')

ax.scatter(L_Kruising[0][0], L_Kruising[0][1], L_Kruising[0][2], c='k', marker='x', s=60, label='Crossing Point')
ax.scatter(L_IVF[0][0], L_IVF[0][1], L_IVF[0][2], c='k', marker='d', s=40, label='IVF')
ax.scatter(L_Facet_MED[0][0], L_Facet_MED[0][1], L_Facet_MED[0][2], c='m', marker='s', label='L: Facet Lateral')
ax.scatter(L_Facet_POST[0][0], L_Facet_POST[0][1], L_Facet_POST[0][2], c='m', marker='s', label='P: Facet Posterior')

# Plot the smooth curves
ax.plot(x_new_A, y_new_A, z_new_A, c='r', label='Nerve (TON)', linewidth=2)
ax.plot(x_new_B, y_new_B, z_new_B, c='m', label='Facet (C2-C3)', linewidth=2)

# Plot tangent vectors at the closest points to 'L_Kruising'
if L_Kruising_coords is not None and not np.isnan(L_Kruising_coords).any():
    ax.quiver(*closest_point_A_to_L_Kruising, *tangent_vector_A_to_L_Kruising, color='b', label="Nerve Tangent vector at crossing", pivot='tail', linewidth=1, length=40, arrow_length_ratio=0.1)
    ax.quiver(*closest_point_B_to_L_Kruising, *tangent_vector_B_to_L_Kruising, color='g', label="Facet Tangent vector at crossing", pivot='tail', linewidth=1, length=20, arrow_length_ratio=0.1)
   # ax.scatter(closest_point_A_to_L_Kruising[0], closest_point_A_to_L_Kruising[1], closest_point_A_to_L_Kruising[2], c='k', marker='x', s=50, label="Closest Point on A to 'L_Kruising'")
   # ax.scatter(closest_point_B_to_L_Kruising[0], closest_point_B_to_L_Kruising[1], closest_point_B_to_L_Kruising[2], c='k', marker='x', s=50, label="Closest Point on B to 'L_Kruising'")

ax.scatter(Tubpost_C1[0][0], Tubpost_C1[0][1], Tubpost_C1[0][2], c='b', marker='d', s=35, label='Post Tub - C1')
ax.scatter(PS_C2[0][0], PS_C2[0][1], PS_C2[0][2], c='b', marker='o', label='PS - C2')
ax.scatter(PS_C3[0][0], PS_C3[0][1], PS_C3[0][2], c='b', marker='o', label='PS - C3')
ax.scatter(PS_C4[0][0], PS_C4[0][1], PS_C4[0][2], c='b', marker='o', label='PS - C4')

# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # Add a custom legend for the plane
# custom_legend = ax.legend(['L Facet Plane'], loc='center left', bbox_to_anchor=(1, 0.5))

# Show the plot with the updated legend
plt.show()