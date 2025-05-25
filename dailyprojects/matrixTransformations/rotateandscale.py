import numpy as np
import matplotlib.pyplot as plt


def transform_points(points):
    """
    Applies rotation by 45 degrees followed by non-uniform scaling

    Args:
        points: List of 2D points [[x1,y1], [x2,y2], ...]

    Returns:
        transformed_points: The transformed points
        composite_matrix: The single transformation matrix that combines both operations
    """
    # Convert points to numpy array for easier manipulation
    points = np.array(points)

    # Step 1: Create rotation matrix for 45 degrees
    # Convert 45 degrees to radians
    theta = np.radians(45)

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    print("Rotation matrix (45 degrees):")
    print(R)
    print()

    # Step 2: Create scaling matrix
    # Scale by 2 in x-direction, 0.5 in y-direction
    S = np.array([
        [2.0, 0.0],
        [0.0, 0.5]
    ])

    print("Scaling matrix:")
    print(S)
    print()

    # Step 3: Combine transformations
    # Order matters! We first rotate, then scale
    # So the composite matrix is: S × R
    composite_matrix = S @ R

    print("Composite matrix (Scale × Rotate):")
    print(composite_matrix)
    print()

    # Step 4: Apply transformation to points
    # For each point, we need to apply: composite_matrix @ point
    # We can do this efficiently by transposing
    transformed_points = (composite_matrix @ points.T).T

    return transformed_points, composite_matrix


def visualize_transformation(original_points, transformed_points, title="Transformation"):
    """
    Visualizes the original and transformed points
    """
    original_points = np.array(original_points)
    transformed_points = np.array(transformed_points)

    plt.figure(figsize=(10, 5))

    # Plot original points
    plt.subplot(1, 2, 1)
    plt.scatter(original_points[:, 0], original_points[:, 1], c='blue', s=100)
    for i, point in enumerate(original_points):
        plt.annotate(f'P{i}', (point[0], point[1]), xytext=(5, 5), textcoords='offset points')

    # Connect points to show shape
    points_closed = np.vstack([original_points, original_points[0]])
    plt.plot(points_closed[:, 0], points_closed[:, 1], 'b-', alpha=0.5)

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title('Original Points')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot transformed points
    plt.subplot(1, 2, 2)
    plt.scatter(transformed_points[:, 0], transformed_points[:, 1], c='red', s=100)
    for i, point in enumerate(transformed_points):
        plt.annotate(f'P{i}\'', (point[0], point[1]), xytext=(5, 5), textcoords='offset points')

    # Connect points to show shape
    points_closed = np.vstack([transformed_points, transformed_points[0]])
    plt.plot(points_closed[:, 0], points_closed[:, 1], 'r-', alpha=0.5)

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title('Transformed Points')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def verify_step_by_step(points):
    """
    Verifies the transformation by applying it step by step
    """
    points = np.array(points)

    print("=== Step-by-step verification ===\n")
    print("Original points:")
    print(points)
    print()

    # Step 1: Apply rotation
    theta = np.radians(45)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    rotated_points = (R @ points.T).T
    print("After rotation by 45°:")
    print(rotated_points)
    print()

    # Step 2: Apply scaling
    S = np.array([
        [2.0, 0.0],
        [0.0, 0.5]
    ])

    scaled_points = (S @ rotated_points.T).T
    print("After scaling (2x in X, 0.5x in Y):")
    print(scaled_points)
    print()

    return scaled_points


# Main execution
if __name__ == "__main__":
    # Given points
    points = [[1, 0], [0, 1], [-1, 0], [0, -1]]

    print("=== Problem 1.1 Solution ===\n")

    # Apply transformation
    transformed_points, composite_matrix = transform_points(points)

    print("Original points:")
    for i, p in enumerate(points):
        print(f"  P{i}: {p}")
    print()

    print("Transformed points:")
    for i, p in enumerate(transformed_points):
        print(f"  P{i}': [{p[0]:.4f}, {p[1]:.4f}]")
    print()

    # Verify step by step
    verified_points = verify_step_by_step(points)

    # Check if results match
    print("Verification: Results match?", np.allclose(transformed_points, verified_points))

    # Visualize
    visualize_transformation(points, transformed_points,
                             "Rotation (45°) + Non-uniform Scaling (2x, 0.5y)")

    # Additional analysis
    print("\n=== Additional Analysis ===")
    print("\nComposite transformation matrix:")
    print(composite_matrix)
    print("\nThis matrix combines:")
    print("1. Rotation by 45° (counter-clockwise)")
    print("2. Scaling by 2 in X-direction")
    print("3. Scaling by 0.5 in Y-direction")

    # Show what happens with different ordering
    print("\n=== Effect of Order ===")

    # If we scale first, then rotate
    theta = np.radians(45)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    S = np.array([
        [2.0, 0.0],
        [0.0, 0.5]
    ])

    # Different order: R × S
    alternate_composite = R @ S
    print("\nIf we rotate AFTER scaling (R × S):")
    print(alternate_composite)

    alternate_transformed = (alternate_composite @ np.array(points).T).T
    print("\nResults would be different:")
    for i, p in enumerate(alternate_transformed):
        print(f"  P{i}': [{p[0]:.4f}, {p[1]:.4f}]")

    print("\nThis demonstrates that matrix multiplication order matters!")