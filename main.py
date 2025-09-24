import cv2, random
import mediapipe as mp
import numpy as np
from math import degrees, acos

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def point_line_distance(pt, line_pt1, line_pt2):
        # pt: (x, y), line_pt1: (x1, y1), line_pt2: (x2, y2)
        x0, y0 = pt
        x1, y1 = line_pt1
        x2, y2 = line_pt2
        num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        return num / den if den != 0 else 0

def compute_beauty_score(ratios):
    deviations = []
    for _, (measured, ideal) in ratios.items():
        try:
            measured_val = float(measured)
            ideal_val = float(ideal)

            # Relative deviation (normalized)
            deviation = abs(measured_val - ideal_val) / ideal_val
            deviations.append(deviation)
        except ValueError:
            continue       # skip non-numeric ratios

    if not deviations:
        return 0
    
    rms = np.sqrt(np.mean(np.square(deviations)))
    score = max(0, 100 * (1 - rms))
    return round(score, 2)

def plot_on_image(annotated_img, points):
    # Color palette
    point_color = (255, 215, 0)
    arrow_thickness = 2
    point_radius = 3
    ring_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw and label key points as rings
    for _, (x, y) in points.items():
        cv2.circle(annotated_img, (x, y), point_radius, point_color, ring_thickness, lineType=cv2.LINE_AA)

    # To draw dotted (dashed) lines
    def draw_dotted_line(img, pt1, pt2, color, thickness=2, gap=5):
        dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
        pts = [
            (
                int(pt1[0] + (pt2[0] - pt1[0]) * i / dist),
                int(pt1[1] + (pt2[1] - pt1[1]) * i / dist)
            )
            for i in range(0, dist, gap)
        ]
        for i in range(len(pts) - 1):
            if i % 2 == 0:
                cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

    fx1, fy1 = points['forehead_top']
    fx2, fy2 = points['chin']
    draw_dotted_line(annotated_img, (fx1, fy1), (fx2, fy2), point_color, arrow_thickness)

    cv2.putText(
        annotated_img, "Face Length",
        (fx1 + 10, fy1 + 40), font, 0.5, point_color, 1, cv2.LINE_AA
    )
    
    wx1, wy1 = points['left_cheek']
    wx2, wy2 = points['right_cheek']
    draw_dotted_line(annotated_img, (wx1, wy1), (wx2, wy2), point_color, arrow_thickness)
    
    # Label for face width
    cv2.putText(
        annotated_img, "Face Width",
        (wx1 + 20, wy1 + 20), font, 0.5, point_color, 1, cv2.LINE_AA
    )
    return annotated_img
    
def comment_on(face_ratio):
    """
    Returns playful labels based on face ratio ranges
    Based on population distribution research with finer granularity
    """
    
    RATIO_LABELS = {
        # Ultra-elongated - Very rare
        (1.62, float('inf')): [
            "VerticalVortex", "SkyscraperStyle", "RectangleRoyalty", "ElongatedElegance",
            "GoldenGiraffe", "EtherealEllipse", "RegalRectangle", "GeometricGiant",
            "TallTitan", "NobleNarrow", "StatuesqueStar", "LankyLegend",
            "SvelteSculpture", "LoftyLuxe", "StatelyShape", "ElegantEchelon"
        ],
        
        # Near golden ratio - Target zone
        (1.32, 1.62): [
            "GreekGod", "PhiPerfect", "DivineDimension", "PhiPhantom",
            "GeometricGenius", "RatioRoyalty", "PerfectProportion",
            "SymmetrySupreme", "IdealIcon", "ClassicCrown", "TimelessTitan"
        ],
        
        # Balanced oval - Common ideal
        (1.28, 1.32): [
            "BalancedBeauty", "OvalOracle", "GeometricGem", "ProportionPro",
            "OvalOptimal", "NaturalNorm", "TimelessType", "ElegantEquation"
        ],
        
        # Rounded oval - Common
        (1.25, 1.28): [
            "CurveClassic", "RoundedRhythm", "CircleChic",
            "BubbleBoss", "CurvedCrown", "RoundedRuler",
            "OvalOriginal", "SoftwareSpecial", "FlowingForm"
        ],
        
        # Round dominant - Less common
        (1.2, 1.25): [
            "CircleChamp", "RoundRoyalty", "BubbleBliss", "SphereSensation",
            "OrbitalOriginal", "RoundReign", "CircularSage", "OrbitalOracle",
            "BubbleBoost", "CircularChic", "BubbleBeast", "SphereSupreme"
        ],
        
        # Very round - Rare
        (1.1, 1.2): [
            "MegaBubble", "UltraRound", "CircleExtreme", "RoundedRealm",
            "CircularCyborg", "BubbleBomb", "SphereSuperior", "CircleInfinity", 
            "CircularCosmos", "SphericalSpace", "BubbleUniverse", "RoundedReality"
        ]
    }
    
    for (min_val, max_val), labels in RATIO_LABELS.items():
        if min_val <= face_ratio < max_val:
            return random.choice(labels)
    
    return "GeometricGhost"

def greek_phi(image_path):
    """
    Takes an image, calculates golden ratio distances & angles,
    draws them on the image, and returns annotated image and metrics.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None, "No face detected."
    
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Key points
    points = {
        'mid_forehead': (int(landmarks[10].x * w), int(landmarks[10].y * h)),
        
        'eyebrow_mid': ((int(landmarks[8].x * w) + int(landmarks[9].x * w))//2,
                        (int(landmarks[8].y * h) + int(landmarks[9].y * h))//2),
        'eyebrow_left_inner': ((int(landmarks[55].x * w) + int(landmarks[107].x * w))//2, 
                               (int(landmarks[107].y * h) + int(landmarks[55].y *h))//2),
        'eyebrow_left_outer': ((int(landmarks[46].x * w) + int(landmarks[70].x * w))//2, 
                               (int(landmarks[46].y * h) + int(landmarks[70].y *h))//2),
        'eyebrow_left_mid': (int(landmarks[105].x * w), int(landmarks[105].y * h)),
        'eyebrow_right_inner': ((int(landmarks[285].x * w) + int(landmarks[336].x * w))//2, 
                                (int(landmarks[285].y * h) + int(landmarks[336].y *h))//2),
        'eyebrow_right_outer': ((int(landmarks[276].x * w) + int(landmarks[300].x * w))//2,
                                (int(landmarks[276].y * h) + int(landmarks[300].y *h))//2),
        'eyebrow_right_mid': (int(landmarks[334].x * w), int(landmarks[334].y * h)),

        'left_eye_outer': (int(landmarks[33].x * w), int(landmarks[33].y * h)),
        'left_eye_inner': (int(landmarks[133].x * w), int(landmarks[133].y * h)),
        'left_pupil': (int(landmarks[468].x * w), int(landmarks[468].y * h)),
        'right_eye_inner': (int(landmarks[362].x * w), int(landmarks[362].y * h)),
        'right_eye_outer': (int(landmarks[263].x * w), int(landmarks[263].y * h)),
        'right_pupil': (int(landmarks[473].x * w), int(landmarks[473].y * h)),
        
        'left_cheek': (int(landmarks[234].x * w), int(landmarks[234].y * h)),
        'right_cheek': (int(landmarks[454].x * w), int(landmarks[454].y * h)),

        'nose_left': (int(landmarks[48].x * w), int(landmarks[48].y * h)),
        'nose_right': (int(landmarks[278].x * w), int(landmarks[278].y * h)),
        'nose_bridge': ((int(landmarks[168].x * w) + int(landmarks[8].x * w))//2, 
                        (int(landmarks[168].y * h) + int(landmarks[8].y *h))//2),
        'nose_tip': (int(landmarks[1].x * w), int(landmarks[1].y * h)),
        'nose_base': (int(landmarks[2].x * w), int(landmarks[2].y * h)),

        'jaw_left': (int(landmarks[172].x * w), int(landmarks[172].y * h)),
        'jaw_right': (int(landmarks[397].x * w), int(landmarks[397].y * h)),

        'mouth_left': (int(landmarks[61].x * w), int(landmarks[61].y * h)),
        'mouth_right': (int(landmarks[291].x * w), int(landmarks[291].y * h)),
        
        'upper_lip_top': (int(landmarks[0].x * w), int(landmarks[0].y * h)),
        'upper_lip_bottom': (int(landmarks[13].x * w), int(landmarks[13].y * h)),
        'lower_lip_top': (int(landmarks[14].x * w), int(landmarks[14].y * h)),
        'lower_lip_bottom': (int(landmarks[17].x * w), int(landmarks[17].y * h)),

        'chin': (int(landmarks[152].x * w), int(landmarks[152].y * h))
    }
    
    # Approximate full forehead top
    forehead_height = points['mid_forehead'][1] - points['eyebrow_mid'][1]
    full_forehead_top_y = max(points['mid_forehead'][1] + int(0.67*forehead_height), 0)
    points['forehead_top'] = (points['mid_forehead'][0], full_forehead_top_y)
    
    # Distances
    face_length = calculate_distance(points['forehead_top'], points['chin'])
    face_width = calculate_distance(points['left_cheek'], points['right_cheek'])
    eyebrow_highest = (
        (points['eyebrow_left_mid'][0] + points['eyebrow_right_mid'][0]) // 2,
        (points['eyebrow_left_mid'][1] + points['eyebrow_right_mid'][1]) // 2
    )
    forehead_length = calculate_distance(points['forehead_top'], points['eyebrow_mid'])

    left_eye_width = calculate_distance(points['left_eye_inner'], points['left_eye_outer'])
    right_eye_width = calculate_distance(points['right_eye_inner'], points['right_eye_outer'])
    eye_width = ((left_eye_width + right_eye_width) / 2)*1.1  # slight adjustment for accuracy
    eye_dist = calculate_distance(points['left_eye_inner'], points['right_eye_inner'])
    interocular_distance = calculate_distance(points['left_pupil'], points['right_pupil'])
    
    nose_width = calculate_distance(points['nose_left'], points['nose_right'])
    nose_length = calculate_distance(points['nose_bridge'], points['nose_tip'])
    face2 = calculate_distance(points['eyebrow_mid'], points['nose_bridge']) + nose_length

    mouth_width = calculate_distance(points['mouth_left'], points['mouth_right'])

    nose_upper_lip = calculate_distance(points['nose_base'], points['upper_lip_top'])
    upper_lip_height = calculate_distance(points['upper_lip_top'], points['upper_lip_bottom'])
    lower_lip_height = calculate_distance(points['lower_lip_top'], points['lower_lip_bottom'])
    lower_lip_chin = calculate_distance(points['lower_lip_bottom'], points['chin'])

    # Angles
    jaw_angle = calculate_angle(points['jaw_left'], points['chin'], points['jaw_right'])
    eyebrow_left_angle = 180-calculate_angle(points['eyebrow_left_outer'], points['eyebrow_left_mid'], points['eyebrow_left_inner'])
    eyebrow_right_angle = 180-calculate_angle(points['eyebrow_right_outer'], points['eyebrow_right_mid'], points['eyebrow_right_inner'])

    # Symmetry (distance of key points from midline)
    midline_pt1 = points['chin']
    midline_pt2 = points['forehead_top']

    symmetry_points = [
        points['left_eye_inner'], points['right_eye_inner'],
        points['mouth_left'], points['mouth_right']
    ]

    distances = [point_line_distance(pt, midline_pt1, midline_pt2) for pt in symmetry_points]
    symmetry_deviation = np.std(distances) / face_width * 100  # percent deviation

    # Ratios
    ratios = {
        'Face length / Face width': (f"{face_length / face_width:.3f}", 1.618),
        'Mouth width / Nose width': (f"{mouth_width / nose_width:.3f}", 1.618),
        'Interocular dist. / Nose width': (f"{interocular_distance / nose_width:.3f}", 1.618),
        'Lower lip / Upper lip': (f"{lower_lip_height / upper_lip_height:.3f}", 1.618),
        'Eye dist / Eye width': (f"{eye_dist / eye_width:.3f}", 1.000),

        'Mouth width / Face width': (f"{mouth_width / face_width:.3f}", 0.38),
        'Forehead length / Face length': (f"{forehead_length / face_length:.3f}", 0.333),
        'Eyebrow to nose / Face length': (f"{face2 / face_length:.3f}", 0.333),

        'Philtrum to chin': (f"{nose_upper_lip / lower_lip_chin if lower_lip_chin else 0:.3f}", 0.5),
        'Eye width / Face width': (f"{eye_width / face_width if face_width else 0:.3f}", 0.25),
        'Nose width / Face width': (f"{nose_width / face_width if face_width else 0:.3f}", 0.2),

        'Jaw angle (degrees)': (f"{jaw_angle:.3f}", 125),
        'Left eyebrow angle (degrees)': (f"{eyebrow_left_angle:.3f}", 45),
        'Right eyebrow angle (degrees)': (f"{eyebrow_right_angle:.3f}", 45),

        'Midline symmetry deviation (%)': (f"{symmetry_deviation:.3f}", "<5")
    }

    greek_score = compute_beauty_score(ratios)
    face_ratio = float(ratios['Face length / Face width'][0])

    # Draw points and distances on the image
    annotated_img = plot_on_image(img.copy(), points)
    return annotated_img, ratios, greek_score, face_ratio

# Example usage
if __name__ == "__main__":
    img_path = r"test_images\nik3.jpg"
    output_path="annotated_face.jpg"
    annotated_img, ratios, greek_score, face_ratio = greek_phi(img_path)
    comment = comment_on(face_ratio)

    # Print ratios and angles
    print("-" * 60)
    print(f"{'Facial Ratios & Angles':<35} {'Value':<15} {'Ideal':<15}")
    print("-" * 60)
    for key, (value, ideal) in ratios.items():
        print(f"{key:<35} {value:<15} {ideal:<15}")

    print("\n-----------------------------------------")
    print(f"💫 Face Ratio (Length/Width): {face_ratio:.3f} 💫")
    print(f"✨ Geometric Harmony Index: {greek_score}%!!! ✨")
    print(f"\n🎭 Face Type: #{comment} 🎭")
    print("-----------------------------------------\n")

    if annotated_img is not None:
        cv2.imwrite(output_path, annotated_img)
        print("Annotated image saved as 'annotated_face.jpg'.")
    else:
        print("No face detected in the image.")