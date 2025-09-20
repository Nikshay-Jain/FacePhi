import cv2
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

def draw_golden_spiral(img, center, scale, color, thickness, turns):
    phi = (1 + 5 ** 0.5) / 2
    points = []
    for t in np.linspace(0, turns * 2 * np.pi, 300):
        r = scale * (phi ** (t / (np.pi/2)) - 1)  # subtract 1 so r=0 at t=0
        x = int(center[0] - r * np.sin(t))
        y = int(center[1] + r * np.cos(t))
        points.append((x, y))
    for i in range(1, len(points)):
        cv2.line(img, points[i-1], points[i], color, thickness)
    return img
    
def greek_phi(image_path, output_path="output.jpg"):
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
    forehead_length = calculate_distance(points['forehead_top'], points['nose_bridge'])

    left_eye_width = calculate_distance(points['left_eye_inner'], points['left_eye_outer'])
    right_eye_width = calculate_distance(points['right_eye_inner'], points['right_eye_outer'])
    eye_width = (left_eye_width + right_eye_width) / 2
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

    # Horizontal symmetry: distances from midline (x of chin) for eyes, nose, mouth
    mid_x = points['chin'][0]
    horizontal_symmetry = np.mean([
        abs(points['left_eye_inner'][0] - mid_x) - abs(points['right_eye_inner'][0] - mid_x),
        abs(points['nose_left'][0] - mid_x) - abs(points['nose_right'][0] - mid_x),
        abs(points['mouth_left'][0] - mid_x) - abs(points['mouth_right'][0] - mid_x)
    ])
    horizontal_symmetry = abs(horizontal_symmetry) / face_width * 100  # percent deviation
    
    vh = (forehead_length / face_length if face_length else 0,
        face2 / face_length if face_length else 0,
        (face_length - forehead_length - face2) / face_length if face_length else 0)
    vertical_harmony = f"{vh[0]/vh[1]:.3f} : {1:.3f} : {vh[2]/vh[1]:.3f}"

    # Ratios
    ratios = {
        'Vertical harmony (Forehead: Nose: Chin)': (vertical_harmony, "1:1:1"),
        'Face length / Face width': (f"{face_length / face_width:.3f}", 1.618),

        'Mouth width / Face width': (f"{mouth_width / face_width:.3f}", 0.38),
        'Nose width / Mouth width': (f"{nose_width / mouth_width:.3f}", 0.618),
        
        'Interocular distance / Nose width': (f"{interocular_distance / nose_width:.3f}", 1.618),
        
        'Eye width / Face width': (f"{eye_width / face_width if face_width else 0:.3f}", 0.25),
        'Nose width / Face width': (f"{nose_width / face_width if face_width else 0:.3f}", 0.2),
        
        'Upper lip / Lower lip ratio': (f"{upper_lip_height / lower_lip_height if lower_lip_height else 0:.3f}", 0.618),
        'Nose base to upper lip / Lower lip to chin': (f"{nose_upper_lip / lower_lip_chin if lower_lip_chin else 0:.3f}", 0.5),

        'Jaw angle (degrees)': (f"{jaw_angle:.3f}", "120-130"),
        'Left eyebrow angle (degrees)': (f"{eyebrow_left_angle:.3f}", 45),
        'Right eyebrow angle (degrees)': (f"{eyebrow_right_angle:.3f}", 45),

        'Symmetry deviation (%)': (f"{horizontal_symmetry:.3f}", "<5%")
    }
    
    # Draw points and distances on the image
    annotated_img = img.copy()
    
    # Print ratios and angles
    print("\n--- Facial Ratios & Angles ---")
    for key, (value, ideal) in ratios.items():
        print(f"{key}: {value} (ideal: {ideal})")
    
    # Draw all key points
    for key, (x, y) in points.items():
        cv2.circle(annotated_img, (x, y), 3, (0, 0, 255), -1)
        # cv2.putText(annotated_img, key, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

    # Draw main lines for ratios/distances
    line_specs = [
        # (start, end, color, label, label_offset)
        ('forehead_top', 'chin', (0,255,0), 'Face Length', (10, 0)),
        ('left_cheek', 'right_cheek', (0,255,0), 'Face Width', (0, -10)),
        ('forehead_top', 'nose_bridge', (0,128,255), 'Forehead Length', (10, 10)),
        ('nose_left', 'nose_right', (255,128,0), 'Nose Width', (0, 20)),
        ('nose_bridge', 'nose_tip', (128,0,255), 'Nose Length', (10, 10)),
        ('mouth_left', 'mouth_right', (0,255,128), 'Mouth Width', (0, 20)),
        ('upper_lip_top', 'upper_lip_bottom', (0,128,128), 'Upper Lip', (10, 10)),
        ('lower_lip_top', 'lower_lip_bottom', (128,128,0), 'Lower Lip', (10, 10)),
        ('left_eye_inner', 'left_eye_outer', (128,255,255), 'Left Eye Width', (10, 10)),
        ('right_eye_inner', 'right_eye_outer', (128,255,255), 'Right Eye Width', (10, 10)),
        ('left_pupil', 'right_pupil', (255,0,128), 'Interocular', (10, 10)),
    ]
    for p1, p2, color, label, offset in line_specs:
        pt1, pt2 = points[p1], points[p2]
        cv2.line(annotated_img, pt1, pt2 ,color, 2)

    # Highlight jaw angle lines and put text
    jaw_color = (0, 0, 255)
    cv2.line(annotated_img, points['chin'], points['jaw_left'], jaw_color, 2)
    cv2.line(annotated_img, points['chin'], points['jaw_right'], jaw_color, 2)
    
    # Highlight eyebrow angle lines (left)
    eyebrow_color = (255, 0, 128)
    cv2.line(annotated_img, points['eyebrow_left_outer'], points['eyebrow_left_mid'], eyebrow_color, 2)
    cv2.line(annotated_img, points['eyebrow_left_inner'], points['eyebrow_left_mid'], eyebrow_color, 2)
    
    # Highlight eyebrow angle lines (right)
    cv2.line(annotated_img, points['eyebrow_right_outer'], points['eyebrow_right_mid'], eyebrow_color, 2)
    cv2.line(annotated_img, points['eyebrow_right_inner'], points['eyebrow_right_mid'], eyebrow_color, 2)
    
    # highlight symmetry lines
    mid_x = points['chin'][0]
    cv2.line(annotated_img, (mid_x, 0), (mid_x, h), (200,200,200), 1)
    cv2.putText(annotated_img, "Midline Symmetry", (mid_x+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    
    # Save annotated image
    cv2.imwrite(output_path, annotated_img)

    # Draw golden spiral overlay (centered at nose tip, scale based on face length)
    annotated_img = draw_golden_spiral(annotated_img, points['nose_tip'], scale=int(face_length/30), color=(0,215,255), thickness=2, turns=1.5)

    # Save annotated image (now includes spiral)
    cv2.imwrite(output_path, annotated_img)

    return annotated_img, ratios

# Example usage
if __name__ == "__main__":
    img_path = r"./nik1.jpg"
    annotated_img, ratios = greek_phi(img_path, output_path="annotated_face.jpg")
    if annotated_img is not None:
        print("\nAnnotated image saved as 'annotated_face.jpg'.")