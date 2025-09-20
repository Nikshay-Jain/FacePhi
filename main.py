import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Helper functions
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Core analysis function
def analyze_face(image):
    img = np.array(image.convert('RGB'))
    results = face_mesh.process(img)

    if not results.multi_face_landmarks:
        return image, "No face detected. Please try again."

    face_landmarks = results.multi_face_landmarks[0]

    h, w, _ = img.shape
    landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

    points = {
        'forehead': landmarks[10],
        'chin': landmarks[152],
        'left_cheek': landmarks[234],
        'right_cheek': landmarks[454],
        'left_eye_inner': landmarks[33],
        'left_eye_outer': landmarks[133],
        'right_eye_inner': landmarks[362],
        'right_eye_outer': landmarks[263],
        'nose_left': landmarks[93],
        'nose_right': landmarks[323],
        'mouth_left': landmarks[61],
        'mouth_right': landmarks[291]
    }

    face_length = calculate_distance(points['forehead'], points['chin'])
    face_width = calculate_distance(points['left_cheek'], points['right_cheek'])
    left_eye_width = calculate_distance(points['left_eye_inner'], points['left_eye_outer'])
    right_eye_width = calculate_distance(points['right_eye_inner'], points['right_eye_outer'])
    nose_width = calculate_distance(points['nose_left'], points['nose_right'])
    mouth_width = calculate_distance(points['mouth_left'], points['mouth_right'])

    jaw_angle = calculate_angle(points['left_cheek'], points['chin'], points['right_cheek'])

    phi = 1.618
    ratios = {
        'face_length_to_width': face_length / face_width,
        'eye_width_to_face_width': ((left_eye_width + right_eye_width)/2) / face_width,
        'nose_width_to_face_width': nose_width / face_width,
        'mouth_width_to_face_width': mouth_width / face_width
    }

    def ratio_score(ratio):
        return max(0, 100 - abs(ratio - phi)/phi*100)

    scores = [ratio_score(v) for v in ratios.values()]
    final_score = int(np.mean(scores))

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    for name, pt in points.items():
        draw.ellipse((pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2), fill=(255,0,0))

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    draw.text((10,10), f"FacePhi Score: {final_score}%", fill=(255,255,0), font=font)

    return pil_img, final_score

# Streamlit UI
st.set_page_config(page_title="FacePhi - Discover Your Golden Ratio", layout="centered")
st.title("FacePhi 🌟")
st.write("Upload your selfie or take one now to discover how closely your face matches the golden ratio and ideal aesthetic proportions!")

uploaded_file = st.file_uploader("Choose a selfie...", type=["jpg","jpeg","png"])
camera_image = st.camera_input("Or take a selfie")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif camera_image is not None:
    image = Image.open(camera_image)

if image is not None and st.button("Analyze Face"):
    with st.spinner('Analyzing...'):
        analyzed_img, score_or_msg = analyze_face(image)
        # If no face detected, score_or_msg is a string
        if isinstance(score_or_msg, str):
            st.error(score_or_msg)
        else:
            # Calculate and display ratios
            img = np.array(image.convert('RGB'))
            results = face_mesh.process(img)
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = img.shape
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]
            points = {
                'forehead': landmarks[10],
                'chin': landmarks[152],
                'left_cheek': landmarks[234],
                'right_cheek': landmarks[454],
                'left_eye_inner': landmarks[33],
                'left_eye_outer': landmarks[133],
                'right_eye_inner': landmarks[362],
                'right_eye_outer': landmarks[263],
                'nose_left': landmarks[93],
                'nose_right': landmarks[323],
                'mouth_left': landmarks[61],
                'mouth_right': landmarks[291]
            }
            face_length = calculate_distance(points['forehead'], points['chin'])
            face_width = calculate_distance(points['left_cheek'], points['right_cheek'])
            left_eye_width = calculate_distance(points['left_eye_inner'], points['left_eye_outer'])
            right_eye_width = calculate_distance(points['right_eye_inner'], points['right_eye_outer'])
            nose_width = calculate_distance(points['nose_left'], points['nose_right'])
            mouth_width = calculate_distance(points['mouth_left'], points['mouth_right'])
            ratios = {
                'Face Length / Width': round(face_length / face_width, 3),
                'Avg Eye Width / Face Width': round(((left_eye_width + right_eye_width)/2) / face_width, 3),
                'Nose Width / Face Width': round(nose_width / face_width, 3),
                'Mouth Width / Face Width': round(mouth_width / face_width, 3)
            }
            st.image(analyzed_img, caption='Analyzed Face', width='stretch')
            st.subheader("Your Facial Ratios")
            for k, v in ratios.items():
                st.write(f"**{k}:** {v}")
            st.success(f'Your FacePhi Score is {score_or_msg}%')