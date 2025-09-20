# Golden Ratio Face Analyzer 🌟

A playful, interactive Python web app that analyzes facial proportions based on the **Golden Ratio** and other beauty standards (90s Hollywood, K-Pop, AI Future Beauty). The app provides instant visual feedback, era-based comparisons, symmetry analysis, and shareable results to maximize user curiosity, engagement, and virality.

---

## Features 🎯

### Core Features

1. **Golden Ratio Analysis**

   * Detects facial landmarks using **Mediapipe Face Mesh** (468 points).
   * Crops the face area using Mediapipe bounding box to standardize the region for analysis.
   * Calculates distances and ratios between key points to evaluate adherence to the golden ratio (φ = 1.618).
   * Key points and distances include:

     * **Face length**: top of forehead (landmark 10) to chin (landmark 152)
     * **Face width**: distance between left cheek (landmark 234) and right cheek (landmark 454)
     * **Eye width**: distance between inner and outer corners of left/right eyes (landmarks 33–133 left, 362–263 right)
     * **Interocular distance**: distance between pupils (landmarks 468–473)
     * **Nose width**: distance between left and right alar points (landmarks 93–323)
     * **Lips width**: distance between left and right mouth corners (landmarks 61–291)
   * Angles calculated for symmetry and jawline:

     * Jawline angle: angle between left jaw (landmark 234), chin (152), right jaw (454)
     * Nose angle: tip (1) relative to nostril points (93 & 323)
     * Eye tilt angle: inner vs outer eye corners
   * **Final Score Calculation**:

     * Compute the deviation of each measured ratio from the ideal golden ratio or predefined era ratio.
     * Normalize deviations to a 0–100% scale per metric.
     * Average all relevant metric scores for final **Golden Ratio Score**.

2. **Era-Based Beauty Comparison**

   * Greek Golden Ratio: classic φ-based ratios.
   * 90s Hollywood: jawline angle, eye width, nose width within celebrity-inspired ranges.
   * K-Pop Idol: V-line jaw, eye-to-face ratio, small nose constraints.
   * AI Future Beauty: exaggerated hyper-symmetry, idealized ratios.
   * Carousel or tabbed interface to view all eras in one place.

3. **Face Symmetry Mirror**

   * Splits the face vertically and mirrors each half.
   * Interactive slider to compare left vs. right symmetry.
   * Symmetry score calculated by measuring average deviation of left landmarks vs mirrored right landmarks.

4. **Optional “What If” Filters**

   * Cartoonify: enlarge eyes, smooth skin.
   * Greek Statue: grayscale stone texture.
   * K-Pop Idol: smooth, bright skin, playful styling.
   * Lightweight visual filters, no ML needed.

5. **Shareable Result Cards**

   * Generates **image cards** with:

     * User photo
     * Overlay masks and landmark highlights
     * Era scores
     * Playful captions (e.g., “Certified 82% Greek God ✨”)
   * Optimized for **social sharing** (Twitter, Instagram, TikTok).

6. **Optional Gamification**

   * Daily streak: track scores over multiple visits.
   * Leaderboard: anonymous top scores for friendly competition.
   * Encourages repeat visits and engagement.

---

## Tech Stack 🛠️

* **Frontend / UI**:

  * Python frameworks: **Gradio** (fastest for MVP) or **Streamlit** (interactive tabs, sliders)
  * Optional: React + Flask/FastAPI for full UI customization

* **Face Landmark Detection**:

  * **Mediapipe Face Mesh** (468 landmarks, detects face, crops region, provides (x,y,z) coordinates)

* **Backend / Optional Features**:

  * Python functions for ratio/angle calculations
  * Optional storage for leaderboards/streaks: **Firebase**, **Supabase**

* **Deployment Platforms**:

  * MVP: **Hugging Face Spaces** (Gradio) or **Streamlit Cloud**
  * Polished version: **Vercel (frontend) + Railway / Render (Python backend)**

* **Libraries**:

  ```text
  mediapipe
  opencv-python
  numpy
  gradio / streamlit
  pillow (for image processing)
  matplotlib (optional for overlays)
  firebase-admin or supabase-py (optional leaderboard)
  ```

---

## Architecture / Flow 📊

1. **User Uploads Image**

   * Webcam or file upload.
   * Mediapipe detects landmarks, crops face region, standardizes for analysis.

2. **Core Analysis**

   * Calculate distances between key points.
   * Compute angles (jawline, eyes, nose) for symmetry.
   * Derive ratios and compare to golden ratio or era-specific thresholds.
   * Normalize to 0–100% per metric, then average for final score.

3. **Visual Overlays**

   * Apply masks/grids for each era.
   * Highlight matched vs. deviated features.

4. **Optional Filters**

   * Cartoonify / Statue / K-Pop effects applied via lightweight image transformations.

5. **Result Generation**

   * Score display + era comparison carousel.
   * Symmetry mirror view.
   * Generate **shareable result card**.

6. **Engagement Features**

   * Daily streak / leaderboard (optional)
   * Pre-filled captions for social media sharing

---

## Implementation Notes 💡

* All scoring deterministic using landmark ratios and angles.
* Landmark coordinates → numpy arrays for ratio and angle calculations.
* Sliders, carousels, filters optional for clean UI.
* Tone playful & non-judgmental for virality.
* Export shareable cards as PNG using Pillow or OpenCV.

---

## Deployment Recommendations 🚀

**Most Engaging Free Deployment for Python App**:

| Option                       | Pros                                                                                | Cons                                               | Recommendation                                 |
| ---------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------- |
| Gradio + Hugging Face Spaces | Extremely fast to deploy, built-in shareable links, supports image uploads, sliders | Minimal UI customization                           | **Best for MVP and shareable demos**           |
| Streamlit Cloud              | Interactive tabs/sliders, Python-only, free tier                                    | Slightly slower, less social link integration      | Good for slightly polished interactive MVP     |
| React + Flask/FastAPI        | Full control of UI, advanced visuals, custom shareable cards                        | Needs more coding, backend setup, free tier limits | Best for full polished version but more effort |

✅ **Recommendation**: Use **Gradio + Hugging Face Spaces** for MVP → later expand to **Streamlit or React + Flask/FastAPI** if more advanced visuals or leaderboards needed.

---

## UX / Engagement Principles 🧩

* Immediate feedback: upload → instant score & overlay.
* Interactive visuals: symmetry slider, era carousel.
* Optional extras: playful filters, streaks, leaderboards.
* Shareability first: screenshot-friendly results with captions.
* Playful tone: avoids clinical judgment → more viral.

---

## Future Extensions 🌈

* Animated overlays (landmark lines drawing dynamically)
* Augmented reality real-time selfie camera
* Friend comparison / challenges
* More “beauty standards” (historical or cultural)
* Multilingual captions for global engagement

---

## References 📚

* [Mediapipe Face Mesh Documentation](https://developers.google.com/mediapipe/solutions/vision/face_mesh)
* [Gradio Documentation](https://gradio.app/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* Golden Ratio theory in aesthetics

---