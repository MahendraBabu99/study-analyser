import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
drawing_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# Define eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """
    Calculate Eye Aspect Ratio (EAR)
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    # Get coordinates for the eye landmarks
    points = []
    for index in eye_indices:
        point = landmarks.landmark[index]
        x = int(point.x * img_w)
        y = int(point.y * img_h)
        points.append((x, y))
    
    # Calculate vertical distances
    vertical1 = math.sqrt((points[1][0] - points[5][0])**2 + (points[1][1] - points[5][1])**2)
    vertical2 = math.sqrt((points[2][0] - points[4][0])**2 + (points[2][1] - points[4][1])**2)
    
    # Calculate horizontal distance
    horizontal = math.sqrt((points[0][0] - points[3][0])**2 + (points[0][1] - points[3][1])**2)
    
    # Calculate EAR
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear, points

def main():
    # EAR threshold for blink detection (adjust based on testing)
    EAR_THRESHOLD = 0.25
    frame_count = 0
    blink_count = 0
    closed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face.process(frame_rgb)
        
        h, w, _ = frame.shape
        
        if result.multi_face_landmarks:
            for lm in result.multi_face_landmarks:
                # Calculate EAR for both eyes
                left_ear, left_points = calculate_ear(lm, LEFT_EYE, w, h)
                right_ear, right_points = calculate_ear(lm, RIGHT_EYE, w, h)
                
                # Average EAR
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Detect if eyes are closed
                if avg_ear < EAR_THRESHOLD:
                    eye_status = "CLOSED"
                    eye_color = (0, 0, 255)  # Red
                    closed_frames += 1
                    
                    # Count blink if eyes were closed for 2-3 frames (avoid false positives)
                    if closed_frames == 3:
                        blink_count += 1
                else:
                    eye_status = "OPEN"
                    eye_color = (0, 255, 0)  # Green
                    closed_frames = 0
                
                # Draw eye landmarks
                for point in left_points + right_points:
                    cv2.circle(frame, point, 2, eye_color, -1)
                
                # Draw connections for better visualization
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=lm,
                    connections=mp_face.FACEMESH_LEFT_EYE,
                    connection_drawing_spec=drawing_spec
                )
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=lm,
                    connections=mp_face.FACEMESH_RIGHT_EYE,
                    connection_drawing_spec=drawing_spec
                )
                
                # Display information
                cv2.putText(frame, f"Eye Status: {eye_status}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
                cv2.putText(frame, f"Blinks: {blink_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw bounding box
                x_coords = [int(p.x * w) for p in lm.landmark]
                y_coords = [int(p.y * h) for p in lm.landmark]
                x_min = max(min(x_coords) - 20, 0)
                x_max = min(max(x_coords) + 20, w)
                y_min = max(min(y_coords) - 20, 0)
                y_max = min(max(y_coords) + 20, h)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        else:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        frame_count += 1
        cv2.imshow('Eye Blink Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()