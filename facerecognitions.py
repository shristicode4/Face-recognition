import cv2
import face_recognition

# Load known face encodings and names
known_face_encoding = []
known_face_names = []

# Load known faces and their names 
known_person1_image = face_recognition.load_image_file(r"C:\Users\user\Documents\Personal Pro\Face-recognition\Person1.jpg")
known_person2_image = face_recognition.load_image_file(r"C:\Users\user\Documents\Personal Pro\Face-recognition\Person2.jpg")
known_person3_image = face_recognition.load_image_file(r"C:\Users\user\Documents\Personal Pro\Face-recognition\Person3.jpg")

known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]
known_person3_encoding = face_recognition.face_encodings(known_person3_image)[0]

known_face_encoding.append(known_person1_encoding)
known_face_encoding.append(known_person2_encoding)
known_face_encoding.append(known_person3_encoding)

known_face_names.append("Shristika Adhikari")
known_face_names.append("Rohit Sharma")
known_face_names.append("Rohit Sharma")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = "unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # The resulting frame
    cv2.imshow("Video", frame)

    # Break the loop when the 'k' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

# Release the webcam and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
